"""Drive a per-tensor merge rule over whole checkpoints.

``run_merge`` streams every tensor in one process. ``run_range`` processes
one byte-balanced slice of the tensor list so N workers can run in parallel,
and ``finalize`` stitches their parts into one checkpoint. Both paths write
into ``<output>.tmp`` and atomically rename it to ``<output>`` when complete.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from .checkpoint import (INDEX_FILE, PART_PATTERN, ShardWriter, TensorSource, copy_assets,
                         load_weight_map, ordered_names, output_dtype, parse_size,
                         restore_parts, slice_nbytes, tensor_nbytes,
                         validate_sources, write_json_atomic)

TensorMerger = Callable[[str, list[torch.Tensor]], torch.Tensor]
PROGRESS_FILE = "merge_progress.json"
RANGE_DONE = "RANGE_DONE"
RANGE_DIR = re.compile(r"range-(\d+)")


@dataclass
class MergeJob:
    """A merge rule plus its inputs and output settings.

    ``inputs[0]`` is the base for base-relative methods (the first expert for
    a linear soup). It defines the tensor order, the reference dtype for
    ``--output-dtype same``, and the config/tokenizer files that are copied.
    """

    inputs: Sequence[str | Path]
    merger: TensorMerger
    method_info: dict[str, object]
    output: str | Path
    output_dtype: str = "bfloat16"
    max_shard_size: str = "5GiB"
    progress_every: int = 100
    fetch_threads: int = 4
    paths: list[Path] = field(init=False)

    def __post_init__(self) -> None:
        self.paths = [Path(path).resolve() for path in self.inputs]
        self.output = Path(self.output).resolve()

    @property
    def temporary(self) -> Path:
        return self.output.with_name(self.output.name + ".tmp")

    def provenance(self, tensor_count: int) -> dict[str, object]:
        """Settings a resumed run must match exactly."""
        return {
            "version": 1,
            "method_info": self.method_info,
            "inputs": [str(path) for path in self.paths],
            "asset_dir": str(self.paths[0]),
            "output_dtype": self.output_dtype,
            "max_shard_size": parse_size(self.max_shard_size),
            "tensor_count": tensor_count,
        }

    def merge_info(self) -> dict[str, object]:
        return {**self.method_info, "inputs": [str(path) for path in self.paths]}


def _open_sources(job: MergeJob) -> tuple[list[TensorSource], list[str]]:
    sources = [TensorSource(path) for path in job.paths]
    validate_sources(sources)
    return sources, ordered_names(sources[0].weight_map)


def _merge_stream(
    job: MergeJob,
    sources: Sequence[TensorSource],
    names: Sequence[str],
    writer: ShardWriter,
    done: int,
    total: int,
) -> None:
    """Merge ``names`` in order, prefetching the next tensor while merging.

    Each input is read by one thread at a time; ``fetch_threads`` bounds how
    many inputs are read concurrently (keep it modest on shared filesystems).
    """
    fetch_pool = ThreadPoolExecutor(max_workers=max(1, min(len(sources), job.fetch_threads)))
    prefetcher = ThreadPoolExecutor(max_workers=1)

    def fetch(name: str) -> list[torch.Tensor]:
        return list(fetch_pool.map(lambda source: source.get(name), sources))

    try:
        with torch.no_grad():
            pending = prefetcher.submit(fetch, names[0]) if names else None
            for position, name in enumerate(names):
                tensors = pending.result()
                pending = (prefetcher.submit(fetch, names[position + 1])
                           if position + 1 < len(names) else None)
                if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
                    raise ValueError(f"Shape mismatch for {name}")
                merged = job.merger(name, tensors)
                if torch.is_floating_point(merged):
                    merged = merged.to(output_dtype(job.output_dtype, tensors[0]))
                writer.add(name, merged.cpu().contiguous())
                index = done + position + 1
                if index % job.progress_every == 0 or index == total:
                    print(f"[merge] {index}/{total} {name}", flush=True)
                del tensors, merged
    finally:
        prefetcher.shutdown()
        fetch_pool.shutdown()


def _close(sources: Sequence[TensorSource]) -> None:
    for source in sources:
        source.close()


def run_merge(
    job: MergeJob,
    *,
    force: bool = False,
    resume: bool = False,
    resume_legacy: bool = False,
    dry_run: bool = False,
) -> None:
    """Merge every tensor in one process.

    ``resume`` continues a matching ``<output>.tmp`` after verifying the saved
    provenance and every completed part. ``resume_legacy`` additionally
    accepts a partial directory written before progress manifests existed.
    """
    if resume_legacy and not resume:
        raise ValueError("--resume-legacy requires --resume")
    sources, names = _open_sources(job)
    try:
        print(f"method={job.method_info['method']}")
        print(f"inputs={len(job.paths)} tensors={len(names)}")
        print(f"output={job.output}")
        if dry_run:
            return

        output, temporary = job.output, job.temporary
        provenance = job.provenance(len(names))
        restored_parts: list[tuple[str, list[str]]] = []
        restored_size = completed = 0
        if resume:
            if output.exists():
                _check_finished_output(output, temporary, job.merge_info(), names)
                return
            if not temporary.is_dir():
                raise FileNotFoundError(f"Resume directory does not exist: {temporary}")
            if _range_dirs(temporary):
                raise ValueError(f"{temporary} holds a range-parallel merge; "
                                 "rerun its range workers or `finalize` it")
            progress_path = temporary / PROGRESS_FILE
            if progress_path.exists():
                if json.loads(progress_path.read_text()) != provenance:
                    raise ValueError(f"Resume configuration differs from {progress_path}")
            else:
                _check_legacy_resume(temporary, resume_legacy)
            restored_parts, restored_size, completed = restore_parts(
                temporary, names, sources[0], job.output_dtype)
            if not progress_path.exists():
                write_json_atomic(progress_path, provenance)
        else:
            for path in (output, temporary):
                if path.exists():
                    if not force:
                        raise FileExistsError(f"{path} exists; pass --force to replace it")
                    shutil.rmtree(path)
            temporary.mkdir(parents=True)
            copy_assets(job.paths[0], temporary)
            write_json_atomic(temporary / PROGRESS_FILE, provenance)

        writer = ShardWriter(temporary, parse_size(job.max_shard_size))
        writer.restore(restored_parts, restored_size)
        _merge_stream(job, sources, names[completed:], writer, completed, len(names))
        writer.finish(names)
        write_json_atomic(temporary / "merge_info.json", job.merge_info())
        for quarantine in temporary.glob("*.safetensors.resume-*.corrupt"):
            quarantine.unlink()
        os.replace(temporary, output)
        (output / PROGRESS_FILE).unlink()
        print(f"done: {output}")
    finally:
        _close(sources)


def _check_finished_output(output: Path, temporary: Path, merge_info: dict,
                           names: Sequence[str]) -> None:
    if temporary.exists():
        raise FileExistsError(f"Both final and temporary outputs exist: {output}, {temporary}")
    if json.loads((output / "merge_info.json").read_text()) != merge_info:
        raise ValueError("Existing final output has different merge provenance")
    weight_map = json.loads((output / INDEX_FILE).read_text()).get("weight_map", {})
    if set(weight_map) != set(names):
        raise ValueError("Existing final output has different tensor names")
    (output / PROGRESS_FILE).unlink(missing_ok=True)
    print(f"[resume] final output already assembled: {output}", flush=True)


def _check_legacy_resume(temporary: Path, resume_legacy: bool) -> None:
    if not resume_legacy:
        raise ValueError(
            f"{temporary} has no {PROGRESS_FILE}; "
            "pass --resume-legacy after verifying its provenance"
        )
    final_files = [temporary / INDEX_FILE, temporary / "merge_info.json"]
    if any(path.exists() for path in final_files) or any(
        temporary.glob("model-*-of-*.safetensors")
    ):
        raise ValueError("Legacy resume directory contains final-output files")
    print("[resume] accepting legacy directory without progress manifest", flush=True)


def split_ranges(costs: Sequence[int], num_splits: int) -> list[tuple[int, int]]:
    """Cut an ordered cost list into ``num_splits`` contiguous ranges of
    roughly equal total cost. Trailing ranges may be empty."""
    total = sum(costs)
    bounds = [0]
    accumulated = 0
    for index, cost in enumerate(costs):
        accumulated += cost
        if len(bounds) < num_splits and accumulated >= len(bounds) * total / num_splits:
            bounds.append(index + 1)
    bounds += [len(costs)] * (num_splits + 1 - len(bounds))
    return list(zip(bounds, bounds[1:]))


def run_range(job: MergeJob, num_splits: int, split_id: int, dry_run: bool = False) -> None:
    """Merge one byte-balanced slice of the tensor list.

    Writes parts to ``<output>.tmp/range-XX/`` and a ``RANGE_DONE`` marker.
    Re-running the same split resumes it; a different configuration is
    rejected. Run ``finalize`` once every split is done.
    """
    if not 0 <= split_id < num_splits:
        raise ValueError(f"--split-id must be in [0, {num_splits})")
    sources, names = _open_sources(job)
    try:
        sizes = tensor_nbytes(job.paths[0], sources[0].weight_map)
        lo, hi = split_ranges([sizes[name] for name in names], num_splits)[split_id]
        range_names = names[lo:hi]
        print(f"split {split_id}/{num_splits}: tensors [{lo}:{hi}) ({len(range_names)})",
              flush=True)
        if dry_run:
            return

        if job.output.exists():
            raise FileExistsError(f"{job.output} exists; remove it before a range-parallel merge")
        if (job.temporary / PROGRESS_FILE).exists() or any(job.temporary.glob("part-*.safetensors")):
            raise ValueError(f"{job.temporary} holds a single-process merge; "
                             "resume it without --num-splits or remove it")
        range_dir = job.temporary / f"range-{split_id:02d}"
        range_dir.mkdir(parents=True, exist_ok=True)
        provenance = {**job.provenance(len(names)), "num_splits": num_splits,
                      "split_id": split_id, "range": [lo, hi]}
        progress_path = range_dir / PROGRESS_FILE
        if progress_path.exists():
            if json.loads(progress_path.read_text()) != provenance:
                raise ValueError(f"Range configuration differs from {progress_path}")
        elif any(range_dir.glob("*.safetensors")) or (range_dir / RANGE_DONE).exists():
            raise ValueError(f"{range_dir} has results but no {PROGRESS_FILE}, so its "
                             "settings are unknown; delete it to recompute this range")
        else:
            write_json_atomic(progress_path, provenance)
        if (range_dir / RANGE_DONE).exists():
            print("range already complete", flush=True)
            return

        parts, total_size, completed = restore_parts(
            range_dir, range_names, sources[0], job.output_dtype)
        writer = ShardWriter(range_dir, parse_size(job.max_shard_size))
        writer.restore(parts, total_size)
        _merge_stream(job, sources, range_names[completed:], writer, completed,
                      len(range_names))
        writer.flush()
        write_json_atomic(range_dir / RANGE_DONE, {
            **provenance, "merge_info": job.merge_info(),
            "total_size": writer.total_size, "part_names": writer.parts,
        })
        print(f"range {split_id} done: {len(writer.parts)} parts, "
              f"{writer.total_size / 1e9:.1f} GB", flush=True)
    finally:
        _close(sources)


def finalize(output: str | Path) -> None:
    """Combine finished ranges into one checkpoint and promote it.

    Renumbers parts as ``model-XXXXX-of-NNNNN.safetensors``, writes the index,
    copies config/tokenizer files from ``inputs[0]``, records
    ``merge_info.json``, and renames ``<output>.tmp`` to ``<output>``.
    Safe to re-run after an interruption.
    """
    output = Path(output).resolve()
    if output.name.endswith(".tmp"):
        raise ValueError(f"Pass the final output path, not its .tmp directory: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if output.exists():
        _remove_promoted_range_dirs(output)
        return

    markers = _load_range_markers(temporary)
    shared = _shared_settings(markers)
    part_count = sum(len(marker["part_names"]) for _, marker in markers)
    final_names = {f"model-{index:05d}-of-{part_count:05d}.safetensors"
                   for index in range(part_count)}
    stray = {path.name for path in temporary.glob("*.safetensors")} - final_names
    if stray or (temporary / PROGRESS_FILE).exists():
        raise ValueError(f"{temporary} also holds files from a single-process merge: "
                         f"{sorted(stray)[:5] or PROGRESS_FILE}")
    weight_map: dict[str, str] = {}
    total_size = 0
    part_index = 0
    for range_dir, marker in markers:
        for old_name, names in marker["part_names"]:
            if not PART_PATTERN.fullmatch(old_name):
                raise ValueError(f"{range_dir / RANGE_DONE}: invalid part name {old_name!r}")
            new_name = f"model-{part_index:05d}-of-{part_count:05d}.safetensors"
            source, target = range_dir / old_name, temporary / new_name
            if source.exists():
                os.replace(source, target)
            elif not target.exists():
                raise FileNotFoundError(f"Missing part {source} (and {target})")
            total_size += _check_part(target, names)
            weight_map.update((name, new_name) for name in names)
            part_index += 1

    base = Path(shared["asset_dir"])
    expected = ordered_names(load_weight_map(base))
    if list(weight_map) != expected:
        missing = set(expected) - set(weight_map)
        extra = set(weight_map) - set(expected)
        raise ValueError(f"Tensor list mismatch: missing={len(missing)} extra={len(extra)}")

    write_json_atomic(temporary / INDEX_FILE, {
        "metadata": {"total_size": total_size}, "weight_map": weight_map})
    copy_assets(base, temporary, overwrite=True)
    write_json_atomic(temporary / "merge_info.json", shared["merge_info"])
    os.replace(temporary, output)
    for range_dir, _ in markers:
        shutil.rmtree(output / range_dir.name)
    print(f"finalized: {output} ({part_count} shards, {total_size / 1e9:.1f} GB)")


def _range_dirs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("range-*")
                  if path.is_dir() and RANGE_DIR.fullmatch(path.name))


def _remove_promoted_range_dirs(output: Path) -> None:
    """Finish a ``finalize`` that was interrupted after its rename."""
    finished = (output / INDEX_FILE).exists() and (output / "merge_info.json").exists()
    leftovers = _range_dirs(output)
    if not finished or not leftovers:
        raise FileExistsError(output)
    if any(any(path.glob("*.safetensors")) for path in leftovers):
        raise ValueError(f"{output} has range dirs that still hold parts")
    for path in leftovers:
        shutil.rmtree(path)
    print(f"finalized: {output} (removed {len(leftovers)} leftover range dirs)")


def _load_range_markers(temporary: Path) -> list[tuple[Path, dict]]:
    range_dirs = _range_dirs(temporary)
    if not range_dirs:
        raise FileNotFoundError(f"No range directories under {temporary}")
    markers = []
    for range_dir in range_dirs:
        marker = range_dir / RANGE_DONE
        if not marker.exists():
            raise FileNotFoundError(f"Range not finished: {marker}")
        markers.append((range_dir, json.loads(marker.read_text())))
    num_splits = markers[0][1]["num_splits"]
    if [marker["split_id"] for _, marker in markers] != list(range(num_splits)):
        raise FileNotFoundError(
            f"Expected ranges 0..{num_splits - 1}, found "
            f"{[marker['split_id'] for _, marker in markers]}")
    return markers


def _shared_settings(markers: Sequence[tuple[Path, dict]]) -> dict:
    """Settings every range must agree on (everything but its own slice)."""
    per_range = {"split_id", "range", "total_size", "part_names"}
    settings = [{key: value for key, value in marker.items() if key not in per_range}
                for _, marker in markers]
    if any(setting != settings[0] for setting in settings[1:]):
        raise ValueError("Ranges were produced with different merge settings")
    bounds = [marker["range"] for _, marker in markers]
    if any(left[1] != right[0] for left, right in zip(bounds, bounds[1:])):
        raise ValueError(f"Ranges are not contiguous: {bounds}")
    return settings[0]


def _check_part(path: Path, names: Sequence[str]) -> int:
    """Verify a part holds exactly ``names``; return its tensor bytes."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(names):
            raise ValueError(f"{path.name}: tensor names do not match its range marker")
        return sum(slice_nbytes(handle, name) for name in names)
