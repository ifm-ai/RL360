"""Streaming safetensors I/O: read one tensor at a time, write bounded shards."""

from __future__ import annotations

import ctypes
import gc
import json
import math
import os
import re
import shutil
import uuid
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Sequence

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

INDEX_FILE = "model.safetensors.index.json"
PART_PATTERN = re.compile(r"part-(\d{5})\.safetensors")
FINAL_PART_PATTERN = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors")
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4,
    "F64": 8, "I64": 8, "U64": 8,
}
OUTPUT_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}

try:
    _malloc_trim = ctypes.CDLL(None).malloc_trim
    _malloc_trim.argtypes = [ctypes.c_size_t]
except AttributeError:
    _malloc_trim = None


class CorruptResumePart(Exception):
    pass


def release_unused_memory() -> None:
    """Return freed Python/PyTorch CPU allocations to the OS when possible."""
    gc.collect()
    if _malloc_trim is not None:
        _malloc_trim(0)


def parse_size(text: str) -> int:
    units = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3,
             "KB": 1000, "MB": 1000**2, "GB": 1000**3}
    value = text.strip()
    for suffix in sorted(units, key=len, reverse=True):
        if value.lower().endswith(suffix.lower()):
            return int(float(value[:-len(suffix)]) * units[suffix])
    return int(value)


def output_dtype(name: str, reference: torch.Tensor) -> torch.dtype:
    """Resolve ``--output-dtype``; ``same`` keeps the first input's dtype."""
    return reference.dtype if name == "same" else OUTPUT_DTYPES[name]


def write_json_atomic(path: Path, data: object) -> None:
    temporary = path.with_name(path.name + ".new")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def load_weight_map(model_dir: Path) -> OrderedDict[str, str]:
    index_path = model_dir / INDEX_FILE
    if index_path.exists():
        with index_path.open() as handle:
            weight_map = OrderedDict(json.load(handle)["weight_map"].items())
        for filename in set(weight_map.values()):
            # Shards must live in the checkpoint directory itself.
            if Path(filename).name != filename or filename in {"", ".", ".."}:
                raise ValueError(f"{index_path}: invalid shard filename {filename!r}")
        return weight_map

    single_path = model_dir / "model.safetensors"
    if single_path.exists():
        with safe_open(str(single_path), framework="pt", device="cpu") as handle:
            return OrderedDict((name, single_path.name) for name in sorted(handle.keys()))
    raise FileNotFoundError(f"No safetensors checkpoint found in {model_dir}")


def ordered_names(weight_map: OrderedDict[str, str]) -> list[str]:
    """Canonical merge order: by shard file, then by name, so reads stay sequential."""
    by_file: dict[str, list[str]] = defaultdict(list)
    for name, filename in weight_map.items():
        by_file[filename].append(name)
    return [name for filename in sorted(by_file)
            for name in sorted(by_file[filename])]


def slice_nbytes(handle, name: str) -> int:
    """Byte size of one tensor, read from an open safetensors header."""
    tensor = handle.get_slice(name)
    return math.prod(tensor.get_shape()) * DTYPE_BYTES[tensor.get_dtype()]


def tensor_nbytes(model_dir: Path, weight_map: OrderedDict[str, str]) -> dict[str, int]:
    """Exact per-tensor byte sizes, read from safetensors headers only."""
    sizes: dict[str, int] = {}
    for filename in sorted(set(weight_map.values())):
        with safe_open(str(model_dir / filename), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                sizes[name] = slice_nbytes(handle, name)
    return sizes


class TensorSource:
    """Read tensors while keeping only a few safetensors files open."""

    def __init__(self, model_dir: Path, cache_size: int = 1):
        self.model_dir = model_dir
        self.weight_map = load_weight_map(model_dir)
        self.cache_size = cache_size
        self._readers: OrderedDict[str, object] = OrderedDict()

    def get(self, name: str) -> torch.Tensor:
        filename = self.weight_map[name]
        if filename not in self._readers:
            while len(self._readers) >= self.cache_size:
                _, old_reader = self._readers.popitem(last=False)
                old_reader.__exit__(None, None, None)
            reader = safe_open(str(self.model_dir / filename), framework="pt", device="cpu")
            reader.__enter__()
            self._readers[filename] = reader
        self._readers.move_to_end(filename)
        return self._readers[filename].get_tensor(name)

    def close(self) -> None:
        for reader in self._readers.values():
            reader.__exit__(None, None, None)
        self._readers.clear()


def validate_sources(sources: Sequence[TensorSource]) -> None:
    expected = set(sources[0].weight_map)
    for source in sources[1:]:
        actual = set(source.weight_map)
        if actual != expected:
            raise ValueError(
                f"Tensor names differ in {source.model_dir}: "
                f"missing={len(expected - actual)}, extra={len(actual - expected)}"
            )


def copy_assets(source: Path, destination: Path, overwrite: bool = False) -> None:
    """Copy config, tokenizer and other non-weight files from a checkpoint.

    Symlinks are followed, so the output stays self-contained when the input
    is a Hugging Face cache snapshot or a directory of links.
    """
    for child in source.iterdir():
        if child.suffix in {".safetensors", ".bin"} or child.name.endswith(".index.json"):
            continue
        target = destination / child.name
        if overwrite and (target.is_symlink() or target.is_file()):
            target.unlink()
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=overwrite)
        else:
            shutil.copy2(child, target)


class ShardWriter:
    """Buffer merged tensors and flush ``part-NNNNN.safetensors`` files."""

    def __init__(self, output_dir: Path, max_size: int):
        self.output_dir = output_dir
        self.max_size = max_size
        self.tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.size = 0
        self.total_size = 0
        self.parts: list[tuple[str, list[str]]] = []

    def restore(self, parts: list[tuple[str, list[str]]], total_size: int) -> None:
        self.parts = parts
        self.total_size = total_size

    def add(self, name: str, tensor: torch.Tensor) -> None:
        size = tensor.numel() * tensor.element_size()
        if self.tensors and self.size + size > self.max_size:
            self.flush()
        self.tensors[name] = tensor
        self.size += size
        self.total_size += size

    def flush(self) -> None:
        if not self.tensors:
            return
        filename = f"part-{len(self.parts):05d}.safetensors"
        names = list(self.tensors)
        save_file(self.tensors, str(self.output_dir / filename), metadata={"format": "pt"})
        print(f"[write] {filename}: {len(names)} tensors, {self.size / 1e9:.2f} GB", flush=True)
        self.parts.append((filename, names))
        self.tensors = OrderedDict()
        self.size = 0
        release_unused_memory()

    def finish(self, names_in_order: Sequence[str]) -> None:
        """Rename parts to ``model-XXXXX-of-NNNNN`` and write the index."""
        self.flush()
        weight_map: dict[str, str] = {}
        count = len(self.parts)
        for index, (old_name, names) in enumerate(self.parts):
            new_name = f"model-{index:05d}-of-{count:05d}.safetensors"
            if old_name != new_name:
                os.replace(self.output_dir / old_name, self.output_dir / new_name)
            weight_map.update((name, new_name) for name in names)
        write_json_atomic(self.output_dir / INDEX_FILE, {
            "metadata": {"total_size": self.total_size},
            "weight_map": {name: weight_map[name] for name in names_in_order},
        })


def restore_parts(
    directory: Path,
    names_in_order: Sequence[str],
    reference_source: TensorSource,
    output_dtype_name: str,
) -> tuple[list[tuple[str, list[str]]], int, int]:
    """Verify existing parts and return ``(parts, total_size, tensors_done)``.

    Parts must hold the exact next prefix of ``names_in_order`` with the
    expected shapes and dtypes. A truncated trailing part (an interrupted
    write) is quarantined so the merge can resume from the previous part.
    Parts already renamed to their final names (an interrupted ``finish``)
    are accepted as long as their declared shard count is consistent.
    """
    indexed_by_number: dict[int, Path] = {}
    declared_counts: set[int] = set()
    for path in directory.glob("*.safetensors"):
        part_match = PART_PATTERN.fullmatch(path.name)
        final_match = FINAL_PART_PATTERN.fullmatch(path.name)
        if part_match:
            number = int(part_match.group(1))
        elif final_match:
            number = int(final_match.group(1))
            declared_counts.add(int(final_match.group(2)))
        else:
            raise ValueError(f"Unexpected part filename: {path}")
        if number in indexed_by_number:
            raise ValueError(f"Duplicate resume part number {number:05d}")
        indexed_by_number[number] = path
    indexed = sorted(indexed_by_number.items())
    if len(declared_counts) > 1:
        raise ValueError("Finalized resume shards disagree on shard count")
    if declared_counts and declared_counts != {len(indexed)}:
        raise ValueError("Finalized resume shard count does not match files present")
    if [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError("Resume parts are not contiguous from part-00000")

    parts: list[tuple[str, list[str]]] = []
    completed = 0
    total_size = 0
    for position, (_, path) in enumerate(indexed):
        try:
            handle = None
            entered = False
            try:
                try:
                    handle = safe_open(str(path), framework="pt", device="cpu")
                    handle.__enter__()
                    entered = True
                    shard_names = list(handle.keys())
                except SafetensorError as error:
                    raise CorruptResumePart(str(error)) from error
                expected = list(names_in_order[completed:completed + len(shard_names)])
                if not shard_names or set(shard_names) != set(expected):
                    raise ValueError(f"{path.name} is not the expected tensor-name prefix")
                shard_size = 0
                for name in expected:
                    try:
                        tensor = handle.get_tensor(name)
                    except SafetensorError as error:
                        raise CorruptResumePart(str(error)) from error
                    reference = reference_source.get(name)
                    expected_dtype = (
                        output_dtype(output_dtype_name, reference)
                        if torch.is_floating_point(reference) else reference.dtype
                    )
                    if tensor.shape != reference.shape:
                        raise ValueError(f"Shape mismatch in {path.name}: {name}")
                    if tensor.dtype != expected_dtype:
                        raise ValueError(
                            f"Dtype mismatch in {path.name}: {name}: "
                            f"expected {expected_dtype}, got {tensor.dtype}"
                        )
                    shard_size += tensor.numel() * tensor.element_size()
                    del tensor, reference
            finally:
                if entered:
                    handle.__exit__(None, None, None)
        except CorruptResumePart as error:
            if position != len(indexed) - 1:
                raise ValueError(f"Invalid non-trailing resume part: {path}") from error
            quarantine = path.with_name(f"{path.name}.resume-{uuid.uuid4().hex}.corrupt")
            print(f"[resume] quarantine corrupt trailing part: {path.name} -> "
                  f"{quarantine.name}: {error}", flush=True)
            os.replace(path, quarantine)
            break
        parts.append((path.name, expected))
        completed += len(expected)
        total_size += shard_size
    if completed > len(names_in_order):
        raise ValueError("Resume parts contain more tensors than the model")
    print(f"[resume] {len(parts)} parts, {completed}/{len(names_in_order)} tensors",
          flush=True)
    return parts, total_size, completed
