"""Rebuild the K2-Horizon 375B RL merge (HF tag ``rl_merged``) from its five RL experts.

The merged checkpoint is already on Hugging Face; this module is the recipe behind it, for
reproducing it or adapting it to your own experts. It downloads the base (``mid_4_6000``) and
the five RL experts of ``IFM/K2-Horizon-375B-A23B`` and runs ``merge360 iso`` with the release
settings: ISO on ``self_attn`` and ``shared_experts`` matrices, RAM everywhere else, uniform
weights, BF16 output. On layers 0 and 30 the result equals ``rl_merged`` except for a few
elements per million (relative error about 1e-8); the full model has not been compared.
``--layers`` merges only some layers (one shard each) as a cheap trial run. Flags this module
does not know, such as ``--device``, ``--num-splits``, ``--split-id`` and ``--resume``, are
passed to ``merge360 iso``, and a later flag overrides the recipe's (for example ``--keep``).

    # Environment: a CPU node is enough, since ISO runs its SVDs on the CPU by default
    # (add --device cuda to use a GPU). Needs torch, safetensors and huggingface_hub; the
    # repository is public, so no token is required. For example:
    #   srun --cpus-per-task=32 --mem=64G --pty bash
    export PYTHONPATH="$PWD/src"
    export OMP_NUM_THREADS=32   # the CPU SVDs of ISO follow this

    # 1. Trial run on layer 0 (dense) and layer 30 (MoE): about 80 GB of downloads (13 GB per
    #    checkpoint for layer 30) and about 15 minutes with 32 threads on one 96-core node.
    #    The output is a partial checkpoint, not loadable. Add --dry-run to print the shards
    #    and the download size without downloading.
    python -m merge360.k2_375b_rl_merge --workdir WORK --output WORK/trial --layers 0 30

    # 2. The full model has 4.5 TB of inputs and a 0.76 TB output. Download first; this takes
    #    3 to 4 hours at about 0.4 GB/s, and an interrupted download resumes.
    python -m merge360.k2_375b_rl_merge --workdir WORK --output WORK/rl-merged --steps download

    # 3. Merge. One process takes about 5 hours on 32 threads (extrapolated from one layer).
    #    Peak memory was 17 GB on layer 0, which holds the largest ISO matrices; the last shard
    #    (embedding and lm_head) was not measured, so allow 64 GB. As a Slurm array:
    #      #SBATCH --array=0-15 --cpus-per-task=32 --mem=64G
    python -m merge360.k2_375b_rl_merge --workdir WORK --output WORK/rl-merged --steps merge \
      --num-splits 16 --split-id "$SLURM_ARRAY_TASK_ID"

    # 4. After every task succeeds: combine the ranges and validate the checkpoint.
    python -m merge360 finalize --output WORK/rl-merged
    python -m merge360 validate WORK/rl-merged --method iso
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from .cli import main as merge_main

REPO = "IFM/K2-Horizon-375B-A23B"
BASE = "mid_4_6000"
EXPERTS = ("rl_knowledge_work", "rl_if", "rl_search", "rl_tool-use", "rl_reasoning")
NUM_LAYERS = 61
INDEX = "model.safetensors.index.json"
# Copied from the base into the output, like in the release.
ASSETS = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "chat_template.jinja", "configuration_k2_horizon.py",
          "modeling_k2_horizon.py")
# The release recipe, with uniform weights and BF16 output as the defaults.
ISO_FLAGS = ["--iso-pattern", "self_attn|shared_experts", "--fallback", "ram",
             "--epsilon", "1e-5", "--keep", "0.9", "--iso-ridge", "1e-12",
             "--coefficient-clip", "1.5"]
LAYER_NAME = re.compile(r"model\.layers\.(\d+)\.")


def _parse_layers(specs: Sequence[str]) -> list[int]:
    """Layer numbers from tokens such as ``0``, ``30`` or ``5-8`` (commas also work)."""
    layers: set[int] = set()
    for token in filter(None, ",".join(specs).split(",")):
        low, _, high = token.partition("-")
        first, last = int(low), int(high or low)
        if first > last:
            raise ValueError(f"empty layer range {token!r}")
        layers.update(range(first, last + 1))
    if not layers or min(layers) < 0 or max(layers) >= NUM_LAYERS:
        raise ValueError(f"--layers needs layers in 0..{NUM_LAYERS - 1}")
    return sorted(layers)


def _select(weight_map: Mapping[str, str], layers: Sequence[int] | None):
    """Shards and tensor names to merge. Each layer is one shard; the last one also holds
    ``embed_tokens``, ``lm_head`` and the final norm."""
    if layers is None:
        return sorted(set(weight_map.values())), list(weight_map)
    shards = {shard for name, shard in weight_map.items()
              if (match := LAYER_NAME.match(name)) and int(match[1]) in layers}
    if len(shards) != len(layers):
        raise ValueError(f"layers {list(layers)} do not map to {len(layers)} shards")
    return sorted(shards), [name for name, shard in weight_map.items() if shard in shards]


def _download(tag: str, hf_dir: Path, shards: Sequence[str], assets: bool) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, revision=tag, local_dir=hf_dir / tag,
                      allow_patterns=[INDEX, *shards, *(ASSETS if assets else ())])


def _weight_map(tag: str, hf_dir: Path) -> dict[str, str]:
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(REPO, INDEX, revision=tag, local_dir=hf_dir / tag)) as handle:
        return json.load(handle)["weight_map"]


def _view(tag: str, hf_dir: Path, view_dir: Path, shards: Sequence[str],
          weight_map: Mapping[str, str], names: Sequence[str]) -> Path:
    """Symlinks to the downloaded files plus an index limited to ``names``, which is what
    merge360 reads. Range workers build the same view at once, so this is idempotent and the
    index appears atomically."""
    view, source = view_dir / tag, (hf_dir / tag).resolve()
    view.mkdir(parents=True, exist_ok=True)
    for name in (*shards, *ASSETS):
        if (source / name).exists() and not (view / name).is_symlink():
            try:
                (view / name).symlink_to(source / name)
            except FileExistsError:
                pass
    temporary = view / f"{INDEX}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(
        {"metadata": {}, "weight_map": {name: weight_map[name] for name in names}}))
    os.replace(temporary, view / INDEX)
    return view


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m merge360.k2_375b_rl_merge",
        description="Rebuild the K2-Horizon 375B RL merge from its five RL experts.",
        epilog="Other flags are passed to `merge360 iso`.")
    parser.add_argument("--workdir", required=True,
                        help="downloads go to WORKDIR/hf, symlink views to WORKDIR/view")
    parser.add_argument("--output", required=True, help="merged checkpoint directory")
    parser.add_argument("--layers", nargs="+", metavar="L",
                        help="only these layers, e.g. 0 30 or 5-8 (default: all 61)")
    parser.add_argument("--steps", nargs="+", choices=("download", "merge"),
                        default=["download", "merge"])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the shards and the download size, then stop")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, merge_flags = parser.parse_known_args(argv)
    work, output = Path(args.workdir), Path(args.output)
    hf_dir, view_dir = work / "hf", work / "view"
    tags = (BASE, *EXPERTS)

    # The six inputs share one tensor layout, so the base index decides what to merge.
    weight_map = _weight_map(BASE, hf_dir)
    try:
        shards, names = _select(weight_map, _parse_layers(args.layers) if args.layers else None)
    except ValueError as error:
        parser.error(str(error))
    print(f"[plan] {len(shards)} shards, {len(names)} tensors per checkpoint", flush=True)

    if args.dry_run:
        from huggingface_hub import HfApi
        sizes = {file.rfilename: file.size for file in HfApi().model_info(
            REPO, revision=BASE, files_metadata=True).siblings}
        each = sum(sizes[shard] for shard in shards)
        print(f"[plan] download {each / 1e9:.1f} GB x {len(tags)} checkpoints = "
              f"{each * len(tags) / 1e12:.2f} TB")
        return

    if "download" in args.steps:
        for tag in tags:
            print(f"[download] {tag}", flush=True)
            _download(tag, hf_dir, shards, assets=tag == BASE)

    if "merge" in args.steps:
        views = [_view(tag, hf_dir, view_dir, shards, weight_map, names) for tag in tags]
        merge_main(["iso", "--base", str(views[0]), "--experts", *map(str, views[1:]),
                    *ISO_FLAGS, "--output", str(output), *merge_flags])


if __name__ == "__main__":
    main()
