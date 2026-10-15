"""Validate a finished sharded safetensors merge."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open

from .checkpoint import INDEX_FILE, OUTPUT_DTYPES


def validate_output(
    output: str | Path,
    method: str | None = None,
    tensor_count: int | None = None,
    dtype: str = "bfloat16",
    expect: Sequence[str] = (),
) -> None:
    """Read every tensor and check the index, shard set, sizes and dtypes.

    ``expect`` holds ``KEY=VALUE`` requirements on ``merge_info.json``;
    VALUE is parsed as JSON when possible (``weights=[0.5,0.5]``).
    """
    if dtype not in OUTPUT_DTYPES:
        raise ValueError(f"--dtype must be one of {sorted(OUTPUT_DTYPES)}, got {dtype!r}")
    output = Path(output)
    index = json.loads((output / INDEX_FILE).read_text())
    info = json.loads((output / "merge_info.json").read_text())
    weight_map = index["weight_map"]
    if method is not None and info["method"] != method:
        raise ValueError(f"Expected method {method}, got {info['method']}")
    if tensor_count is not None and len(weight_map) != tensor_count:
        raise ValueError(f"Expected {tensor_count} tensors, got {len(weight_map)}")
    for expectation in expect:
        key, separator, raw_value = expectation.partition("=")
        if not separator:
            raise ValueError(f"Expected KEY=VALUE, got {expectation!r}")
        try:
            expected = json.loads(raw_value)
        except json.JSONDecodeError:
            expected = raw_value
        if info.get(key) != expected:
            raise ValueError(f"merge_info mismatch for {key}: "
                             f"expected={expected!r}, actual={info.get(key)!r}")

    shards = sorted(set(weight_map.values()))
    actual_shards = sorted(path.name for path in output.glob("*.safetensors"))
    if actual_shards != shards:
        raise ValueError(f"Shard files differ from index: "
                         f"missing={sorted(set(shards) - set(actual_shards))}, "
                         f"extra={sorted(set(actual_shards) - set(shards))}")

    expected_dtype = OUTPUT_DTYPES[dtype]
    seen: set[str] = set()
    total_size = 0
    float_dtypes: set[torch.dtype] = set()
    for shard in shards:
        with safe_open(str(output / shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in seen:
                    raise ValueError(f"Duplicate tensor: {name}")
                if weight_map.get(name) != shard:
                    raise ValueError(f"Index mismatch for tensor: {name}")
                seen.add(name)
                tensor = handle.get_tensor(name)
                total_size += tensor.numel() * tensor.element_size()
                if torch.is_floating_point(tensor):
                    float_dtypes.add(tensor.dtype)
                del tensor
    if seen != set(weight_map):
        raise ValueError(f"Missing tensors: {len(set(weight_map) - seen)}")
    declared_size = index.get("metadata", {}).get("total_size")
    if declared_size != total_size:
        raise ValueError(f"total_size mismatch: index={declared_size}, actual={total_size}")
    if float_dtypes - {expected_dtype}:
        raise ValueError(f"Expected floating tensors in {expected_dtype}, "
                         f"got {sorted(map(str, float_dtypes))}")

    print(f"validated {len(weight_map)} tensors in {len(shards)} shards "
          f"({total_size} bytes): {output}")
