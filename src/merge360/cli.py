"""Command line: ``python -m merge360 <method|finalize|validate> ...``."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .engine import MergeJob, TensorMerger, finalize, run_merge, run_range
from .methods import (gramsoup_merge, iso_matrix, linear_average, normalized_weights,
                      ram_merge, task_arithmetic, ties_merge, ulpgate_merge)
from .validate import validate_output

Build = Callable[[argparse.Namespace, list[float]], tuple[TensorMerger, dict[str, object]]]


@dataclass(frozen=True)
class Method:
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    build: Build
    uses_base: bool = True


def _no_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def _scale(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiplier on the merged task vector")


def _ram_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epsilon", type=float, default=1e-5,
                        help="an expert is active where |delta| > epsilon")


def _gramsoup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ridge", type=float, default=0.05,
                        help="Gram-soup ridge, as a fraction of mean(diag(G))")
    parser.add_argument("--cmax", type=float, default=1.0,
                        help="upper clamp on Gram-soup coefficients")


def _linear(args, weights):
    return (lambda _name, tensors: linear_average(tensors, weights),
            {"method": "linear", "weights": weights})


def _task_arithmetic(args, weights):
    return (lambda _name, tensors: task_arithmetic(tensors[0], tensors[1:], weights, args.scale),
            {"method": "task-arithmetic", "weights": weights, "scale": args.scale})


def _ties_arguments(parser):
    parser.add_argument("--density", type=float, default=0.2,
                        help="fraction of each task vector kept by magnitude, in (0, 1]")
    _scale(parser)


def _ties(args, weights):
    if not 0 < args.density <= 1:
        raise ValueError("--density must be in (0, 1]")
    return (lambda _name, tensors: ties_merge(
                tensors[0], tensors[1:], weights, args.density, args.scale),
            {"method": "ties", "weights": weights, "density": args.density,
             "scale": args.scale})


def _ram_plus_arguments(parser):
    _ram_arguments(parser)
    _scale(parser)
    parser.add_argument("--plus-r", type=float, default=0.1,
                        help="RAM+ gain on unique updates; 0 gives plain RAM")
    parser.add_argument("--plus-alpha", type=float, default=0.2,
                        help="RAM+ cap on the shared/unique count ratio")


def _ram(args, weights):
    return (lambda _name, tensors: ram_merge(
                tensors[0], tensors[1:], weights, args.epsilon, args.scale,
                args.plus_r, args.plus_alpha),
            {"method": "ram+" if args.plus_r else "ram", "weights": weights,
             "epsilon": args.epsilon, "scale": args.scale,
             "plus_r": args.plus_r, "plus_alpha": args.plus_alpha})


def _ulpgate_arguments(parser):
    parser.add_argument("--k", type=float, default=1.5,
                        help="gate threshold in BF16 ULPs of the base weight")


def _ulpgate(args, weights):
    return (lambda _name, tensors: ulpgate_merge(tensors[0], tensors[1:], weights, args.k),
            {"method": "ulpgate", "weights": weights, "k": args.k})


def _gramsoup(args, weights):
    return (lambda _name, tensors: gramsoup_merge(
                tensors[0], tensors[1:], weights, args.ridge, args.cmax),
            {"method": "gramsoup", "weights": weights, "ridge": args.ridge,
             "cmax": args.cmax})


def _iso_arguments(parser):
    parser.add_argument("--iso-pattern", default=".*",
                        help="regex selecting 2D tensors for ISO, e.g. 'self_attn|shared_experts'")
    parser.add_argument("--keep", type=float, default=0.9,
                        help="fraction of singular directions ISO may rotate, in (0, 1]")
    parser.add_argument("--iso-ridge", type=float, default=1e-12,
                        help="ridge on the ISO coefficient system")
    parser.add_argument("--coefficient-clip", type=float, default=1.5,
                        help="upper clamp on ISO coefficients")
    parser.add_argument("--device", default="cpu", help="device for the SVDs, e.g. cuda")
    parser.add_argument("--fallback", choices=("ram", "task-arithmetic", "gramsoup"),
                        default="ram", help="rule for tensors ISO does not select")
    _ram_arguments(parser)
    _gramsoup_arguments(parser)


def _iso(args, weights):
    if not 0 < args.keep <= 1:
        raise ValueError("--keep must be in (0, 1]")
    pattern = re.compile(args.iso_pattern)
    fallbacks = {
        "ram": (lambda base, experts: ram_merge(base, experts, weights, args.epsilon),
                {"epsilon": args.epsilon}),
        "task-arithmetic": (lambda base, experts: task_arithmetic(base, experts, weights, 1.0),
                            {}),
        "gramsoup": (lambda base, experts: gramsoup_merge(
                         base, experts, weights, args.ridge, args.cmax),
                     {"ridge": args.ridge, "cmax": args.cmax}),
    }
    fallback, fallback_info = fallbacks[args.fallback]

    def merge(name: str, tensors: list[torch.Tensor]) -> torch.Tensor:
        base, experts = tensors[0], tensors[1:]
        if (torch.is_floating_point(base) and base.ndim == 2
                and pattern.search(name) is not None):
            return iso_matrix(base, experts, args.keep, args.iso_ridge,
                              args.coefficient_clip, args.device)
        return fallback(base, experts)

    return merge, {"method": "iso", "iso_pattern": args.iso_pattern, "keep": args.keep,
                   "iso_ridge": args.iso_ridge, "coefficient_clip": args.coefficient_clip,
                   "device": args.device, "fallback": args.fallback, "weights": weights,
                   **fallback_info}


METHODS: dict[str, Method] = {
    "linear": Method("weighted model soup of the experts (no base)",
                     _no_arguments, _linear, uses_base=False),
    "task-arithmetic": Method("base plus a scaled weighted sum of task vectors",
                              _scale, _task_arithmetic),
    "ties": Method("trim, elect sign, and average agreeing task-vector entries",
                   _ties_arguments, _ties),
    "ram": Method("average shared updates, keep task-unique updates (RAM/RAM+)",
                  _ram_plus_arguments, _ram),
    "ulpgate": Method("gate each update at k BF16 ULPs of the base, then average",
                      _ulpgate_arguments, _ulpgate),
    "gramsoup": Method("per-tensor Gram-solved task-vector coefficients",
                       _gramsoup_arguments, _gramsoup),
    "iso": Method("fixed-spectrum ISO on selected matrices, a fallback rule elsewhere",
                  _iso_arguments, _iso),
}


def _add_job_arguments(parser: argparse.ArgumentParser, uses_base: bool) -> None:
    inputs = parser.add_argument_group("inputs")
    if uses_base:
        inputs.add_argument("--base", required=True,
                            help="checkpoint the experts were trained from")
    inputs.add_argument("--experts", nargs="+", required=True)
    inputs.add_argument("--weights", nargs="+", type=float,
                        help="relative expert weights (normalized; default uniform)")

    output = parser.add_argument_group("output")
    output.add_argument("--output", required=True)
    output.add_argument("--output-dtype", default="bfloat16",
                        choices=("bfloat16", "float16", "float32", "same"))
    output.add_argument("--max-shard-size", default="5GiB")
    output.add_argument("--progress-every", type=int, default=100)
    output.add_argument("--fetch-threads", type=int, default=4,
                        help="inputs read concurrently; keep modest on shared filesystems")
    mode = output.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="replace an existing output")
    mode.add_argument("--resume", action="store_true",
                      help="continue a matching <output>.tmp")
    output.add_argument("--resume-legacy", action="store_true",
                        help="with --resume, accept a .tmp without merge_progress.json")
    output.add_argument("--dry-run", action="store_true",
                        help="check inputs and print the plan without writing")

    parallel = parser.add_argument_group("range-parallel")
    parallel.add_argument("--num-splits", type=int,
                          help="run as one of N range workers, then `finalize`")
    parallel.add_argument("--split-id", type=int, help="this worker's range, 0..N-1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m merge360",
        description="Streaming merges of matching Hugging Face safetensors checkpoints.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, method in METHODS.items():
        sub = commands.add_parser(name, help=method.help, description=method.help)
        _add_job_arguments(sub, method.uses_base)
        method.add_arguments(sub.add_argument_group(f"{name} options"))

    sub = commands.add_parser("finalize", help="combine finished range workers")
    sub.add_argument("--output", required=True)

    sub = commands.add_parser("validate", help="check a finished merge output")
    sub.add_argument("output")
    sub.add_argument("--method", help="required merge_info method")
    sub.add_argument("--tensor-count", type=int)
    sub.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"),
                     help="required dtype of floating tensors")
    sub.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE",
                     help="require a merge_info field; VALUE is parsed as JSON when possible")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "finalize":
        finalize(args.output)
        return
    if args.command == "validate":
        validate_output(args.output, args.method, args.tensor_count, args.dtype, args.expect)
        return

    method = METHODS[args.command]
    try:
        weights = normalized_weights(args.weights, len(args.experts))
        merger, method_info = method.build(args, weights)
    except ValueError as error:
        parser.error(str(error))
    inputs = [args.base, *args.experts] if method.uses_base else args.experts
    job = MergeJob(inputs, merger, method_info, args.output,
                   output_dtype=args.output_dtype, max_shard_size=args.max_shard_size,
                   progress_every=args.progress_every, fetch_threads=args.fetch_threads)
    if args.num_splits is None:
        if args.split_id is not None:
            parser.error("--split-id requires --num-splits")
        run_merge(job, force=args.force, resume=args.resume,
                  resume_legacy=args.resume_legacy, dry_run=args.dry_run)
        return
    if args.split_id is None:
        parser.error("--num-splits requires --split-id")
    if args.force or args.resume_legacy:
        parser.error("range workers always resume; --force/--resume-legacy do not apply")
    run_range(job, args.num_splits, args.split_id, dry_run=args.dry_run)
