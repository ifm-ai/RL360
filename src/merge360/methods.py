"""Per-tensor merge rules.

Every rule maps the matching tensors of one parameter to a merged FP32
tensor. Base-relative rules work on task vectors

    tau_k = theta_k - bf16(theta_0)

where the base is first rounded to the experts' dtype, so FP32-to-BF16
rounding of the base never shows up as an RL update. Non-floating tensors
(integer buffers) must be identical in every input and are copied through.
See README.md for the formulas.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator, Sequence

import torch

# Elements per streamed chunk. Gram-soup accumulates across chunks, so this
# also fixes its floating-point summation order.
CHUNK = 32 * 1024 * 1024
TINY = torch.finfo(torch.float32).tiny


def normalized_weights(values: Sequence[float] | None, count: int) -> list[float]:
    """Expert weights normalized to sum to one; uniform when omitted."""
    if values is None:
        return [1.0 / count] * count
    if len(values) != count:
        raise ValueError(f"Expected {count} weights, got {len(values)}")
    weights = [float(value) for value in values]
    if any(value < 0 for value in weights):
        raise ValueError("Weights must be non-negative")
    total = sum(weights)
    if total == 0:
        raise ValueError("At least one weight must be positive")
    return [value / total for value in weights]


def require_equal_non_float(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
        raise ValueError("Non-floating tensor differs across checkpoints")
    return tensors[0].clone()


def base_for_deltas(base: torch.Tensor, experts: Sequence[torch.Tensor]) -> torch.Tensor:
    """Match the base to expert precision before subtraction, then compute in FP32."""
    if any(expert.dtype != experts[0].dtype for expert in experts[1:]):
        raise ValueError("All experts must use the same dtype")
    return base.to(experts[0].dtype).float()


def _delta_chunks(
    reference_flat: torch.Tensor, expert_flats: Sequence[torch.Tensor]
) -> Iterator[tuple[slice, torch.Tensor, torch.Tensor]]:
    """Yield ``(span, base_chunk, stacked_deltas)`` over bounded chunks."""
    for start in range(0, reference_flat.numel(), CHUNK):
        span = slice(start, min(start + CHUNK, reference_flat.numel()))
        ref = reference_flat[span]
        yield span, ref, torch.stack([expert[span].float() - ref for expert in expert_flats])


def linear_average(tensors: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    """Model soup: ``sum_k w_k theta_k``."""
    if not torch.is_floating_point(tensors[0]):
        return require_equal_non_float(tensors)
    result = torch.zeros_like(tensors[0], dtype=torch.float32)
    for weight, tensor in zip(weights, tensors):
        result.add_(tensor.float(), alpha=weight)
    return result


def task_arithmetic(
    base: torch.Tensor,
    experts: Sequence[torch.Tensor],
    weights: Sequence[float],
    scale: float,
) -> torch.Tensor:
    """``theta_0 + scale * sum_k w_k tau_k``."""
    if not torch.is_floating_point(base):
        return require_equal_non_float([base, *experts])
    reference = base_for_deltas(base, experts)
    result = reference.clone()
    for weight, expert in zip(weights, experts):
        result.add_(expert.float() - reference, alpha=scale * weight)
    return result


def ties_merge(
    base: torch.Tensor,
    experts: Sequence[torch.Tensor],
    weights: Sequence[float],
    density: float,
    scale: float,
) -> torch.Tensor:
    """TIES: trim each task vector to its top ``density`` fraction, elect a
    sign per element, and average only the entries that agree with it."""
    if not torch.is_floating_point(base):
        return require_equal_non_float([base, *experts])
    reference = base_for_deltas(base, experts)
    reference_flat = reference.reshape(-1)
    expert_flats = [expert.reshape(-1) for expert in experts]
    thresholds = [_trim_threshold(expert.float() - reference, density) for expert in experts]

    result = torch.empty_like(reference_flat)
    for span, ref, deltas in _delta_chunks(reference_flat, expert_flats):
        if density >= 1:
            active = torch.ones_like(deltas, dtype=torch.bool)
        else:
            active = torch.stack([deltas[index].abs() >= threshold
                                  for index, threshold in enumerate(thresholds)])
        vote = torch.zeros_like(ref)
        for weight, delta, mask in zip(weights, deltas, active):
            vote.add_(delta * mask, alpha=weight)
        elected_sign = torch.sign(vote)
        agrees = active & (torch.sign(deltas) == elected_sign) & (elected_sign != 0)
        update = torch.zeros_like(ref)
        active_weight = torch.zeros_like(ref)
        for weight, delta, mask in zip(weights, deltas, agrees):
            update.add_(delta * mask, alpha=weight)
            active_weight.add_(mask, alpha=weight)
        update.div_(active_weight.clamp_min_(TINY))
        result[span].copy_(update).mul_(scale).add_(ref)
    return result.reshape(reference.shape)


def _trim_threshold(delta: torch.Tensor, density: float) -> torch.Tensor | None:
    if density >= 1:
        return None
    keep = max(1, math.ceil(delta.numel() * density))
    return torch.kthvalue(delta.abs().reshape(-1), delta.numel() - keep + 1).values


def ram_merge(
    base: torch.Tensor,
    experts: Sequence[torch.Tensor],
    weights: Sequence[float],
    epsilon: float,
    scale: float = 1.0,
    plus_r: float = 0.0,
    plus_alpha: float = 0.2,
) -> torch.Tensor:
    """RAM: average entries that several experts update (``|tau| > epsilon``)
    and keep entries that only one expert updates in full.

    With ``plus_r > 0`` (RAM+), expert ``k``'s unique entries are rescaled by
    ``1 + plus_r * min(shared_k / unique_k, plus_alpha)``, counted per tensor.
    """
    if not torch.is_floating_point(base):
        return require_equal_non_float([base, *experts])
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    reference = base_for_deltas(base, experts)
    reference_flat = reference.reshape(-1)
    expert_flats = [expert.reshape(-1) for expert in experts]
    result = torch.empty_like(reference_flat)

    # RAM+ uses one gain per expert and tensor, so first collect the
    # shared/unique counts in bounded chunks. Plain RAM skips this pass.
    gains = [1.0] * len(experts)
    if plus_r:
        shared_counts = [0] * len(experts)
        unique_counts = [0] * len(experts)
        for _, _, deltas in _delta_chunks(reference_flat, expert_flats):
            active = deltas.abs() > epsilon
            active_count = active.sum(dim=0, dtype=torch.uint8)
            shared = active_count > 1
            unique = active_count == 1
            for index in range(len(experts)):
                shared_counts[index] += (active[index] & shared).sum().item()
                unique_counts[index] += (active[index] & unique).sum().item()
        for index in range(len(experts)):
            ratio = min(shared_counts[index] / max(unique_counts[index], 1), plus_alpha)
            gains[index] = 1.0 + plus_r * ratio

    weight_tensor = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    for span, ref, deltas in _delta_chunks(reference_flat, expert_flats):
        active = deltas.abs() > epsilon
        active_count = active.sum(dim=0, dtype=torch.uint8)
        weighted_active = active.float().mul_(weight_tensor)
        active_weight = weighted_active.sum(dim=0)
        numerator = deltas.mul(weighted_active).sum(dim=0)
        update = torch.where(
            active_count > 1, numerator / active_weight.clamp_min_(TINY), 0.0
        )
        unique = active_count == 1
        for index, gain in enumerate(gains):
            update.add_(torch.where(active[index] & unique, deltas[index], 0.0), alpha=gain)
        result[span].copy_(update).mul_(scale).add_(ref)
    return result.reshape(reference.shape)


def bf16_ulp(weight: torch.Tensor) -> torch.Tensor:
    """BF16 unit in the last place: ``2^(floor(log2|w|) - 7)``.

    BF16 keeps 8 significant bits; ``frexp`` returns ``|w| = m * 2^e`` with
    ``m`` in ``[0.5, 1)``, so the spacing at ``w`` is ``2^(e - 8)``.
    """
    exponent = torch.frexp(weight.abs().clamp_min(1e-30)).exponent
    return torch.ldexp(torch.ones_like(weight), exponent - 8)


def ulpgate_merge(
    base: torch.Tensor,
    experts: Sequence[torch.Tensor],
    weights: Sequence[float],
    k: float,
) -> torch.Tensor:
    """ULP-gate: an expert is active at an element only if it moved by more
    than ``k`` BF16 ULPs of the base weight there; active experts are averaged
    and elements with no active expert keep the base value."""
    if not torch.is_floating_point(base):
        return require_equal_non_float([base, *experts])
    reference = base_for_deltas(base, experts)
    ref_flat = reference.reshape(-1)
    expert_flats = [expert.reshape(-1) for expert in experts]
    weight_col = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    result = torch.empty_like(ref_flat)

    for span, ref, deltas in _delta_chunks(ref_flat, expert_flats):
        active = deltas.abs() > k * bf16_ulp(ref)
        weighted = active.float().mul_(weight_col)
        denom = weighted.sum(dim=0)
        update = torch.where(
            denom > 0,
            deltas.mul_(weighted).sum(dim=0) / denom.clamp_min_(TINY),
            0.0,
        )
        result[span].copy_(ref).add_(update)
    return result.reshape(reference.shape)


def gramsoup_coefficients(
    gram: torch.Tensor, prior_weights: Sequence[float], ridge: float, cmax: float
) -> torch.Tensor:
    """Solve ``(G + ridge * mean(diag G) I) c = diag G``, clamp to
    ``[0, cmax]``, then scale by ``K * w_k`` (a no-op for uniform weights)."""
    count = gram.shape[0]
    diag = gram.diagonal()
    if diag.max() == 0:  # tensor untouched by every expert
        return torch.zeros(count, dtype=torch.float64)
    regularized = gram + ridge * diag.mean() * torch.eye(count, dtype=torch.float64)
    coefficients = torch.linalg.solve(regularized, diag).clamp(0.0, cmax)
    prior = torch.tensor(prior_weights, dtype=torch.float64)
    return coefficients * prior * count


def gramsoup_merge(
    base: torch.Tensor,
    experts: Sequence[torch.Tensor],
    prior_weights: Sequence[float],
    ridge: float,
    cmax: float,
) -> torch.Tensor:
    """Gram-soup: per-tensor coefficients chosen so the merged task vector
    keeps every expert's own component, ``<sum_j c_j tau_j, tau_i> = |tau_i|^2``."""
    if not torch.is_floating_point(base):
        return require_equal_non_float([base, *experts])
    reference = base_for_deltas(base, experts)
    ref_flat = reference.reshape(-1)
    expert_flats = [expert.reshape(-1) for expert in experts]

    # Pass 1: Gram matrix of the flattened task vectors.
    gram = torch.zeros(len(experts), len(experts), dtype=torch.float64)
    for _, _, deltas in _delta_chunks(ref_flat, expert_flats):
        gram += (deltas @ deltas.T).double()
    coefficients = gramsoup_coefficients(gram, prior_weights, ridge, cmax)

    # Pass 2: apply the merged task vector.
    result = torch.empty_like(ref_flat)
    coeff_col = coefficients.float().view(-1, 1)
    for span, ref, deltas in _delta_chunks(ref_flat, expert_flats):
        result[span].copy_(ref).add_((deltas * coeff_col).sum(dim=0))
    return result.reshape(reference.shape)


def _thin_svd(matrix: torch.Tensor):
    if matrix.shape[0] >= matrix.shape[1]:
        return torch.linalg.svd(matrix, full_matrices=False)
    u_t, singular, vh_t = torch.linalg.svd(matrix.T.contiguous(), full_matrices=False)
    return vh_t.T.contiguous(), singular, u_t.T.contiguous()


def _project_tangent(frame: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Project onto the tangent space of the Stiefel manifold at ``frame``."""
    product = frame.T @ displacement
    return displacement - frame @ ((product + product.T) / 2)


def _polar(matrix: torch.Tensor) -> torch.Tensor:
    """Nearest matrix with orthonormal columns (polar retraction)."""
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def iso_matrix(
    base: torch.Tensor,
    experts: Sequence[torch.Tensor],
    keep: float,
    ridge: float,
    clip: float,
    device: str | torch.device,
) -> torch.Tensor:
    """Fixed-spectrum ISO merge of one 2D weight.

    Keeps the base singular values and merges only the singular-vector frames:
    each expert's frames are mapped to tangent vectors at the base frames
    (top ``keep`` fraction of directions), weighted by Gram-solved
    coefficients, summed, and retracted back to orthonormal frames.
    """
    reference = base_for_deltas(base, experts).to(device=device, dtype=torch.float64)
    expert_matrices = [expert.to(device=device, dtype=torch.float64) for expert in experts]
    u0, s0, vh0 = _thin_svd(reference)
    v0 = vh0.T.contiguous()
    kept = max(1, round(keep * s0.numel()))
    tangent_us, tangent_vs = [], []

    for expert in expert_matrices:
        u, _, vh = _thin_svd(expert)
        signs = torch.where((u0 * u).sum(0) >= 0, 1.0, -1.0)
        u = u * signs.unsqueeze(0)
        v = vh.T.contiguous() * signs.unsqueeze(0)
        tangent_u = _project_tangent(u0, u - u0)
        tangent_v = _project_tangent(v0, v - v0)
        tangent_u[:, kept:] = 0
        tangent_v[:, kept:] = 0
        tangent_us.append(tangent_u)
        tangent_vs.append(tangent_v)

    # First-order effect of each expert's frame change on the weight matrix.
    effects = [
        (du * s0.unsqueeze(0)) @ v0.T + (u0 * s0.unsqueeze(0)) @ dv.T
        for du, dv in zip(tangent_us, tangent_vs)
    ]
    gram = torch.stack([
        torch.stack([(left * right).sum() for right in effects]) for left in effects
    ])
    try:
        coefficients = torch.linalg.solve(
            gram + ridge * torch.eye(len(experts), dtype=gram.dtype, device=gram.device),
            torch.diagonal(gram),
        ).clamp_(0, clip)
    except RuntimeError as error:
        print(f"[iso] singular Gram system ({error}); using unit coefficients",
              file=sys.stderr, flush=True)
        coefficients = torch.ones(len(experts), dtype=gram.dtype, device=gram.device)

    du = sum(c * value for c, value in zip(coefficients, tangent_us))
    dv = sum(c * value for c, value in zip(coefficients, tangent_vs))
    u = _polar(u0 + _project_tangent(u0, du))
    v = _polar(v0 + _project_tangent(v0, dv))
    return ((u * s0.unsqueeze(0)) @ v.T).float().cpu()
