"""GBV/BV kernels; Thomas & Pal (2026) and Sun et al. (2024).

The CDF difference is evaluated after scaling and cancellation of its common
prefix mass, using a nonnegative polynomial instead of subtracting powers.
"""
from __future__ import annotations

import torch


def probabilities(logits: torch.Tensor, temperature: float, dtype=torch.float64):
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).to(dtype)
    if temperature < 0:
        raise ValueError("Negative temperature")
    return torch.softmax(logits.to(dtype) / temperature, dim=-1)


def sample(probs: torch.Tensor, generator=None):
    shape = probs.shape[:-1]
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1,
                             generator=generator).reshape(shape)


def power_sum(upper, lower, k: int):
    # (upper**k - lower**k)/(upper-lower), including upper==lower.
    result = torch.ones_like(upper)
    lower_power = torch.ones_like(lower)
    for _ in range(1, k):
        lower_power = lower_power * lower
        result = result * upper + lower_power
    return result


def select_and_reweight(paths, target_by_path, q):
    k, length = paths.shape
    if q.shape[0] != length or target_by_path.shape != (k, length + 1, q.shape[-1]):
        raise ValueError("GBV tensor shape mismatch")
    if not bool(torch.isfinite(q).all()) or not bool((q > 0).all()):
        raise FloatingPointError("GBV requires finite positive draft probabilities")
    scores = []
    for j in range(k):
        chosen_p = target_by_path[j, :length].gather(1, paths[j, :, None])[:, 0]
        chosen_q = q.gather(1, paths[j, :, None])[:, 0]
        scores.append((chosen_p.log() - chosen_q.log()).tolist())
    cpu_paths = paths.tolist()
    selected = max(range(k), key=lambda j: tuple(zip(scores[j], cpu_paths[j])))
    # t=lambda/(lambda+Q), s=Q/(lambda+Q). Keep s separately near t=1.
    t, s = q.new_zeros(()), q.new_ones(())
    rows = []
    for i, token in enumerate(cpu_paths[selected]):
        row = q[i]
        log_ratio = target_by_path[selected, i].log() - row.log()
        order = torch.argsort(log_ratio, stable=True)
        ordered = row[order]
        cumulative = ordered.cumsum(0)
        lower = torch.empty_like(row)
        lower[order] = torch.cat((row.new_zeros(1), cumulative[:-1]))
        a = t + s * lower
        b = s * row
        numerator = row * power_sum(a + b, a, k)
        denominator = power_sum(torch.ones_like(t), t, k)
        conditional = numerator / denominator
        total = conditional.sum()
        if not bool(torch.isfinite(conditional).all()) or not bool(total > 0):
            raise FloatingPointError("Invalid GBV conditional probabilities")
        conditional = conditional / total
        rows.append(conditional)
        scale = a[token] + b[token]
        if not bool(scale > 0):
            raise FloatingPointError("Selected prefix has zero probability")
        t, s = a[token] / scale, b[token] / scale
    return selected, torch.stack(rows)


def block_verify(path, p, r, generator=None):
    length, vocab = path.numel(), p.shape[-1]
    w = p.new_ones(())
    best, bonus = 0, None
    for i in range(length + 1):
        proposal = r[i] if i < length else torch.zeros_like(p[i])
        residual = (w * p[i] - proposal).clamp_min(0)
        weights = torch.cat((residual, (1 - w).clamp_min(0).reshape(1)))
        total = weights.sum()
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Nonfinite BV residual")
        chosen = int(sample(p[i] if total == 0 else weights / total, generator).item())
        if chosen < vocab:
            best, bonus = i, chosen
        if i < length:
            denom = r[i, path[i]]
            if not bool(denom > 0):
                raise FloatingPointError("Selected token has zero proposal mass")
            w = torch.minimum(torch.ones_like(w), w * p[i, path[i]] / denom)
    if bonus is None:
        raise RuntimeError("BV failed to produce a nonempty output")
    return best, bonus


def token_verify(path, p, q, generator=None):
    for i, token in enumerate(path):
        ratio = p[i, token] / q[i, token]
        if torch.rand((), device=q.device, dtype=q.dtype, generator=generator) >= ratio:
            residual = (p[i] - q[i]).clamp_min(0)
            total = residual.sum()
            if not bool(total > 0):
                raise FloatingPointError("Rejection with empty residual")
            return i, int(sample(residual / total, generator).item())
    return len(path), int(sample(p[len(path)], generator).item())


def matching_verify(path, p, generator=None):
    """DFlash rule: match a greedy draft against sampled Target tokens."""
    if p.shape[0] != path.numel() + 1:
        raise ValueError("Matching verifier needs one Target row per draft token plus bonus")
    posterior = sample(p, generator)
    accepted = int((path == posterior[:-1]).long().cumprod(0).sum().item())
    return accepted, int(posterior[accepted].item())
