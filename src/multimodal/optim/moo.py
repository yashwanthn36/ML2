"""Closed-form two-loss minimum-norm convex gradient combination (PDF Eq. 9.3-9.5)."""
from __future__ import annotations

import torch
from torch import nn


def _grads(loss: torch.Tensor, parameters: list[nn.Parameter], *, retain_graph: bool):
    return torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)


def _dot(a, b) -> torch.Tensor:
    terms = [torch.sum(x * y) for x, y in zip(a, b) if x is not None and y is not None]
    if not terms:
        return torch.tensor(0.0)
    return torch.stack(terms).sum()


def min_norm_combine(grad_ci, grad_mi, *, epsilon: float = 1e-12):
    """Return alpha and the minimum-norm convex combination alpha*g_ci+(1-alpha)*g_mi."""
    dot_ci_ci = _dot(grad_ci, grad_ci)
    dot_mi_mi = _dot(grad_mi, grad_mi)
    dot_ci_mi = _dot(grad_ci, grad_mi)
    denom = dot_ci_ci + dot_mi_mi - 2.0 * dot_ci_mi
    if float(denom.detach()) <= epsilon:
        alpha = torch.tensor(0.5, device=denom.device, dtype=denom.dtype)
    else:
        alpha = ((dot_mi_mi - dot_ci_mi) / denom).clamp(0.0, 1.0)
    combined = [None if a is None and b is None else
                (0.0 if a is None else alpha * a) + (0.0 if b is None else (1.0 - alpha) * b)
                for a, b in zip(grad_ci, grad_mi)]
    return alpha, combined


def normalize_gradients(gradients, *, epsilon: float = 1e-12):
    """Scale an objective's full gradient vector to unit norm for sensitivity studies."""
    norm = _dot(gradients, gradients).sqrt()
    scale = max(float(norm.detach()), epsilon)
    return tuple(None if gradient is None else gradient / scale for gradient in gradients)


def gradient_diagnostics(grad_ci, grad_mi, *, epsilon: float = 1e-12) -> dict[str, float]:
    dot = _dot(grad_ci, grad_mi)
    norm_ci = _dot(grad_ci, grad_ci).sqrt()
    norm_mi = _dot(grad_mi, grad_mi).sqrt()
    cosine = dot / (norm_ci * norm_mi + epsilon)
    return {"cosine": float(cosine.detach()), "norm_ci": float(norm_ci.detach()),
            "norm_mi": float(norm_mi.detach())}


def moo_optimizer_step(loss_ci: torch.Tensor, loss_mi: torch.Tensor,
                       model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Collect both gradients, combine once, assign once, and step the optimizer once."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    grad_ci = _grads(loss_ci, parameters, retain_graph=True)
    grad_mi = _grads(loss_mi, parameters, retain_graph=False)
    diagnostics = gradient_diagnostics(grad_ci, grad_mi)
    alpha, combined = min_norm_combine(grad_ci, grad_mi)
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, combined):
        parameter.grad = gradient
    optimizer.step()
    diagnostics["alpha"] = float(alpha.detach())
    return diagnostics
