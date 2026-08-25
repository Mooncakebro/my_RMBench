"""
Loss wiring for CompactMoDE.

  - EDM diffusion loss through MoDE's GCDenoiser (Karras preconditioning,
    log-logistic sigma sampling).
  - COMPACT auxiliary losses (L_obs, L_nll, memory L2) re-implemented from
    jamel_compact/loss.py:compute_compact_loss, WITHOUT the text CE term
    (we never generate text).

Also contains local copies of the trivial EDM sampling helpers
(exponential sigma schedule + DDIM sampler) so we don't have to import
mode.models.edm_diffusion.gc_sampling, which pulls in torchsde / torchdiffeq /
matplotlib at module import time.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

import compact_mode  # noqa: F401  (sys.path setup for MODE_ROOT)
import mode.models.edm_diffusion.utils as diffusion_utils


# ── EDM diffusion loss ────────────────────────────────────────────────────────

def sample_sigma(shape, device, sigma_data: float, sigma_min: float, sigma_max: float):
    """Log-logistic sigma sampling, same as MoDE's my_code/model.py."""
    return diffusion_utils.rand_log_logistic(
        shape=shape,
        loc=math.log(sigma_data),
        scale=0.5,
        min_value=sigma_min,
        max_value=sigma_max,
        device=device,
    )


def edm_loss(denoiser, state: dict, actions: torch.Tensor, goal: torch.Tensor,
             sigma_data: float, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """denoiser: GCDenoiser; actions: (B, action_seq_len, action_dim) fp32."""
    actions = actions.float()
    sigma = sample_sigma((actions.shape[0],), actions.device, sigma_data, sigma_min, sigma_max)
    noise = torch.randn_like(actions)
    loss, _ = denoiser.loss(state, actions, goal, noise, sigma)
    return loss


# ── EDM sampling helpers (local, dependency-light copies) ─────────────────────

def get_sigmas_exponential(n: int, sigma_min: float, sigma_max: float, device):
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return torch.cat([sigmas, sigmas.new_zeros([1])])


@torch.no_grad()
def sample_ddim(model, state: dict, action: torch.Tensor, goal: torch.Tensor, sigmas) -> torch.Tensor:
    """DDIM / DPM-Solver-1 sampler, copied from MoDE's gc_sampling.sample_ddim."""
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    for i in range(len(sigmas) - 1):
        denoised = model(state, action, goal, sigmas[i] * s_in)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
    return action


# ── COMPACT auxiliary losses (no text CE) ─────────────────────────────────────

def compact_aux_loss(
    loss_obs: torch.Tensor,
    loss_nll: torch.Tensor,
    memory_states: List[torch.Tensor],
    lambda_obs: float,
    lambda_nll: float,
    lambda_mem: float,
) -> "tuple[torch.Tensor, Dict[str, torch.Tensor]]":
    """
    L_aux = lambda_obs * L_obs + lambda_nll * L_nll + lambda_mem * L_mem

    L_obs / L_nll are the per-layer-averaged terms returned by the patched
    JAMELCompactWrapper.forward (return_hidden_states=True). L_mem is the
    per-layer mean of ||M_l||^2, exactly as in jamel_compact/loss.py.
    """
    device = loss_obs.device
    loss_mem = torch.zeros((), device=device, dtype=torch.float32)
    for M in memory_states:
        loss_mem = loss_mem + M.float().pow(2).sum()
    loss_mem = loss_mem / max(len(memory_states), 1)

    total = (
        lambda_obs * loss_obs.float()
        + lambda_nll * loss_nll.float()
        + lambda_mem * loss_mem
    )
    return total, {
        "obs": loss_obs.detach().float(),
        "nll": loss_nll.detach().float(),
        "mem_l2": loss_mem.detach(),
    }
