"""Fast CPU checks for COMPACT state semantics (no pretrained checkpoint)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from jamel_compact.config import CompactConfig
from jamel_compact.model import SideMemoryModule


def test_zero_init_injection():
    module = SideMemoryModule(0, 1, 32, mem_dim=8, num_mem=2, num_heads=2, num_obs_tokens=2,
                              config=CompactConfig(mem_dim=8, num_mem_tokens=2, num_heads=2, num_obs_tokens=2))
    hidden = torch.randn(2, 5, 32)
    memory = torch.randn(2, 2, 8)
    assert torch.equal(module.inject(hidden, memory), hidden)


def test_masked_observation_and_backward():
    module = SideMemoryModule(0, 1, 32, mem_dim=8, num_mem=2, num_heads=2, num_obs_tokens=2,
                              config=CompactConfig(mem_dim=8, num_mem_tokens=2, num_heads=2, num_obs_tokens=2))
    hidden = torch.randn(2, 5, 32, requires_grad=True)
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])
    z = module.extract_observation(hidden, mask)
    state = module.init_memory[None].expand(2, -1, -1).clone()
    variance = torch.ones(2, 2)
    predicted, p_hat, q, surprise = module.predict(state, variance, torch.randn(2, 32))
    updated, _, _, obs, nll = module.correct(predicted, p_hat, z, q_noise=q, surprise_inflation=surprise)
    (updated.float().pow(2).mean() + obs + nll).backward()
    assert hidden.grad is not None


if __name__ == "__main__":
    test_zero_init_injection()
    test_masked_observation_and_backward()
    print("COMPACT invariants OK")
