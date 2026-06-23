"""
smoke_test.py
-------------
Standalone verification script for the Pokémon TCG RL pipeline.

Checks:
  1. Tensor shape contract (bmm attention)
  2. Invalid-action masking correctness
  3. 5-episode mock training run (REINFORCE)
  4. Checkpoint save + load round-trip
  5. TimeGatedAgent fallback trigger

Run with:
  python smoke_test.py
  conda run -n pokemage python smoke_test.py
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

PASS = "✓"
FAIL = "✗"

def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────
section("1. Import checks")

try:
    import torch
    import numpy as np
    print(f"  {PASS} torch  {torch.__version__}")
    print(f"  {PASS} numpy  {np.__version__}")
except ImportError as e:
    print(f"  {FAIL} {e}")
    sys.exit(1)

try:
    from config import Config
    from card_data import card_id_to_idx, init as card_init
    from env_wrapper import MockCabtEnv, extract_action_features, make_env
    from model import PolicyNetwork, PolicyOutput
    from train import (
        RolloutBuffer, Transition, collect_episode,
        reinforce_loss, save_checkpoint, load_checkpoint,
        flush_memory, get_device,
    )
    from eval import TimeGatedAgent
    print(f"  {PASS} All project modules imported successfully")
except ImportError as e:
    print(f"  {FAIL} Project import failed: {e}")
    sys.exit(1)

# Report active accelerator
_device = get_device()
_dev_label = _device.type.upper()
print(f"  {'⚡' if _device.type == 'mps' else PASS} Accelerator : {_dev_label}")
if _device.type == "mps":
    print(f"  {PASS} MPS built   : {torch.backends.mps.is_built()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Card vocabulary
# ─────────────────────────────────────────────────────────────────────────────
section("2. Card vocabulary")

card_init()  # load CSV or synthetic vocab
from card_data import _card_to_idx   # peek at first real key
_first_card = next(iter(_card_to_idx))
idx = card_id_to_idx(_first_card)
assert idx > 0, f"Expected idx > 0, got {idx}"
unk = card_id_to_idx("NOT_A_REAL_CARD_XYZ")
assert unk == 0, f"Expected UNK=0, got {unk}"
print(f"  {PASS} card_id_to_idx('{_first_card}') = {idx}  (vocab size: {len(_card_to_idx)})")
print(f"  {PASS} UNK lookup = {unk}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Tensor shape contract
# ─────────────────────────────────────────────────────────────────────────────
section(f"3. PolicyNetwork tensor shape contract (bmm)  [{_device.type.upper()}]")

cfg = Config(num_games=5, flush_every=10, batch_games=5)
net = PolicyNetwork(cfg).to(_device)
net.eval()

B, N = 4, 12
s = torch.randn(B, cfg.state_dim, device=_device)
a = torch.randn(B, N, cfg.action_dim, device=_device)
m = torch.zeros(B, N, dtype=torch.bool, device=_device)
m[:, -2:] = True   # last 2 actions illegal

out = net(s, a, m)

# Move to CPU for assertions (avoids MPS scalar comparison issues)
logits_cpu = out.logits.cpu()
probs_cpu  = out.probs.cpu()

assert out.logits.shape    == (B, N), f"logits {out.logits.shape}"
assert out.probs.shape     == (B, N), f"probs {out.probs.shape}"
assert out.log_probs.shape == (B, N), f"log_probs {out.log_probs.shape}"
assert out.value.shape     == (B, 1), f"value {out.value.shape}"
assert out.entropy.shape   == (B,),   f"entropy {out.entropy.shape}"

# Masked logits must be effectively -inf
assert (logits_cpu[:, -2:] < -1e8).all(), "Masking not applied to last 2 actions"

# Valid probs must sum to ~1
valid_sum = probs_cpu[:, :-2].sum(dim=-1)
assert torch.allclose(valid_sum, torch.ones(B), atol=1e-4), \
    f"Probs don't sum to 1: {valid_sum}"

print(f"  {PASS} logits     : {tuple(out.logits.shape)}")
print(f"  {PASS} probs      : {tuple(out.probs.shape)}  (sum over legal = {valid_sum[0]:.6f})")
print(f"  {PASS} log_probs  : {tuple(out.log_probs.shape)}")
print(f"  {PASS} value      : {tuple(out.value.shape)}")
print(f"  {PASS} entropy    : {tuple(out.entropy.shape)}")
print(f"  {PASS} Masked logits verified (< -1e8 for illegal actions)")
print(f"  {PASS} bmm: (B={B}, N={N}, D=512) × (B={B}, D=512, 1) → (B={B}, N={N}) ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Mock environment step
# ─────────────────────────────────────────────────────────────────────────────
section("4. MockCabtEnv reset / step")

env = MockCabtEnv(deck_key="dragapult_ex", seed=42)
sf, af, am = env.reset()

assert sf.shape == (cfg.state_dim,),                 f"state shape {sf.shape}"
assert af.shape[1] == cfg.action_dim,                f"action_dim {af.shape}"
assert am.shape == (af.shape[0],),                   f"mask shape {am.shape}"
assert not am.all(), "All actions masked – should have some legal moves"

print(f"  {PASS} reset(): state={sf.shape}, actions={af.shape}, mask={am.shape}")

sf2, af2, am2, r, done = env.step(0)
assert sf2.shape == (cfg.state_dim,)
print(f"  {PASS} step(0): reward={r:.2f}, done={done}")

env.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5. 5-episode REINFORCE training run
# ─────────────────────────────────────────────────────────────────────────────
section("5. 5-episode REINFORCE training run")

import torch.optim as optim

device = _device
policy = PolicyNetwork(cfg).to(device)
optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
buffer = RolloutBuffer()
env    = make_env("mock", deck_key="dragapult_ex")

# Synchronise MPS before timing so we measure real compute, not queued work
if device.type == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
episode_rewards = []

for ep in range(5):
    trajectory = collect_episode(env, policy, cfg, device)
    for t in trajectory:
        buffer.add(t)
    ep_reward = sum(t.reward for t in trajectory)
    episode_rewards.append(ep_reward)

# One gradient update
policy.train()
optimizer.zero_grad()
loss = reinforce_loss(policy, buffer, cfg, device)
loss.backward()
optimizer.step()

# Synchronise MPS before reading loss to CPU
if device.type == "mps":
    torch.mps.synchronize()
elapsed = time.perf_counter() - t0
print(f"  {PASS} 5 episodes collected in {elapsed:.2f}s")
print(f"  {PASS} Loss: {loss.item():.4f}")
print(f"  {PASS} Episode rewards: {[f'{r:+.1f}' for r in episode_rewards]}")

# Loss must be a valid scalar
assert not torch.isnan(loss), "Loss is NaN!"
assert not torch.isinf(loss), "Loss is Inf!"
print(f"  {PASS} Loss is finite and non-NaN")

env.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Checkpoint save + load round-trip
# ─────────────────────────────────────────────────────────────────────────────
section("6. Checkpoint save / load round-trip")

with tempfile.TemporaryDirectory() as tmpdir:
    cfg_ckpt = Config(outdir=tmpdir, num_games=5, flush_every=5, batch_games=5)
    save_checkpoint(policy, optimizer, game_idx=5, cfg=cfg_ckpt)

    latest = Path(tmpdir) / "latest.pth"
    assert latest.exists(), "latest.pth not created"

    policy2    = PolicyNetwork(cfg)
    optimizer2 = optim.Adam(policy2.parameters(), lr=cfg.lr)
    resumed_at = load_checkpoint(str(latest), policy2, optimizer2)
    assert resumed_at == 5, f"Expected game_idx=5, got {resumed_at}"

    # Verify weights are identical (compare on CPU to be device-agnostic)
    for (n1, p1), (n2, p2) in zip(policy.named_parameters(), policy2.named_parameters()):
        assert torch.allclose(p1.cpu(), p2.cpu()), f"Weight mismatch in {n1}"

print(f"  {PASS} Checkpoint saved to latest.pth")
print(f"  {PASS} Loaded checkpoint at game_idx=5")
print(f"  {PASS} Model weights identical after round-trip")

# ─────────────────────────────────────────────────────────────────────────────
# 7. TimeGatedAgent fallback trigger
# ─────────────────────────────────────────────────────────────────────────────
section("7. TimeGatedAgent time-based fallback")

agent = TimeGatedAgent(policy=PolicyNetwork(cfg).to(device), device=device,
                       fallback_threshold_s=30.0, budget_s=570.0)
agent.reset_match_timer()

env = make_env("mock")
sf, af, am = env.reset()

# Normal turn: should use model
idx_model = agent.act(sf, af, am, time_remaining=300.0)
assert 0 <= idx_model < af.shape[0], f"Invalid action idx: {idx_model}"
assert agent._model_count == 1, "Expected model to be used"
print(f"  {PASS} Normal turn → model used, action={idx_model}")

# Low-time turn: should use fallback
idx_fallback = agent.act(sf, af, am, time_remaining=10.0)
assert 0 <= idx_fallback < af.shape[0], f"Invalid fallback idx: {idx_fallback}"
assert agent._fallback_count == 1, "Expected fallback to trigger"
print(f"  {PASS} Low-time turn → fallback used, action={idx_fallback}")

# Summary
print(f"  {PASS} {agent.summary()}")
env.close()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
section("All checks passed ✓")
print("""
  ╔══════════════════════════════════════════════════════╗
  ║   Pokémon TCG RL Pipeline – Smoke Test PASSED  ✓    ║
  ╚══════════════════════════════════════════════════════╝
""")
