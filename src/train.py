"""
train.py
--------
Custom REINFORCE / PPO training loop for the Pokémon TCG RL pipeline.

Usage
-----
# Fresh run (mock env, REINFORCE, 500 games)
python train.py --num_games 500 --env mock --algo reinforce

# Resume from checkpoint
python train.py --num_games 500 --checkpoint ./checkpoints/latest.pth

# PPO on live cabt env
python train.py --num_games 1000 --env live --algo ppo --deck dragapult_ex

Memory safeguards
-----------------
Every --flush_every games:
  1. Save weights to checkpoints/checkpoint_{N:06d}.pth + checkpoints/latest.pth
  2. gc.collect()
  3. Clear the accelerator cache (CUDA / MPS / CPU as appropriate)
  4. Clear rollout buffer

The companion run_batched.sh terminates the entire Python process after each
batch, completely flushing the C++ heap to mitigate the libcg.so memory leak.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from env_wrapper import CabtEnvBase, make_env
from model import PolicyNetwork

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    state_feat:  np.ndarray
    action_feat: np.ndarray
    action_mask: np.ndarray
    action_idx:  int
    log_prob:    float
    reward:      float
    done:        bool
    value:       float


class RolloutBuffer:
    """Stores transitions for one batch of games before a gradient update."""

    def __init__(self) -> None:
        self._transitions: List[Transition] = []
        self._episode_boundaries: List[int] = []   # indices of final steps

    # ------------------------------------------------------------------
    def add(self, t: Transition) -> None:
        self._transitions.append(t)
        if t.done:
            self._episode_boundaries.append(len(self._transitions) - 1)

    # ------------------------------------------------------------------
    def compute_returns(self, gamma: float) -> np.ndarray:
        """Compute discounted returns for every stored transition.

        Handles multiple episodes end-to-end; reward is reset at done=True.
        """
        n = len(self._transitions)
        returns = np.zeros(n, dtype=np.float32)
        running = 0.0
        for i in reversed(range(n)):
            t = self._transitions[i]
            if t.done:
                running = 0.0
            running = t.reward + gamma * running
            returns[i] = running
        return returns

    # ------------------------------------------------------------------
    def compute_gae(self, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation for PPO.

        Returns
        -------
        advantages : (N,) float32
        returns    : (N,) float32   (advantages + values, used as value targets)
        """
        n = len(self._transitions)
        advantages = np.zeros(n, dtype=np.float32)
        returns    = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for i in reversed(range(n)):
            t = self._transitions[i]
            next_val = 0.0 if t.done else self._transitions[i + 1].value
            delta = t.reward + gamma * next_val - t.value
            gae = delta + gamma * lam * (0.0 if t.done else gae)
            advantages[i] = gae
            returns[i]    = gae + t.value
        return advantages, returns

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._transitions.clear()
        self._episode_boundaries.clear()

    def __len__(self) -> int:
        return len(self._transitions)

    def is_empty(self) -> bool:
        return len(self._transitions) == 0

    def num_episodes(self) -> int:
        return len(self._episode_boundaries)


# ---------------------------------------------------------------------------
# Batch construction helpers
# ---------------------------------------------------------------------------
def _pad_actions(
    feats: List[np.ndarray],
    masks: List[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length action sets to the same N within a mini-batch.

    Parameters
    ----------
    feats : list of (Ni, ACTION_DIM) arrays
    masks : list of (Ni,) bool arrays

    Returns
    -------
    a_feat_t : (B, N_max, ACTION_DIM)
    a_mask_t : (B, N_max) bool    True = padded (illegal)
    """
    max_n = max(f.shape[0] for f in feats)
    action_dim = feats[0].shape[1]
    B = len(feats)

    a_feat_padded = np.zeros((B, max_n, action_dim), dtype=np.float32)
    a_mask_padded = np.ones((B, max_n), dtype=bool)   # padding = True (illegal)

    for i, (f, m) in enumerate(zip(feats, masks)):
        ni = f.shape[0]
        a_feat_padded[i, :ni] = f
        a_mask_padded[i, :ni] = m

    return (
        torch.from_numpy(a_feat_padded),
        torch.from_numpy(a_mask_padded),
    )


def _build_batch(
    transitions: List[Transition],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a list of Transition objects to batched tensors.

    Returns
    -------
    s_t      : (B, STATE_DIM)
    a_feat_t : (B, N_max, ACTION_DIM)
    a_mask_t : (B, N_max) bool
    ai_t     : (B,) long    action indices
    lp_t     : (B,)         old log-probs
    """
    states = np.stack([t.state_feat for t in transitions], axis=0)
    feats  = [t.action_feat for t in transitions]
    masks  = [t.action_mask for t in transitions]
    aidxs  = np.array([t.action_idx for t in transitions], dtype=np.int64)
    lps    = np.array([t.log_prob   for t in transitions], dtype=np.float32)

    a_feat_t, a_mask_t = _pad_actions(feats, masks)

    s_t  = torch.from_numpy(states).to(device)
    ai_t = torch.from_numpy(aidxs).to(device)
    lp_t = torch.from_numpy(lps).to(device)
    a_feat_t = a_feat_t.to(device)
    a_mask_t = a_mask_t.to(device)

    return s_t, a_feat_t, a_mask_t, ai_t, lp_t


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def reinforce_loss(
    policy:     PolicyNetwork,
    buffer:     RolloutBuffer,
    cfg:        Config,
    device:     torch.device,
) -> torch.Tensor:
    """REINFORCE with value baseline.

    Loss = -∑ log π(a|s) · (G_t - V(s))  +  value_loss  -  entropy_bonus
    """
    transitions = buffer._transitions
    if not transitions:
        return torch.tensor(0.0, requires_grad=True)

    returns_np = buffer.compute_returns(cfg.gamma)
    returns_t  = torch.from_numpy(returns_np).to(device)

    s_t, a_feat_t, a_mask_t, ai_t, _ = _build_batch(transitions, device)

    out = policy(s_t, a_feat_t, a_mask_t)

    # Gather log-probs for chosen actions
    chosen_lp = out.log_probs.gather(1, ai_t.unsqueeze(1)).squeeze(1)  # (B,)

    value = out.value.squeeze(1)   # (B,)
    advantage = (returns_t - value.detach()).float()

    # Normalise advantages for stability
    if advantage.std() > 1e-8:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    policy_loss = -(chosen_lp * advantage).mean()
    value_loss  = nn.functional.mse_loss(value, returns_t)
    entropy     = out.entropy.mean()

    total = policy_loss + cfg.value_coeff * value_loss - cfg.entropy_coeff * entropy
    return total


def ppo_loss(
    policy:        PolicyNetwork,
    buffer:        RolloutBuffer,
    cfg:           Config,
    device:        torch.device,
) -> torch.Tensor:
    """PPO-clip loss with mini-batch epochs.

    Iterates over the buffer for ppo_epochs passes, sampling mini-batches of
    ppo_minibatch transitions.  Returns the mean loss over all mini-batches.
    """
    transitions = buffer._transitions
    if not transitions:
        return torch.tensor(0.0, requires_grad=True)

    advantages_np, returns_np = buffer.compute_gae(cfg.gamma, cfg.lam)
    # Normalise advantages across the full buffer before mini-batching
    adv_mean = advantages_np.mean()
    adv_std  = advantages_np.std() + 1e-8
    advantages_np = (advantages_np - adv_mean) / adv_std

    adv_t     = torch.from_numpy(advantages_np).float().to(device)
    returns_t = torch.from_numpy(returns_np).float().to(device)

    n = len(transitions)
    indices = np.arange(n)
    total_loss = torch.tensor(0.0, device=device)
    batch_count = 0

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, cfg.ppo_minibatch):
            end = min(start + cfg.ppo_minibatch, n)
            batch_idx = indices[start:end]
            batch_t   = [transitions[i] for i in batch_idx]

            s_t, a_feat_t, a_mask_t, ai_t, old_lp_t = _build_batch(batch_t, device)
            b_adv = adv_t[batch_idx]
            b_ret = returns_t[batch_idx]

            out = policy(s_t, a_feat_t, a_mask_t)
            new_lp = out.log_probs.gather(1, ai_t.unsqueeze(1)).squeeze(1)
            old_lp = old_lp_t

            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value = out.value.squeeze(1)
            value_loss = nn.functional.mse_loss(value, b_ret)
            entropy    = out.entropy.mean()

            loss = policy_loss + cfg.value_coeff * value_loss - cfg.entropy_coeff * entropy
            total_loss = total_loss + loss
            batch_count += 1

    return total_loss / max(batch_count, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    policy:    PolicyNetwork,
    optimizer: optim.Optimizer,
    game_idx:  int,
    cfg:       Config,
) -> None:
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    state = {
        "game_idx":        game_idx,
        "model_state":     policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config":          cfg.__dict__,
    }
    numbered = outdir / f"checkpoint_{game_idx:06d}.pth"
    latest   = outdir / "latest.pth"

    torch.save(state, numbered)
    torch.save(state, latest)
    logger.info("✓ Checkpoint saved: %s  (also → latest.pth)", numbered.name)


def load_checkpoint(
    path:      str,
    policy:    PolicyNetwork,
    optimizer: optim.Optimizer,
) -> int:
    """Load weights and optimizer state. Returns game_idx to resume from."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    game_idx = state.get("game_idx", 0)
    logger.info("Resumed from %s at game %d", path, game_idx)
    return game_idx


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available accelerator: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Memory flush routine
# ---------------------------------------------------------------------------

def flush_memory(
    policy:    PolicyNetwork,
    optimizer: optim.Optimizer,
    game_idx:  int,
    cfg:       Config,
    buffer:    RolloutBuffer,
) -> None:
    """Save checkpoint, run GC, clear accelerator cache, and empty the buffer."""
    save_checkpoint(policy, optimizer, game_idx, cfg)
    buffer.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    logger.info("  ↳ Memory flushed at game %d (gc + accel cache cleared)", game_idx)


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def collect_episode(
    env:    CabtEnvBase,
    policy: PolicyNetwork,
    cfg:    Config,
    device: torch.device,
) -> List[Transition]:
    """Run one complete game and return the trajectory."""
    state_feat, action_feat, action_mask = env.reset()
    transitions: List[Transition] = []

    # Guard: if reset() finished with done=True (e.g. cabt setup failure or
    # immediate forfeit), return an empty trajectory rather than crashing.
    if getattr(env, "_done", False):
        logger.warning("collect_episode: env already done after reset() — skipping episode")
        return transitions

    done = False
    while not done:
        s_t = torch.from_numpy(state_feat).unsqueeze(0).to(device)   # (1, STATE_DIM)
        a_t = torch.from_numpy(action_feat).unsqueeze(0).to(device)  # (1, N, ACTION_DIM)
        m_t = torch.from_numpy(action_mask).unsqueeze(0).to(device)  # (1, N)

        with torch.no_grad():
            action_idx, log_prob, value = policy.act(s_t, a_t, m_t)

        next_state, next_a_feat, next_a_mask, reward, done = env.step(action_idx)

        transitions.append(Transition(
            state_feat=state_feat,
            action_feat=action_feat,
            action_mask=action_mask,
            action_idx=action_idx,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
        ))

        state_feat  = next_state
        action_feat = next_a_feat
        action_mask = next_a_mask

    return transitions


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    device = get_device()
    logger.info("Device: %s | algo: %s | env: %s | games: %d",
                device, cfg.algo, cfg.env, cfg.num_games)

    # ── Model & optimiser ────────────────────────────────────────────
    policy    = PolicyNetwork(cfg).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    start_game = 0
    if cfg.checkpoint and Path(cfg.checkpoint).exists():
        start_game = load_checkpoint(cfg.checkpoint, policy, optimizer)

    buffer = RolloutBuffer()
    env    = make_env(cfg.env, deck_key=cfg.deck)

    episode_rewards: List[float] = []
    losses: List[float] = []
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("Starting training from game %d", start_game)
    logger.info("=" * 60)

    for game_i in range(start_game, start_game + cfg.num_games):
        # ── Collect one episode ───────────────────────────────────────
        policy.eval()
        trajectory = collect_episode(env, policy, cfg, device)
        for t in trajectory:
            buffer.add(t)

        ep_reward = sum(t.reward for t in trajectory)
        episode_rewards.append(ep_reward)

        # ── Gradient update every batch_games episodes ────────────────
        if buffer.num_episodes() >= cfg.batch_games:
            policy.train()
            optimizer.zero_grad()

            if cfg.algo == "ppo":
                loss = ppo_loss(policy, buffer, cfg, device)
            else:
                loss = reinforce_loss(policy, buffer, cfg, device)

            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)
            buffer.clear()   # data consumed

            avg_reward = np.mean(episode_rewards[-cfg.batch_games:])
            elapsed    = time.time() - t_start
            logger.info(
                "Game %5d | reward: %+.3f | loss: %.4f | elapsed: %.1fs",
                game_i + 1, avg_reward, loss_val, elapsed,
            )

        # ── Periodic memory flush & checkpoint ───────────────────────
        if (game_i + 1) % cfg.flush_every == 0:
            flush_memory(policy, optimizer, game_i + 1, cfg, buffer)

    # ── Final checkpoint ─────────────────────────────────────────────
    flush_memory(policy, optimizer, start_game + cfg.num_games, cfg, buffer)
    env.close()

    total_time = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Training complete. %d games in %.1fs (%.2f games/s)",
                cfg.num_games, total_time, cfg.num_games / max(total_time, 1e-6))
    if losses:
        logger.info("Final loss: %.4f | Mean episode reward: %.3f",
                    losses[-1], np.mean(episode_rewards))
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Pokémon TCG RL training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = Config()

    parser.add_argument("--num_games",   type=int,   default=defaults.num_games)
    parser.add_argument("--algo",        type=str,   default=defaults.algo,
                        choices=["reinforce", "ppo"])
    parser.add_argument("--checkpoint",  type=str,   default=defaults.checkpoint,
                        help="Path to .pth checkpoint to resume from")
    parser.add_argument("--env",         type=str,   default=defaults.env,
                        choices=["mock", "live"])
    parser.add_argument("--flush_every", type=int,   default=defaults.flush_every)
    parser.add_argument("--deck",        type=str,   default=defaults.deck,
                        choices=list(["dragapult_ex", "mega_lucario_ex",
                                      "mega_abomasnow_ex", "iono"]))
    parser.add_argument("--outdir",      type=str,   default=defaults.outdir)
    parser.add_argument("--lr",          type=float, default=defaults.lr)
    parser.add_argument("--gamma",       type=float, default=defaults.gamma)
    parser.add_argument("--entropy_coeff", type=float, default=defaults.entropy_coeff)
    parser.add_argument("--batch_games", type=int,   default=defaults.batch_games)
    parser.add_argument("--card_csv",    type=str,   default=defaults.card_csv)

    args = parser.parse_args()
    cfg = Config(
        num_games    = args.num_games,
        algo         = args.algo,
        checkpoint   = args.checkpoint,
        env          = args.env,
        flush_every  = args.flush_every,
        deck         = args.deck,
        outdir       = args.outdir,
        lr           = args.lr,
        gamma        = args.gamma,
        entropy_coeff= args.entropy_coeff,
        batch_games  = args.batch_games,
        card_csv     = args.card_csv,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
