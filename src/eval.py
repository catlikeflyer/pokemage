"""
eval.py
-------
Time-gated evaluation agent for the Pokémon TCG competition.

The Kaggle submission window is 10 minutes per match.  This module wraps a
trained PolicyNetwork in a guard that:
  - Tracks cumulative elapsed time per episode.
  - If elapsed > TIME_BUDGET_S **or** the caller reports that
    remaining_time < FALLBACK_THRESHOLD_S, bypasses the model entirely and
    instantly returns a random legal action.
  - Otherwise runs the full forward pass and samples from the masked
    categorical distribution.

Submission entry-point
----------------------
The `act(obs, time_remaining)` function is the single callable the Kaggle
runner invokes at each turn.  It should be wired up in agent.py (not provided
here) as:

    from eval import TimeGatedAgent
    agent = TimeGatedAgent.from_checkpoint("./checkpoints/latest.pth")
    def agent_fn(obs):
        return agent.act_from_obs(obs)

Usage
-----
# Quick functional smoke-test against mock env
python eval.py --checkpoint ./checkpoints/latest.pth --num_episodes 3
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import Config
from env_wrapper import CabtEnvBase, MockCabtEnv, extract_action_features, make_env
from model import PolicyNetwork
from train import get_device

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------
MATCH_BUDGET_S:       float = 570.0   # 9.5 min total time budget per match
FALLBACK_THRESHOLD_S: float = 30.0    # drop to fallback if < 30s remain


# ---------------------------------------------------------------------------
# Time-gated agent
# ---------------------------------------------------------------------------
class TimeGatedAgent:
    """Wraps a PolicyNetwork with hard time-based fallback logic.

    Parameters
    ----------
    policy : PolicyNetwork
        Trained model (will be set to eval mode internally).
    device : torch.device
        Where to run inference.
    fallback_threshold_s : float
        If remaining match time falls below this value, use random fallback.
    budget_s : float
        Total agent-side time budget per match (cumulative across all turns).
    """

    def __init__(
        self,
        policy:               PolicyNetwork,
        device:               torch.device | None = None,
        fallback_threshold_s: float = FALLBACK_THRESHOLD_S,
        budget_s:             float = MATCH_BUDGET_S,
    ) -> None:
        self.device               = device or torch.device("cpu")
        self.policy               = policy.to(self.device)
        self.policy.eval()
        self.fallback_threshold_s = fallback_threshold_s
        self.budget_s             = budget_s

        # Per-match state
        self._match_start:  float = time.monotonic()
        self._turn_count:   int   = 0
        self._fallback_count: int = 0
        self._model_count:  int   = 0

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        cfg:             Config | None = None,
        device:          torch.device | None = None,
    ) -> "TimeGatedAgent":
        """Load a trained PolicyNetwork from a .pth file."""
        if cfg is None:
            cfg = Config()
        if device is None:
            device = get_device()

        if not Path(checkpoint_path).exists():
            logger.warning(
                "Checkpoint '%s' not found – using random policy.", checkpoint_path
            )
            policy = PolicyNetwork(cfg)
        else:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            policy = PolicyNetwork(cfg)
            policy.load_state_dict(state["model_state"])
            logger.info("Loaded checkpoint: %s", checkpoint_path)

        return cls(policy=policy, device=device)

    # ------------------------------------------------------------------
    # Core action methods
    # ------------------------------------------------------------------
    def reset_match_timer(self) -> None:
        """Call at the start of each new match/episode."""
        self._match_start    = time.monotonic()
        self._turn_count     = 0
        self._fallback_count = 0
        self._model_count    = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._match_start

    def _should_fallback(self, time_remaining: float | None) -> bool:
        """Return True if we must skip model inference this turn."""
        if time_remaining is not None and time_remaining < self.fallback_threshold_s:
            return True
        if self.elapsed > self.budget_s:
            return True
        return False

    def act(
        self,
        state_feat:  np.ndarray,         # (STATE_DIM,)
        action_feat: np.ndarray,         # (N, ACTION_DIM)
        action_mask: np.ndarray,         # (N,) bool
        time_remaining: float | None = None,
    ) -> int:
        """Choose an action index.

        Parameters
        ----------
        state_feat, action_feat, action_mask :
            Pre-parsed features from env_wrapper.
        time_remaining : float | None
            Seconds remaining in the match as reported by the environment,
            if available.  Pass None to rely solely on elapsed tracking.

        Returns
        -------
        int : index into the legal action list.
        """
        self._turn_count += 1
        n_actions = action_feat.shape[0]

        if self._should_fallback(time_remaining):
            self._fallback_count += 1
            # Legal-action mask: pick randomly from legal (mask=False) actions
            legal_idxs = [i for i in range(n_actions) if not action_mask[i]]
            choice = random.choice(legal_idxs) if legal_idxs else 0
            logger.debug(
                "Turn %d: FALLBACK (elapsed=%.1fs, remaining=%s) → action %d",
                self._turn_count, self.elapsed,
                f"{time_remaining:.1f}s" if time_remaining else "N/A",
                choice,
            )
            return choice

        # ── Model inference ──────────────────────────────────────────
        self._model_count += 1
        s_t = torch.from_numpy(state_feat).unsqueeze(0).to(self.device)   # (1, D)
        a_t = torch.from_numpy(action_feat).unsqueeze(0).to(self.device)  # (1, N, A)
        m_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)  # (1, N)

        with torch.no_grad():
            action_idx, log_prob, value = self.policy.act(s_t, a_t, m_t)

        logger.debug(
            "Turn %d: MODEL → action %d (lp=%.3f, val=%.3f, elapsed=%.1fs)",
            self._turn_count, action_idx, log_prob, value, self.elapsed,
        )
        return action_idx

    def act_from_obs(
        self,
        obs: Any,
        time_remaining: float | None = None,
    ) -> Any:
        """High-level entry-point for the live cabt environment.

        Parses `obs` using LiveCabtEnv helpers and returns the raw action
        object expected by the cabt engine.
        """
        try:
            from env_wrapper import LiveCabtEnv
            # Temporarily build a thin adapter; in production this is
            # done once at startup and cached.
            _adapter = LiveCabtEnv.__new__(LiveCabtEnv)
            state_feat   = _adapter._extract_state(obs)
            raw_actions  = _adapter._get_legal_actions(obs)
            action_feats, action_mask = extract_action_features(raw_actions)

            idx = self.act(state_feat, action_feats, action_mask, time_remaining)
            return _adapter._action_to_cabt(raw_actions[idx])
        except Exception as exc:
            logger.warning("act_from_obs failed (%s) – random fallback", exc)
            return None  # cabt treats None as pass in many versions

    def summary(self) -> str:
        total = self._turn_count or 1
        return (
            f"Turns: {self._turn_count} | "
            f"Model: {self._model_count} ({100*self._model_count//total}%) | "
            f"Fallback: {self._fallback_count} ({100*self._fallback_count//total}%) | "
            f"Elapsed: {self.elapsed:.1f}s"
        )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

def _run_eval(args: argparse.Namespace) -> None:
    cfg    = Config()
    agent  = TimeGatedAgent.from_checkpoint(args.checkpoint, cfg)
    env    = make_env("mock", deck_key="dragapult_ex")

    for ep in range(args.num_episodes):
        agent.reset_match_timer()
        state_feat, action_feat, action_mask = env.reset()
        done = False
        total_reward = 0.0
        turns = 0

        while not done:
            # Simulate time_remaining decrement (mock: always plenty of time)
            fake_remaining = max(0.0, 600.0 - agent.elapsed)
            idx = agent.act(state_feat, action_feat, action_mask, fake_remaining)
            state_feat, action_feat, action_mask, reward, done = env.step(idx)
            total_reward += reward
            turns += 1

        result = "WIN" if total_reward > 0 else ("LOSS" if total_reward < 0 else "DRAW")
        logger.info(
            "Episode %d | %s | reward: %+.1f | turns: %d | %s",
            ep + 1, result, total_reward, turns, agent.summary(),
        )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate time-gated Pokémon TCG agent")
    parser.add_argument("--checkpoint",    type=str, default="./checkpoints/latest.pth")
    parser.add_argument("--num_episodes",  type=int, default=3)
    args = parser.parse_args()
    _run_eval(args)
