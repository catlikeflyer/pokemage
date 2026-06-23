"""
model.py
--------
Policy network for the Pokémon TCG RL agent.

Architecture
============

StateEncoder
  CardEmbedding(VOCAB_SIZE, EMBED_DIM)      -- shared across all zones
  Zone mean-poolers                          -- one per AreaType
  Linear(concat_dim → STATE_DIM) + LayerNorm

ActionEncoder
  MLP: ACTION_DIM → HIDDEN_DIM → ACTION_DIM -- one vector per legal action

PolicyNetwork
  query_proj: Linear(STATE_DIM → STATE_DIM)
  key_proj:   Linear(ACTION_DIM → STATE_DIM)
  Dynamic attention scoring via torch.bmm
  Invalid-action masking (set logit = -1e9)
  Categorical distribution for sampling / log-prob

ValueHead (baseline / PPO critic)
  Linear(STATE_DIM → 1)

Tensor shape contract (verified by smoke test)
----------------------------------------------
  state_feat  : (B, STATE_DIM)                   e.g. (4, 512)
  action_feat : (B, N, ACTION_DIM)               e.g. (4, 12, 128)
  action_mask : (B, N) bool, True=illegal
  ─── forward outputs ───
  logits      : (B, N)                            raw (masked) scores
  probs       : (B, N)                            softmax over legal actions
  log_probs   : (B, N)                            log of probs
  value       : (B, 1)                            scalar state value
  entropy     : (B,)                              per-sample entropy
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ACTION_DIM,
    CARD_EMBED_DIM,
    CARD_SCALAR_FEATS,
    CARD_VOCAB_SIZE,
    HIDDEN_DIM,
    MAX_BENCH,
    NUM_ZONES,
    STATE_DIM,
    Config,
)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
class PolicyOutput(NamedTuple):
    logits:    torch.Tensor   # (B, N)
    probs:     torch.Tensor   # (B, N)
    log_probs: torch.Tensor   # (B, N)
    value:     torch.Tensor   # (B, 1)
    entropy:   torch.Tensor   # (B,)


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------
class StateEncoder(nn.Module):
    """Maps (STATE_DIM,) raw feature vectors → (STATE_DIM,) latent.

    The raw feature vector produced by env_wrapper already contains mean-pooled
    zone information in float32, so we process it with a residual MLP rather
    than re-embedding.  A separate CardEmbedding is provided for any model
    variants that want end-to-end card-index → embedding.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.state_dim  # raw features already at STATE_DIM

        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.state_dim),
        )
        self.norm = nn.LayerNorm(cfg.state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, STATE_DIM) → (B, STATE_DIM)"""
        return self.norm(self.net(x) + x)   # residual


# ---------------------------------------------------------------------------
# Action encoder
# ---------------------------------------------------------------------------
class ActionEncoder(nn.Module):
    """Maps (N, ACTION_DIM) action vectors → (N, STATE_DIM) keys."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.state_dim),
        )
        self.norm = nn.LayerNorm(cfg.state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N, ACTION_DIM) → (B, N, STATE_DIM)"""
        return self.norm(self.proj(x))


# ---------------------------------------------------------------------------
# Value head
# ---------------------------------------------------------------------------
class ValueHead(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.linear = nn.Linear(cfg.state_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state : (B, STATE_DIM) → (B, 1)"""
        return self.linear(state)


# ---------------------------------------------------------------------------
# Full policy network
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """
    Attention-based policy for dynamic action spaces.

    Core idea
    ---------
    Project the state embedding into a query vector and each action embedding
    into a key vector.  The attention scores (dot product) form the action
    logits.  Illegal actions are masked to -inf before softmax so the
    resulting distribution is always valid.

    Parameters
    ----------
    cfg : Config
        Hyperparameter bundle from config.py.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.state_encoder  = StateEncoder(cfg)
        self.action_encoder = ActionEncoder(cfg)
        self.value_head     = ValueHead(cfg)

        # Scaling factor for dot-product attention (stabilises early training)
        self.scale = math.sqrt(cfg.state_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        state_feat:  torch.Tensor,   # (B, STATE_DIM)
        action_feat: torch.Tensor,   # (B, N, ACTION_DIM)
        action_mask: torch.Tensor,   # (B, N) bool  True=illegal
    ) -> PolicyOutput:
        """Compute policy outputs for a batch of (state, actions) pairs.

        The batch size B is typically 1 at inference and ≥1 during training.
        N (number of legal actions) may differ per sample; callers should
        pad to the same N within a batch (the mask handles the padding).
        """
        B, N, _ = action_feat.shape

        # ── Encode state ──────────────────────────────────────────────
        state_emb = self.state_encoder(state_feat)          # (B, STATE_DIM)

        # ── Encode actions ────────────────────────────────────────────
        action_keys = self.action_encoder(action_feat)      # (B, N, STATE_DIM)

        # ── Dot-product attention via bmm ─────────────────────────────
        #   query : (B, STATE_DIM) → (B, STATE_DIM, 1)
        #   keys  : (B, N, STATE_DIM)
        #   logits: bmm(keys, query) → (B, N, 1) → squeeze → (B, N)
        query  = state_emb.unsqueeze(2)                     # (B, STATE_DIM, 1)
        logits = torch.bmm(action_keys, query).squeeze(2)   # (B, N)
        logits = logits / self.scale                         # temperature scaling

        # ── Invalid-action masking ────────────────────────────────────
        if action_mask.any():
            logits = logits.masked_fill(action_mask, -1e9)

        # ── Distribution ──────────────────────────────────────────────
        probs     = F.softmax(logits, dim=-1)                # (B, N)
        log_probs = F.log_softmax(logits, dim=-1)            # (B, N)

        # ── Entropy (masked) ─────────────────────────────────────────
        # Use the distribution-safe version: -∑ p·log p  over legal actions
        safe_log = torch.where(
            action_mask,
            torch.zeros_like(log_probs),
            log_probs,
        )
        entropy = -(probs * safe_log).sum(dim=-1)            # (B,)

        # ── Value estimate ────────────────────────────────────────────
        value = self.value_head(state_emb)                   # (B, 1)

        return PolicyOutput(
            logits=logits,
            probs=probs,
            log_probs=log_probs,
            value=value,
            entropy=entropy,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        state_feat:  torch.Tensor,   # (1, STATE_DIM)
        action_feat: torch.Tensor,   # (1, N, ACTION_DIM)
        action_mask: torch.Tensor,   # (1, N) bool
        greedy: bool = False,
    ) -> tuple[int, float, float]:
        """Sample (or greedy-pick) one action.

        Returns
        -------
        action_idx : int
        log_prob   : float
        value      : float
        """
        out = self.forward(state_feat, action_feat, action_mask)
        if greedy:
            action_idx = int(out.probs.squeeze(0).argmax().item())
        else:
            # torch.multinomial is not fully supported on MPS for all dtypes/sizes.
            # Sampling on CPU from the probability vector is safe and cheap
            # (it's a single 1-D sample — negligible transfer overhead).
            probs_cpu = out.probs.squeeze(0).float().cpu()
            action_idx = int(torch.multinomial(probs_cpu, num_samples=1).item())
        lp  = out.log_probs[0, action_idx].item()
        val = out.value[0, 0].item()
        return action_idx, float(lp), float(val)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, cfg: Config) -> "PolicyNetwork":
        net = cls(cfg)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return net


# ---------------------------------------------------------------------------
# Quick tensor shape verification (run as __main__)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import Config
    cfg = Config()
    net = PolicyNetwork(cfg)
    net.eval()

    B, N = 4, 12
    s = torch.randn(B, cfg.state_dim)
    a = torch.randn(B, N, cfg.action_dim)
    m = torch.zeros(B, N, dtype=torch.bool)
    # Mask two actions per sample to exercise masking logic
    m[:, -2:] = True

    out = net(s, a, m)

    print(f"  logits     : {tuple(out.logits.shape)}   expected ({B}, {N})")
    print(f"  probs      : {tuple(out.probs.shape)}   expected ({B}, {N})")
    print(f"  log_probs  : {tuple(out.log_probs.shape)}   expected ({B}, {N})")
    print(f"  value      : {tuple(out.value.shape)}   expected ({B}, 1)")
    print(f"  entropy    : {tuple(out.entropy.shape)}   expected ({B},)")

    assert out.logits.shape    == (B, N),    f"logits shape mismatch: {out.logits.shape}"
    assert out.probs.shape     == (B, N),    f"probs shape mismatch"
    assert out.log_probs.shape == (B, N),    f"log_probs shape mismatch"
    assert out.value.shape     == (B, 1),    f"value shape mismatch"
    assert out.entropy.shape   == (B,),      f"entropy shape mismatch"

    # Masked logits must be -1e9 (or numerically equivalent)
    assert (out.logits[:, -2:] < -1e8).all(), "Masking not applied correctly"
    # Probs over valid actions must sum to 1
    valid_probs = out.probs[:, :-2].sum(dim=-1)
    assert torch.allclose(valid_probs, torch.ones(B), atol=1e-5), "Probs don't sum to 1"

    print("\n✓  All tensor shape assertions passed.")
    print("✓  bmm attention: (B, N, STATE_DIM) × (B, STATE_DIM, 1) → (B, N, 1) ✓")
    print("✓  Invalid-action masking verified.")
