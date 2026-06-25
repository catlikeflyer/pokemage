"""
agent.py
--------
Kaggle cabt Pokémon TCG AI Battle Challenge — submission entry point.

The Kaggle evaluator calls this once per turn:

    result = agent(obs, config)

Protocol
--------
  obs.step == 0  →  return a list of 60 card IDs  (your deck declaration)
  obs.step  > 0  →  return one action object from obs.actions

This file must be in the same directory as the project modules:
  config.py, card_data.py, env_wrapper.py, model.py, train.py, eval.py

Checkpoint
----------
Upload checkpoints/latest.pth as a Kaggle dataset named "pokemage-weights".
Add it as an input to your submission notebook. It will appear at:
    /kaggle/input/pokemage-weights/latest.pth

Local testing (no Kaggle):
    python agent.py --selftest
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from typing import Any

# ── Resolve submission directory so imports work regardless of cwd ─────────
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

import numpy as np
import torch

from card_data import init as _card_init
from config import Config
from env_wrapper import (
    STARTER_DECKS,
    Action,
    ActionType,
    AreaType,
    extract_action_features,
    _safe_hp_frac,
    _safe_energies,
    _safe_status,
    _safe_is_ex,
    _encode_card_scalar,
    _zone_pool,
    NUM_ACTION_TYPES,
    NUM_ZONES,
)
from model import PolicyNetwork
def get_device() -> torch.device:
    """Return the best available accelerator: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Deck to submit. Must match real competition card IDs when running on Kaggle.
# Swap STARTER_DECKS["dragapult_ex"] with your real 60-card list here.
SUBMIT_DECK_KEY: str = "dragapult_ex"

# Checkpoint search order: Kaggle dataset path first, local fallbacks second.
CHECKPOINT_PATHS: list[str] = [
    "/kaggle/input/pokemage-weights/latest.pth",          # Kaggle runtime
    os.path.join(_AGENT_DIR, "latest.pth"),               # bundled in zip
    os.path.join(_AGENT_DIR, "checkpoints", "latest.pth"),# local dev
    "./checkpoints/latest.pth",
]

# Safety margins (must stay well inside the 10-min Kaggle window)
MATCH_BUDGET_S: float   = 540.0   # 9 min hard ceiling
FALLBACK_THRESHOLD_S: float = 30.0  # drop to random if < 30s remain

# ---------------------------------------------------------------------------
# Module-level state (initialised ONCE when the file is imported)
# ---------------------------------------------------------------------------
_cfg:      Config | None        = None
_policy:   PolicyNetwork | None = None
_device:   torch.device | None  = None
_match_t0: float                = 0.0      # wall time of match start
_ready:    bool                 = False
_n_turns:  int                  = 0
_n_model:  int                  = 0
_n_random: int                  = 0


def _bootstrap() -> None:
    """Load config, vocab, and model weights exactly once."""
    global _cfg, _policy, _device, _ready

    if _ready:
        return

    _cfg    = Config()
    _device = get_device()

    # Card vocabulary
    _card_init(_cfg.card_csv)

    # Locate checkpoint
    ckpt_path: str | None = None
    for p in CHECKPOINT_PATHS:
        if os.path.exists(p):
            ckpt_path = p
            break

    _policy = PolicyNetwork(_cfg).to(_device)
    if ckpt_path:
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _policy.load_state_dict(state["model_state"])
            logger.warning("Loaded checkpoint: %s", ckpt_path)
        except Exception as exc:
            logger.warning("Checkpoint load failed (%s) – using random weights", exc)
    else:
        logger.warning(
            "No checkpoint found (searched %d paths) – using random policy",
            len(CHECKPOINT_PATHS),
        )

    _policy.eval()
    _ready = True


# ---------------------------------------------------------------------------
# Observation parsing  (mirrors LiveCabtEnv private methods)
# ---------------------------------------------------------------------------
_STATE_DIM      = None   # filled lazily from Config
_CARD_FEATS     = None
_PRIZE_CARDS    = None
_HAND_MAX       = None
_MAX_BENCH      = None


def _dims() -> tuple[int, int, int, int, int]:
    global _STATE_DIM, _CARD_FEATS, _PRIZE_CARDS, _HAND_MAX, _MAX_BENCH
    if _STATE_DIM is None:
        from config import STATE_DIM, CARD_SCALAR_FEATS, PRIZE_CARDS, HAND_MAX, MAX_BENCH
        _STATE_DIM    = STATE_DIM
        _CARD_FEATS   = CARD_SCALAR_FEATS
        _PRIZE_CARDS  = PRIZE_CARDS
        _HAND_MAX     = HAND_MAX
        _MAX_BENCH    = MAX_BENCH
    return _STATE_DIM, _CARD_FEATS, _PRIZE_CARDS, _HAND_MAX, _MAX_BENCH


def _extract_state(obs: Any) -> np.ndarray:
    """Convert a cabt Observation → (STATE_DIM,) float32."""
    STATE_DIM, CARD_FEATS, PRIZE_CARDS, HAND_MAX, MAX_BENCH = _dims()
    state = np.zeros(STATE_DIM, dtype=np.float32)
    try:
        from cg.api import AreaType as CgAreaType  # type: ignore
        ps  = obs.current.players[0]
        opp = obs.current.players[1]

        # Active Pokémon (offset 0)
        if ps.active:
            act = ps.active[0]
            state[:CARD_FEATS] = _encode_card_scalar(
                card_id=str(getattr(act, "card_id", "")),
                hp_frac=_safe_hp_frac(act),
                energies=_safe_energies(act),
                status=_safe_status(act),
                is_ex=_safe_is_ex(act),
            )

        # Bench (offset 64)
        bench_cards = [
            dict(
                card_id=str(getattr(p, "card_id", "")),
                hp_frac=_safe_hp_frac(p),
                energies=_safe_energies(p),
                status=_safe_status(p),
                is_ex=_safe_is_ex(p),
            )
            for p in (ps.bench or [])
        ]
        state[64 : 64 + CARD_FEATS] = _zone_pool(bench_cards)

        # Hand (offset 128)
        hand_cards = [
            dict(card_id=str(getattr(c, "card_id", "")),
                 hp_frac=1.0, energies=None, status=None, is_ex=False)
            for c in (ps.hand or [])
        ]
        state[128 : 128 + CARD_FEATS] = _zone_pool(hand_cards)

        # Opponent active (offset 256)
        if opp.active:
            oact = opp.active[0]
            state[256 : 256 + CARD_FEATS] = _encode_card_scalar(
                card_id=str(getattr(oact, "card_id", "")),
                hp_frac=_safe_hp_frac(oact),
                energies=_safe_energies(oact),
                status=_safe_status(oact),
                is_ex=_safe_is_ex(oact),
            )

        # Board scalars (offset 448)
        state[448] = len(ps.prize or [])  / PRIZE_CARDS
        state[449] = len(opp.prize or []) / PRIZE_CARDS
        state[450] = len(ps.hand or [])   / HAND_MAX
        state[451] = len(ps.bench or [])  / MAX_BENCH
        state[452] = len(opp.bench or []) / MAX_BENCH

    except Exception as exc:
        logger.warning("_extract_state error: %s", exc)

    return state


def _get_legal_actions(obs: Any) -> list[Action]:
    """Parse obs.actions into internal Action objects."""
    actions: list[Action] = []
    try:
        for raw in (obs.actions or []):
            atype = ActionType(min(int(getattr(raw, "type",     9)), NUM_ACTION_TYPES - 1))
            src   = AreaType (min(int(getattr(raw, "src_area",  0)), NUM_ZONES - 1))
            tgt   = AreaType (min(int(getattr(raw, "dst_area",  0)), NUM_ZONES - 1))
            cid   = str(getattr(raw, "card_id", ""))
            actions.append(Action(
                action_type=atype, source_zone=src, target_zone=tgt,
                card_id=cid,
                source_idx=int(getattr(raw, "src_idx", 0)),
                target_idx=int(getattr(raw, "dst_idx", 0)),
                extra={"raw": raw},
            ))
    except Exception as exc:
        logger.warning("_get_legal_actions error: %s", exc)

    if not actions:
        actions.append(Action(ActionType.PASS, AreaType.HAND, AreaType.HAND))
    return actions


def _action_to_cabt(action: Action) -> Any:
    """Convert internal Action back to the raw cabt object (or index fallback)."""
    raw = action.extra.get("raw")
    return raw if raw is not None else action.source_idx


def _random_action(obs: Any) -> Any:
    """Instantly return a random legal action (fallback)."""
    try:
        acts = list(obs.actions or [])
        return random.choice(acts) if acts else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def _model_action(obs: Any) -> Any:
    """Run the policy forward pass and return the chosen cabt action object."""
    state_feat   = _extract_state(obs)
    legal_actions = _get_legal_actions(obs)
    action_feat, action_mask = extract_action_features(legal_actions)

    s_t = torch.from_numpy(state_feat).unsqueeze(0).to(_device)   # (1, STATE_DIM)
    a_t = torch.from_numpy(action_feat).unsqueeze(0).to(_device)  # (1, N, ACTION_DIM)
    m_t = torch.from_numpy(action_mask).unsqueeze(0).to(_device)  # (1, N)

    with torch.no_grad():
        action_idx, _, _ = _policy.act(s_t, a_t, m_t)

    chosen = legal_actions[action_idx]
    return _action_to_cabt(chosen)


# ---------------------------------------------------------------------------
# Main entry point — called by Kaggle evaluator every turn
# ---------------------------------------------------------------------------

def agent(obs: Any, config: Any) -> Any:
    """
    Kaggle cabt agent entry point.

    Parameters
    ----------
    obs    : cabt Observation object
    config : cabt environment config

    Returns
    -------
    On step 0 : list[str]  — 60-card deck
    Otherwise : raw cabt action object (or int index as fallback)
    """
    global _match_t0, _n_turns, _n_model, _n_random

    # ── One-time initialization ───────────────────────────────────────
    _bootstrap()

    # ── Step 0: submit deck ───────────────────────────────────────────
    step = getattr(obs, "step", None)
    if step == 0 or step is None:
        _match_t0 = time.monotonic()
        logger.warning("Match start – deck submitted (%s)", SUBMIT_DECK_KEY)
        return STARTER_DECKS.get(SUBMIT_DECK_KEY, STARTER_DECKS["dragapult_ex"])

    # ── Time accounting ───────────────────────────────────────────────
    elapsed        = time.monotonic() - _match_t0
    time_remaining = getattr(obs, "remaining_time", MATCH_BUDGET_S - elapsed)
    _n_turns += 1

    # ── Fallback conditions ───────────────────────────────────────────
    use_fallback = (
        time_remaining < FALLBACK_THRESHOLD_S
        or elapsed > MATCH_BUDGET_S
    )

    if use_fallback:
        _n_random += 1
        logger.warning(
            "Turn %d – FALLBACK (elapsed=%.1fs, remaining=%.1fs)",
            _n_turns, elapsed, time_remaining,
        )
        return _random_action(obs)

    # ── Model inference ───────────────────────────────────────────────
    try:
        _n_model += 1
        return _model_action(obs)
    except Exception as exc:
        _n_random += 1
        logger.warning("Model inference failed (%s) – random fallback", exc)
        return _random_action(obs)


# ---------------------------------------------------------------------------
# Self-test  (python agent.py --selftest)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="agent.py self-test")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if not args.selftest:
        parser.print_help()
        sys.exit(0)

    print("=" * 60)
    print("  agent.py self-test (mock obs, no Kaggle SDK required)")
    print("=" * 60)

    # ── Bootstrap ──────────────────────────────────────────────────────
    _bootstrap()
    print(f"  ✓ Bootstrap complete | device={_device} | "
          f"checkpoint={'found' if any(os.path.exists(p) for p in CHECKPOINT_PATHS) else 'NOT FOUND (random weights)'}")

    # ── Mock obs for deck step ─────────────────────────────────────────
    class _MockObs:
        step           = 0
        actions        = []
        remaining_time = 540.0
        class current:
            class players:
                pass

    deck = agent(_MockObs(), None)
    assert isinstance(deck, list) and len(deck) > 0, f"Expected non-empty deck, got {deck!r}"
    print(f"  ✓ Step 0 → deck submitted ({len(deck)} cards, first={deck[0]!r})")
    if len(deck) < 60:
        print(f"  ⚠ Mock deck has {len(deck)} cards (real decks must be 60 – update STARTER_DECKS)")

    # ── Mock obs for turn step ─────────────────────────────────────────
    class _MockAction:
        type     = 9   # END_TURN
        src_area = 0
        dst_area = 0
        card_id  = ""
        src_idx  = 0
        dst_idx  = 0

    class _MockTurnObs:
        step           = 1
        remaining_time = 400.0
        actions        = [_MockAction(), _MockAction()]
        class current:
            class players:
                pass

    result = agent(_MockTurnObs(), None)
    print(f"  ✓ Step 1 → action returned: {result!r}")
    print(f"  ✓ Stats: turns={_n_turns}, model={_n_model}, random={_n_random}")
    print("=" * 60)
    print("  Self-test PASSED ✓")
    print("=" * 60)
