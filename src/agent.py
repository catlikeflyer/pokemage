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
if "__file__" in globals():
    _AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    _AGENT_DIR = "/kaggle_simulations/agent"
    if not os.path.isdir(_AGENT_DIR):
        _AGENT_DIR = os.path.dirname(os.path.abspath(sys.argv[0])) if (globals().get("sys") and sys.argv) else os.getcwd()

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


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_hp_frac(pokemon: Any) -> float:
    try:
        hp = int(safe_get(pokemon, "hp", 0))
        max_hp = int(safe_get(pokemon, "max_hp", 1))
        return max(0.0, min(1.0, hp / max(max_hp, 1)))
    except Exception:
        return 1.0


def _safe_energies(pokemon: Any) -> list[int]:
    try:
        energy = safe_get(pokemon, "energy")
        return [int(safe_get(energy, t, 0)) for t in
                ("fire", "water", "grass", "lightning", "fighting", "psychic")]
    except Exception:
        return [0] * 6


def _safe_status(pokemon: Any) -> list[int]:
    try:
        s = safe_get(pokemon, "status") or ""
        return [
            int("burn"    in str(s).lower()),
            int("poison"  in str(s).lower()),
            int("confuse" in str(s).lower()),
            int("paraly"  in str(s).lower()),
            int("asleep"  in str(s).lower()),
        ]
    except Exception:
        return [0] * 5


def _safe_is_ex(pokemon: Any) -> bool:
    try:
        name = str(safe_get(pokemon, "name", "") or "")
        return " ex" in name.lower() or name.endswith(" EX")
    except Exception:
        return False


def _extract_state(obs: Any) -> np.ndarray:
    """Convert a cabt Observation → (STATE_DIM,) float32."""
    STATE_DIM, CARD_FEATS, PRIZE_CARDS, HAND_MAX, MAX_BENCH = _dims()
    state = np.zeros(STATE_DIM, dtype=np.float32)
    try:
        current = safe_get(obs, "current")
        if current is None:
            return state

        players = safe_get(current, "players")
        if not players:
            return state

        our_idx  = int(safe_get(current, "yourIndex", 0))
        opp_idx  = 1 - our_idx
        ps  = players[our_idx]
        opp = players[opp_idx]

        # Active Pokémon (offset 0)
        active = safe_get(ps, "active")
        if active:
            act = active[0]
            state[:CARD_FEATS] = _encode_card_scalar(
                card_id=str(safe_get(act, "card_id", "")),
                hp_frac=_safe_hp_frac(act),
                energies=_safe_energies(act),
                status=_safe_status(act),
                is_ex=_safe_is_ex(act),
            )

        # Bench (offset 64)
        bench = safe_get(ps, "bench") or []
        bench_cards = [
            dict(
                card_id=str(safe_get(p, "card_id", "")),
                hp_frac=_safe_hp_frac(p),
                energies=_safe_energies(p),
                status=_safe_status(p),
                is_ex=_safe_is_ex(p),
            )
            for p in bench
        ]
        state[64 : 64 + CARD_FEATS] = _zone_pool(bench_cards)

        # Hand (offset 128)
        hand = safe_get(ps, "hand") or []
        hand_cards = [
            dict(card_id=str(safe_get(c, "card_id", "")),
                 hp_frac=1.0, energies=None, status=None, is_ex=False)
            for c in hand
        ]
        state[128 : 128 + CARD_FEATS] = _zone_pool(hand_cards)

        # Opponent active (offset 256)
        opp_active = safe_get(opp, "active")
        if opp_active:
            oact = opp_active[0]
            state[256 : 256 + CARD_FEATS] = _encode_card_scalar(
                card_id=str(safe_get(oact, "card_id", "")),
                hp_frac=_safe_hp_frac(oact),
                energies=_safe_energies(oact),
                status=_safe_status(oact),
                is_ex=_safe_is_ex(oact),
            )

        # Board scalars (offset 448)
        state[448] = len(safe_get(ps, "prize") or [])  / PRIZE_CARDS
        state[449] = len(safe_get(opp, "prize") or []) / PRIZE_CARDS
        state[450] = len(safe_get(ps, "hand") or [])   / HAND_MAX
        state[451] = len(safe_get(ps, "bench") or [])  / MAX_BENCH
        state[452] = len(safe_get(opp, "bench") or []) / MAX_BENCH
        state[453] = int(safe_get(current, "turn", 0)) / 100.0  # normalised turn

    except Exception as exc:
        logger.warning("_extract_state error: %s", exc)

    return state


def _get_legal_actions(obs: Any) -> list[Action]:
    """Parse obs.actions or obs.select.option into internal Action objects."""
    actions: list[Action] = []
    raw_actions = []
    
    # Check if select.option exists (real Kaggle env)
    sel = safe_get(obs, "select")
    if sel is not None:
        raw_actions = safe_get(sel, "option") or []
    
    # Fallback to actions (mock/self-test)
    if not raw_actions:
        raw_actions = safe_get(obs, "actions") or []

    try:
        for raw in raw_actions:
            atype = ActionType(min(int(safe_get(raw, "type", 9)), NUM_ACTION_TYPES - 1))
            src   = AreaType (min(int(safe_get(raw, "src_area", 0)), NUM_ZONES - 1))
            tgt   = AreaType (min(int(safe_get(raw, "dst_area", 0)), NUM_ZONES - 1))
            cid   = str(safe_get(raw, "card_id", ""))
            actions.append(Action(
                action_type=atype, source_zone=src, target_zone=tgt,
                card_id=cid,
                source_idx=int(safe_get(raw, "src_idx", 0)),
                target_idx=int(safe_get(raw, "dst_idx", 0)),
                extra={"raw": raw},
            ))
    except Exception as exc:
        logger.warning("_get_legal_actions error: %s", exc)

    if not actions:
        actions.append(Action(ActionType.PASS, AreaType.HAND, AreaType.HAND))
    return actions


def get_action_list(action_idx: int, max_count: int, num_options: int) -> list[int]:
    selected = [action_idx]
    for i in range(num_options):
        if len(selected) >= max_count:
            break
        if i not in selected:
            selected.append(i)
    while len(selected) < max_count:
        selected.append(0)
    return selected


def _random_action(obs: Any) -> list[int]:
    """Instantly return a random legal action (fallback) as a list[int]."""
    try:
        sel = safe_get(obs, "select")
        if sel is not None:
            opts = safe_get(sel, "option") or []
            max_count = int(safe_get(sel, "maxCount", 1) or 1)
            if opts:
                idxs = list(range(len(opts)))
                return random.sample(idxs, min(max_count, len(idxs)))
        
        # Fallback for mock/self-test if options aren't in select
        acts = safe_get(obs, "actions") or []
        if acts:
            return [random.choice(range(len(acts)))]
        return [0]
    except Exception:
        return [0]


def _model_action_idx(obs: Any) -> int:
    """Run the policy forward pass and return the chosen action index."""
    state_feat   = _extract_state(obs)
    legal_actions = _get_legal_actions(obs)
    action_feat, action_mask = extract_action_features(legal_actions)

    s_t = torch.from_numpy(state_feat).unsqueeze(0).to(_device)   # (1, STATE_DIM)
    a_t = torch.from_numpy(action_feat).unsqueeze(0).to(_device)  # (1, N, ACTION_DIM)
    m_t = torch.from_numpy(action_mask).unsqueeze(0).to(_device)  # (1, N)

    with torch.no_grad():
        action_idx, _, _ = _policy.act(s_t, a_t, m_t)

    return action_idx


def agent(obs: Any, config: Any) -> Any:
    """
    Kaggle cabt agent entry point.
    """
    global _match_t0, _n_turns, _n_model, _n_random

    # ── One-time initialization ───────────────────────────────────────
    _bootstrap()

    # ── Step 0: submit deck ───────────────────────────────────────────
    step = safe_get(obs, "step")
    if step == 0 or step is None:
        _match_t0 = time.monotonic()
        logger.warning("Match start – deck submitted (%s)", SUBMIT_DECK_KEY)
        
        # In Kaggle/evaluator environment, we must return a valid 60-card deck with integer IDs.
        # Check if STARTER_DECKS contains integer IDs, else build one from the CSV.
        raw_deck = STARTER_DECKS.get(SUBMIT_DECK_KEY, [])
        if raw_deck and isinstance(raw_deck[0], int):
            deck = list(raw_deck[:60])
            while len(deck) < 60:
                deck.append(deck[-1])
            return deck
        
        try:
            from card_data import build_valid_deck
            csv_path = _cfg.card_csv if _cfg else "./EN_Card_Data.csv"
            deck = build_valid_deck(csv_path=csv_path, size=60)
            logger.warning("Built valid 60-card integer ID deck: first 3 = %s", deck[:3])
            return deck
        except Exception as exc:
            logger.warning("build_valid_deck failed (%s) – using hardcoded fallback", exc)
            fallback_deck = [22] * 4 + ([1, 2, 3, 4, 5, 6, 7, 8] * 8)[:56]
            return fallback_deck

    # ── Time accounting ───────────────────────────────────────────────
    elapsed        = time.monotonic() - _match_t0
    time_remaining = safe_get(obs, "remaining_time")
    if time_remaining is None:
        time_remaining = MATCH_BUDGET_S - elapsed
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
        action_idx = _model_action_idx(obs)
        
        # Get maxCount from obs.select
        sel = safe_get(obs, "select")
        max_count = int(safe_get(sel, "maxCount", 1) or 1) if sel else 1
        
        # Get number of options
        legal_actions = _get_legal_actions(obs)
        num_options = len(legal_actions)
        
        # Build list[int] of indices
        action_list = get_action_list(action_idx, max_count, num_options)
        return action_list
        
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

    class _MockPlayer:
        active = []
        bench = []
        hand = []
        prize = []

    # ── Mock obs for deck step ─────────────────────────────────────────
    class _MockObs:
        step           = 0
        actions        = []
        remaining_time = 540.0
        class current:
            yourIndex = 0
            players = [_MockPlayer(), _MockPlayer()]

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
            yourIndex = 0
            players = [_MockPlayer(), _MockPlayer()]

    result = agent(_MockTurnObs(), None)
    print(f"  ✓ Step 1 → action returned: {result!r}")
    print(f"  ✓ Stats: turns={_n_turns}, model={_n_model}, random={_n_random}")
    print("=" * 60)
    print("  Self-test PASSED ✓")
    print("=" * 60)
