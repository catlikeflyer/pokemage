"""
env_wrapper.py
--------------
Two environment backends behind a common interface:

  MockCabtEnv  – pure-Python simulator; works without libcg.so / cg.api.
  LiveCabtEnv  – wraps kaggle-environments + cg.api for real training.

Both expose:
    reset()  -> (state_feat: np.ndarray, action_feat: np.ndarray, action_mask: np.ndarray)
    step(action_idx) -> (state_feat, action_feat, action_mask, reward, done)
    close()
    current_player: int  (0 or 1)

State feature vector shape:  (STATE_DIM,)     = (512,)
Action feature matrix shape: (N_actions, ACTION_DIM) = (N, 128)   N varies per turn
Action mask shape:           (N_actions,)  bool  True == illegal (always False here; masking done in network)

Starter decks (official IDs from competition)
--------------------------------------------
The four starter decks are listed here as 60-card lists using mock card IDs
so the pipeline works without EN_Card_Data.csv.  Swap these for real
competition IDs when running on Kaggle.
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, NamedTuple

import numpy as np

from config import (
    ACTION_DIM,
    CARD_EMBED_DIM,
    CARD_SCALAR_FEATS,
    DECK_MAX,
    HAND_MAX,
    MAX_BENCH,
    NUM_ACTION_TYPES,
    NUM_STATUS,
    NUM_TYPES,
    NUM_ZONES,
    PRIZE_CARDS,
    STATE_DIM,
)
from card_data import card_id_to_idx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Starter decks  (mock IDs → replace with real competition IDs on Kaggle)
# ---------------------------------------------------------------------------
STARTER_DECKS: dict[str, list[str]] = {
    "dragapult_ex": [
        # 4× Dreepy, 4× Drakloak, 4× Dragapult ex, 2× Pidgey, 2× Pidgeot ex,
        # 2× Dudunsparce, 4× Colress's Experiment, 4× Iono, 3× Boss's Orders,
        # 2× Penny, 4× Ultra Ball, 4× Nest Ball, 3× Rare Candy,
        # 2× Counter Catcher, 2× Escape Rope, 2× Lost Vacuum, 1× Pal Pad,
        # 4× Twin Energy, 4× Psychic Energy, 4× Grass Energy
        *[f"DRAG_{i:03d}" for i in range(4)],   # Dreepy ×4
        *[f"DRAG_{i:03d}" for i in range(4, 8)],  # Drakloak ×4
        *[f"DRAG_{i:03d}" for i in range(8, 12)], # Dragapult ex ×4
        *[f"SUPP_{i:03d}" for i in range(20)],    # supporters / items / energy ×44
    ][:60],
    "mega_lucario_ex": [*[f"LUCI_{i:03d}" for i in range(20)], *[f"SUPP_{i:03d}" for i in range(40)]][:60],
    "mega_abomasnow_ex": [*[f"ABOM_{i:03d}" for i in range(20)], *[f"SUPP_{i:03d}" for i in range(40)]][:60],
    "iono": [*[f"IONO_{i:03d}" for i in range(20)], *[f"SUPP_{i:03d}" for i in range(40)]][:60],
}


# ---------------------------------------------------------------------------
# Shared action representation
# ---------------------------------------------------------------------------
class AreaType(IntEnum):
    HAND    = 0
    BENCH   = 1
    ACTIVE  = 2
    DISCARD = 3
    PRIZE   = 4
    DECK    = 5
    STADIUM = 6
    LOOKING = 7


class ActionType(IntEnum):
    ATTACK        = 0
    PLAY_BASIC    = 1
    EVOLVE        = 2
    RETREAT       = 3
    USE_TRAINER   = 4
    ATTACH_ENERGY = 5
    USE_ABILITY   = 6
    PASS          = 7
    TAKE_PRIZE    = 8
    END_TURN      = 9


@dataclass
class Action:
    action_type:  ActionType
    source_zone:  AreaType
    target_zone:  AreaType
    card_id:      str = ""
    source_idx:   int = 0
    target_idx:   int = 0
    extra:        dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction utilities (shared by both env backends)
# ---------------------------------------------------------------------------

def _encode_card_scalar(
    card_id: str,
    hp_frac: float = 1.0,
    energies: list[int] | None = None,
    status: list[int] | None = None,
    is_ex: bool = False,
) -> np.ndarray:
    """Produce a (CARD_SCALAR_FEATS,) float32 array for one card."""
    if energies is None:
        energies = [0] * NUM_TYPES
    if status is None:
        status = [0] * NUM_STATUS
    feats = np.array(
        [hp_frac, float(is_ex)] + energies[:NUM_TYPES] + status[:NUM_STATUS],
        dtype=np.float32,
    )
    return feats  # shape: (14,)


def _zone_pool(cards: list[dict]) -> np.ndarray:
    """Mean-pool a list of card feature dicts → (CARD_SCALAR_FEATS,) array.

    Each dict must have keys: card_id, hp_frac, energies, status, is_ex.
    Returns zeros if the zone is empty.
    """
    if not cards:
        return np.zeros(CARD_SCALAR_FEATS, dtype=np.float32)
    feats = np.stack(
        [_encode_card_scalar(**c) for c in cards], axis=0
    )  # (K, CARD_SCALAR_FEATS)
    return feats.mean(axis=0)


def _encode_action(action: Action) -> np.ndarray:
    """Encode a single Action → (ACTION_DIM,) float32 vector.

    Layout:
      [0:10]   action_type one-hot
      [10:18]  source_zone one-hot
      [18:26]  target_zone one-hot
      [26:27]  source_idx / DECK_MAX  (normalised)
      [27:28]  target_idx / MAX_BENCH (normalised)
      [28:92]  card embedding placeholder (zeros; real embed in model)
      [92:128] padding zeros
    """
    vec = np.zeros(ACTION_DIM, dtype=np.float32)
    # action type (one-hot, 10 dims)
    if 0 <= action.action_type < NUM_ACTION_TYPES:
        vec[action.action_type] = 1.0
    # source zone (one-hot, 8 dims)
    if 0 <= action.source_zone < NUM_ZONES:
        vec[NUM_ACTION_TYPES + action.source_zone] = 1.0
    # target zone (one-hot, 8 dims)
    if 0 <= action.target_zone < NUM_ZONES:
        vec[NUM_ACTION_TYPES + NUM_ZONES + action.target_zone] = 1.0
    # normalised indices
    vec[26] = action.source_idx / max(DECK_MAX, 1)
    vec[27] = action.target_idx / max(MAX_BENCH, 1)
    # card_id hash feature (crude but stable without embedding layer here)
    vec[28] = (card_id_to_idx(action.card_id) % 256) / 255.0
    return vec


def extract_action_features(actions: list[Action]) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of legal actions to model inputs.

    Returns
    -------
    action_feat : (N, ACTION_DIM) float32
    action_mask : (N,)           bool  – all False (legal); masking handled in network
    """
    n = len(actions)
    feat = np.stack([_encode_action(a) for a in actions], axis=0)  # (N, ACTION_DIM)
    mask = np.zeros(n, dtype=bool)                                  # all legal
    return feat, mask


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class CabtEnvBase(ABC):
    current_player: int = 0

    @abstractmethod
    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Start a new game. Returns (state_feat, action_feat, action_mask)."""

    @abstractmethod
    def step(self, action_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """Apply action_idx. Returns (state_feat, action_feat, action_mask, reward, done)."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""


# ---------------------------------------------------------------------------
# Mock environment  (no SDK required)
# ---------------------------------------------------------------------------
class MockCabtEnv(CabtEnvBase):
    """Minimal Pokémon TCG simulator for pipeline smoke-testing.

    Game model:
    - Two players each with a deck of 60 cards (sampled from a starter deck).
    - Each turn the agent picks from 4–20 randomly generated legal actions.
    - The game ends after MAX_TURNS turns (configurable) or when a player
      runs out of prizes; winner is determined by coin flip at terminal state
      to avoid any strategic bias during early training.
    - Reward: +1.0 (win), -1.0 (loss), 0.0 (draw/ongoing).
    """

    MAX_TURNS: int = 40  # safety cap per episode

    def __init__(self, deck_key: str = "dragapult_ex", seed: int | None = None):
        self._rng = random.Random(seed)
        self._deck_template = STARTER_DECKS.get(deck_key, STARTER_DECKS["dragapult_ex"])
        self._turn: int = 0
        self._prizes: list[int] = [PRIZE_CARDS, PRIZE_CARDS]  # [player0, player1]
        self._done: bool = True
        self._state_vec: np.ndarray = np.zeros(STATE_DIM, dtype=np.float32)
        self.current_player: int = 0

    # ------------------------------------------------------------------
    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._turn = 0
        self._prizes = [PRIZE_CARDS, PRIZE_CARDS]
        self._done = False
        self.current_player = 0
        return self._observe()

    # ------------------------------------------------------------------
    def step(self, action_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        if self._done:
            raise RuntimeError("step() called on finished episode; call reset() first.")

        self._turn += 1

        # Simulate prize loss on attacking actions (crude model)
        if action_idx == 0 and self._rng.random() < 0.35:  # attack hit
            opp = 1 - self.current_player
            self._prizes[opp] = max(0, self._prizes[opp] - 1)

        # Check terminal conditions
        reward = 0.0
        if self._prizes[1 - self.current_player] <= 0:
            self._done = True
            reward = 1.0
        elif self._prizes[self.current_player] <= 0:
            self._done = True
            reward = -1.0
        elif self._turn >= self.MAX_TURNS:
            self._done = True
            # mild reward shaping: win if ahead on prizes
            diff = self._prizes[1 - self.current_player] - self._prizes[self.current_player]
            reward = float(np.sign(diff)) * 0.5

        # Alternate turns
        if not self._done:
            self.current_player = 1 - self.current_player

        state, a_feat, a_mask = self._observe()
        return state, a_feat, a_mask, reward, self._done

    # ------------------------------------------------------------------
    def close(self) -> None:
        pass  # nothing to free

    # ------------------------------------------------------------------
    def _observe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a synthetic but structurally valid observation."""
        rng = self._rng

        # ── State features ────────────────────────────────────────────
        #  [0:64]   active Pokémon scalar feats (zero-padded to 64)
        #  [64:192] bench (5 × 14 = 70 → padded to 128)
        #  [192:320] hand (mean-pooled, 14 → padded to 128)
        #  [320:448] opponent active + bench pooled (128)
        #  [448:512] board scalars (64)
        state = np.zeros(STATE_DIM, dtype=np.float32)

        # Active pokemon
        active_feats = _encode_card_scalar(
            card_id=rng.choice(self._deck_template),
            hp_frac=rng.uniform(0.3, 1.0),
            energies=[rng.randint(0, 3) for _ in range(NUM_TYPES)],
            status=[rng.randint(0, 1) for _ in range(NUM_STATUS)],
            is_ex=rng.random() < 0.3,
        )
        state[:CARD_SCALAR_FEATS] = active_feats

        # Board scalars (at offset 448)
        me = self.current_player
        state[448] = self._turn / self.MAX_TURNS
        state[449] = (self._prizes[me] - self._prizes[1 - me]) / PRIZE_CARDS
        state[450] = self._prizes[me] / PRIZE_CARDS
        state[451] = self._prizes[1 - me] / PRIZE_CARDS
        state[452] = float(me)   # player perspective flag

        # ── Legal actions ─────────────────────────────────────────────
        n_actions = rng.randint(4, 20)
        actions: list[Action] = []
        for _ in range(n_actions):
            atype = ActionType(rng.randint(0, NUM_ACTION_TYPES - 1))
            src   = AreaType(rng.randint(0, NUM_ZONES - 1))
            tgt   = AreaType(rng.randint(0, NUM_ZONES - 1))
            cid   = rng.choice(self._deck_template)
            actions.append(Action(
                action_type=atype,
                source_zone=src,
                target_zone=tgt,
                card_id=cid,
                source_idx=rng.randint(0, 9),
                target_idx=rng.randint(0, MAX_BENCH - 1),
            ))
        # Always include END_TURN to avoid no-legal-move edge case
        actions.append(Action(
            action_type=ActionType.END_TURN,
            source_zone=AreaType.HAND,
            target_zone=AreaType.HAND,
        ))

        a_feat, a_mask = extract_action_features(actions)
        return state, a_feat, a_mask


# ---------------------------------------------------------------------------
# Live environment  (requires kaggle-environments + cg.api)
# ---------------------------------------------------------------------------
class LiveCabtEnv(CabtEnvBase):
    """Wraps the official cabt engine from kaggle-environments.

    Import is deferred so the module loads fine even without the SDK.
    """

    def __init__(self, deck_key: str = "dragapult_ex"):
        try:
            from kaggle_environments import make  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "kaggle-environments is not installed. "
                "Use MockCabtEnv or install the competition SDK."
            ) from exc

        self._deck_key = deck_key
        self._deck = STARTER_DECKS.get(deck_key, STARTER_DECKS["dragapult_ex"])
        self._env = make("cabt", debug=False)
        self._trainer = None
        self._done = True
        self._obs = None
        self.current_player = 0

    # ------------------------------------------------------------------
    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Start a new episode.

        cabt protocol (trainer API):
          Step 0  – both players submit their 60-card decks (integer card IDs).
          Step 1+ – cabt runs the setup phase (draw hands, place active, prizes).
                    obs.current remains None until setup is complete.
          Game    – obs.current is populated; normal play begins.
        """
        from kaggle_environments import make  # deferred to keep import lazy
        self._env     = make("cabt", debug=False)
        self._trainer = self._env.train([None, "random"])
        self._obs     = self._trainer.reset()
        self._done    = False
        self.current_player = 0

        # ── Build a valid integer-ID deck from EN_Card_Data.csv ──────────────
        deck = self._build_integer_deck()
        logger.info("Submitting deck (%d cards, first=%s)", len(deck), deck[:3])

        # ── Step 0: deck submission ──────────────────────────────────────────
        try:
            self._obs, _, done, _ = self._trainer.step(deck)
            self._done = bool(done)
        except Exception as exc:
            logger.warning("Deck submission failed: %s", exc)

        # ── Setup phase: loop until the game has actually started ─────────────
        # cabt protocol after deck submission:
        #   1. obs.select may be populated → player must pick Active Pokémon
        #   2. obs.actions may be populated → normal action step
        #   3. Otherwise send None → engine auto-advances (draws, prizes etc.)
        # We keep stepping until handCount > 0 or turn > 0.
        MAX_SETUP = 60
        for setup_step in range(MAX_SETUP):
            if self._done:
                logger.debug("Episode ended during setup at step %d", setup_step)
                break

            # ── Check whether the game has actually begun ─────────────────
            cur = getattr(self._obs, "current", None)
            if cur is not None:
                try:
                    our_idx  = int(getattr(cur, "yourIndex", 0))
                    ps       = cur.players[our_idx]
                    hand_cnt = int(getattr(ps, "handCount", 0))
                    turn_num = int(getattr(cur, "turn", 0))
                    if hand_cnt > 0 or turn_num > 0:
                        logger.info(
                            "Game started: setup_step=%d turn=%d hand=%d",
                            setup_step, turn_num, hand_cnt,
                        )
                        break
                except Exception:
                    pass   # keep looping

            # ── Choose action to advance setup ────────────────────────────
            # Priority:
            #   1. obs.select  – cabt asking us to pick Active/bench card
            #   2. obs.actions – normal action list (first legal action)
            #   3. None        – safe no-op; engine auto-advances turn
            try:
                setup_action: Any = None

                sel = getattr(self._obs, "select", None)
                if sel is not None:
                    try:
                        sel_list = list(sel)
                        if sel_list:
                            setup_action = sel_list[0]
                            logger.debug(
                                "Setup step %d: selecting from obs.select (len=%d)",
                                setup_step, len(sel_list),
                            )
                    except Exception:
                        pass

                if setup_action is None:
                    raw_actions = getattr(self._obs, "actions", None) or []
                    try:
                        raw_actions = list(raw_actions)
                    except Exception:
                        raw_actions = []
                    if raw_actions:
                        setup_action = raw_actions[0]
                        logger.debug(
                            "Setup step %d: using first of %d obs.actions",
                            setup_step, len(raw_actions),
                        )
                    else:
                        logger.debug(
                            "Setup step %d: no select/actions — sending None",
                            setup_step,
                        )

                self._obs, _, done, _ = self._trainer.step(setup_action)
                self._done = bool(done)
            except Exception as exc:
                logger.debug("Setup step %d failed: %s", setup_step, exc)
                break
        else:
            logger.warning(
                "Game did not start after %d setup steps "
                "— returning zero state", MAX_SETUP
            )

        return self._parse_obs(self._obs)

    # ------------------------------------------------------------------
    def _build_integer_deck(self) -> list[int]:
        """Return a 60-card deck as a list of integer card IDs.

        Priority:
          1. STARTER_DECKS entry for self._deck_key if it already contains ints.
          2. build_valid_deck() — reads EN_Card_Data.csv, ensures at least one
             Basic Pokémon so cabt does not auto-forfeit the game.
          3. Hardcoded fallback: 4× ID 22 (Hippopotas) + 56× Basic Energy.
        """
        # 1. If STARTER_DECKS has been set to integer IDs, use them directly
        raw = STARTER_DECKS.get(self._deck_key, [])
        if raw and isinstance(raw[0], int):
            ids = list(raw[:60])
            while len(ids) < 60:
                ids.append(ids[-1])
            return ids[:60]

        # 2. Build from CSV (includes Basic Pokémon so cabt doesn't forfeit)
        try:
            from card_data import build_valid_deck
            from config import Config
            cfg = Config()
            return build_valid_deck(csv_path=cfg.card_csv, size=60)
        except Exception as exc:
            logger.warning("build_valid_deck() failed: %s", exc)

        # 3. Last resort — hardcoded IDs from EN_Card_Data.csv inspection
        # ID 22 = Hippopotas (Basic Pokémon), IDs 1-8 = Basic Energy
        logger.warning("Using hardcoded fallback deck (ID 22 + Basic Energy)")
        return [22] * 4 + ([1, 2, 3, 4, 5, 6, 7, 8] * 8)[:56]


    # ------------------------------------------------------------------
    def step(self, action_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        if self._done:
            raise RuntimeError("step() called on finished episode.")

        actions = self._get_legal_actions(self._obs)
        chosen  = actions[action_idx] if action_idx < len(actions) else actions[-1]

        raw_action = self._action_to_cabt(chosen)
        self._obs, reward, done, info = self._trainer.step(raw_action)
        self._done = done
        state, a_feat, a_mask = self._parse_obs(self._obs)
        return state, a_feat, a_mask, float(reward or 0.0), bool(done)

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            if self._env is not None:
                self._env.reset()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _parse_obs(self, obs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert a cabt Observation object to model inputs."""
        state   = self._extract_state(obs)
        actions = self._get_legal_actions(obs)
        if not actions:
            # Should not happen; insert a dummy pass action
            actions = [Action(ActionType.PASS, AreaType.HAND, AreaType.HAND)]
        a_feat, a_mask = extract_action_features(actions)
        return state, a_feat, a_mask

    def _extract_state(self, obs: Any) -> np.ndarray:
        """Map cabt Observation → (STATE_DIM,) float32 array.

        Confirmed obs.current structure (from live diagnostic):
            obs.current.players    – list of two player-state Structs
            obs.current.yourIndex  – integer index (0 or 1) for our player
            obs.current.turn       – turn counter
            obs.current.result     – game result (None while ongoing)
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)
        try:
            cur = getattr(obs, "current", None)
            if cur is None:
                return state   # setup phase not complete yet

            our_idx  = int(getattr(cur, "yourIndex", 0))
            opp_idx  = 1 - our_idx
            players  = cur.players
            ps       = players[our_idx]
            opp      = players[opp_idx]

            # ─── Active Pokémon (offset 0)
            if ps.active:
                act = ps.active[0]
                state[:CARD_SCALAR_FEATS] = _encode_card_scalar(
                    card_id=str(getattr(act, "card_id", "")),
                    hp_frac=_safe_hp_frac(act),
                    energies=_safe_energies(act),
                    status=_safe_status(act),
                    is_ex=_safe_is_ex(act),
                )

            # ─── Bench (offset 64, mean-pooled)
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
            bench_feat = _zone_pool(bench_cards)
            state[64 : 64 + CARD_SCALAR_FEATS] = bench_feat

            # ─── Hand (offset 128)
            hand_cards = [
                dict(card_id=str(getattr(c, "card_id", "")),
                     hp_frac=1.0, energies=None, status=None, is_ex=False)
                for c in (ps.hand or [])
            ]
            hand_feat = _zone_pool(hand_cards)
            state[128 : 128 + CARD_SCALAR_FEATS] = hand_feat

            # ─── Opponent active (offset 256)
            if opp.active:
                oact = opp.active[0]
                state[256 : 256 + CARD_SCALAR_FEATS] = _encode_card_scalar(
                    card_id=str(getattr(oact, "card_id", "")),
                    hp_frac=_safe_hp_frac(oact),
                    energies=_safe_energies(oact),
                    status=_safe_status(oact),
                    is_ex=_safe_is_ex(oact),
                )

            # ─── Board scalars (offset 448)
            state[448] = len(ps.prize  or []) / PRIZE_CARDS
            state[449] = len(opp.prize or []) / PRIZE_CARDS
            state[450] = len(ps.hand   or []) / HAND_MAX
            state[451] = len(ps.bench  or []) / MAX_BENCH
            state[452] = len(opp.bench or []) / MAX_BENCH
            state[453] = int(getattr(cur, "turn", 0)) / 100.0  # normalised turn

        except Exception as exc:
            logger.warning("State extraction failed: %s – returning zeros", exc)

        return state

    def _get_legal_actions(self, obs: Any) -> list[Action]:
        """Extract legal actions from the cabt observation."""
        actions: list[Action] = []
        raw_actions = []
        if hasattr(obs, "actions"):
            raw_actions = obs.actions or []
        elif isinstance(obs, dict):
            raw_actions = obs.get("actions") or []

        try:
            for raw in raw_actions:
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
            logger.warning("Action extraction failed: %s", exc)

        if not actions:
            actions.append(Action(ActionType.PASS, AreaType.HAND, AreaType.HAND))
        return actions


    def _action_to_cabt(self, action: Action) -> Any:
        """Convert internal Action back to the raw cabt object (or index fallback)."""
        raw = action.extra.get("raw")
        if raw is not None:
            return raw
        return action.source_idx



# ---------------------------------------------------------------------------
# Helper getters for live env (null-safe attribute access)
# ---------------------------------------------------------------------------

def _safe_hp_frac(pokemon: Any) -> float:
    try:
        return max(0.0, min(1.0, pokemon.hp / max(pokemon.max_hp, 1)))
    except Exception:
        return 1.0


def _safe_energies(pokemon: Any) -> list[int]:
    try:
        return [int(getattr(pokemon.energy, t, 0)) for t in
                ("fire", "water", "grass", "lightning", "fighting", "psychic")]
    except Exception:
        return [0] * NUM_TYPES


def _safe_status(pokemon: Any) -> list[int]:
    try:
        s = pokemon.status or ""
        return [
            int("burn"    in str(s).lower()),
            int("poison"  in str(s).lower()),
            int("confuse" in str(s).lower()),
            int("paraly"  in str(s).lower()),
            int("asleep"  in str(s).lower()),
        ]
    except Exception:
        return [0] * NUM_STATUS


def _safe_is_ex(pokemon: Any) -> bool:
    try:
        name = str(getattr(pokemon, "name", "") or "")
        return " ex" in name.lower() or name.endswith(" EX")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_env(env_type: str = "mock", deck_key: str = "dragapult_ex") -> CabtEnvBase:
    """Create and return the appropriate environment backend."""
    if env_type == "live":
        return LiveCabtEnv(deck_key=deck_key)
    return MockCabtEnv(deck_key=deck_key)
