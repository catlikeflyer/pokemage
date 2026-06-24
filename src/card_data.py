"""
card_data.py
------------
Card-ID → integer index vocabulary.

If EN_Card_Data.csv exists in the working directory it is loaded and the
real card IDs are used.  Otherwise a synthetic 100-entry mock vocabulary
is generated so the full pipeline can be smoke-tested without any
competition assets.

Public API
----------
card_id_to_idx(card_id: str | int) -> int
    O(1) lookup; returns 0 (UNK token) for unseen IDs.

VOCAB_SIZE: int
    Total vocabulary size (padded to config.CARD_VOCAB_SIZE).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Union

from config import CARD_VOCAB_SIZE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal state (module-level singleton)
# ---------------------------------------------------------------------------
_card_to_idx: dict[str, int] = {}
_vocab_size:  int = CARD_VOCAB_SIZE

_UNK_IDX  = 0   # index 0 reserved for unknown / padding cards
_FIRST_IDX = 1  # real cards start at 1


def _build_from_csv(csv_path: Path) -> dict[str, int]:
    """Parse EN_Card_Data.csv and build {card_id: idx} mapping."""
    mapping: dict[str, int] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        # Accept either 'card_id', 'id', or 'ID' column name
        id_col = next(
            (c for c in (reader.fieldnames or [])
             if c.lower().replace(" ", "_") in {"card_id", "id"}),
            None,
        )
        if id_col is None:
            raise ValueError(
                f"Cannot find a card-ID column in {csv_path}. "
                f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            cid = str(row[id_col]).strip()
            if cid and cid not in mapping:
                mapping[cid] = len(mapping) + _FIRST_IDX
    return mapping


def _build_mock(n: int = 100) -> dict[str, int]:
    """Synthesise a tiny vocabulary for smoke-testing."""
    return {f"MOCK_{i:04d}": i + _FIRST_IDX for i in range(n)}


def _ensure_loaded(csv_path: str = "./EN_Card_Data.csv") -> None:
    """Lazy-load the vocabulary once; subsequent calls are no-ops."""
    global _card_to_idx, _vocab_size
    if _card_to_idx:
        return  # already loaded

    p = Path(csv_path)
    if p.exists():
        _card_to_idx = _build_from_csv(p)
        logger.info("Loaded %d cards from %s", len(_card_to_idx), p)
    else:
        _card_to_idx = _build_mock()
        logger.warning(
            "EN_Card_Data.csv not found at '%s'; using synthetic %d-card vocabulary.",
            csv_path,
            len(_card_to_idx),
        )

    # Validate we don't overflow the embedding table
    max_idx = max(_card_to_idx.values(), default=0)
    if max_idx >= CARD_VOCAB_SIZE:
        raise ValueError(
            f"Card vocabulary contains index {max_idx} but CARD_VOCAB_SIZE={CARD_VOCAB_SIZE}. "
            "Increase CARD_VOCAB_SIZE in config.py."
        )
    _vocab_size = CARD_VOCAB_SIZE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init(csv_path: str = "./EN_Card_Data.csv") -> None:
    """Explicitly initialise (or re-initialise) the vocabulary.

    Call this once at startup; otherwise the first call to card_id_to_idx()
    triggers lazy loading automatically.
    """
    global _card_to_idx
    _card_to_idx = {}          # force reload
    _ensure_loaded(csv_path)


def card_id_to_idx(card_id: Union[str, int]) -> int:
    """Convert a card ID to its embedding-table integer index.

    Parameters
    ----------
    card_id:
        The card's string ID (e.g. "sv5-001") or integer representation.

    Returns
    -------
    int
        Integer index in [1, VOCAB_SIZE-1], or 0 (UNK) if not found.
    """
    _ensure_loaded()
    return _card_to_idx.get(str(card_id).strip(), _UNK_IDX)


@property
def VOCAB_SIZE() -> int:  # noqa: N802  (uppercase for constant-like access)
    _ensure_loaded()
    return _vocab_size


# Allow `from card_data import VOCAB_SIZE` as a plain int at import time.
# The actual value will be correct after _ensure_loaded() has run.
VOCAB_SIZE: int = CARD_VOCAB_SIZE

# ---------------------------------------------------------------------------
# Deck-building helpers  (for live cabt submission — needs integer card IDs)
# ---------------------------------------------------------------------------

def build_valid_deck(csv_path: str = "./EN_Card_Data.csv", size: int = 60) -> list[int]:
    """Build a minimal legal 60-card deck from EN_Card_Data.csv.

    A legal Pokémon TCG deck must contain at least one Basic Pokémon.
    Without one the cabt engine immediately ends the game as a forfeit.

    Strategy
    --------
    1.  Find all unique Basic Pokémon IDs  (Stage column = 'Basic Pokémon').
    2.  Find all Basic Energy IDs          (Stage column = 'Basic Energy').
    3.  Fill the deck with 4× of the first available Basic Pokémon,
        then pad to 60 with Basic Energy (repeating as needed).

    Returns
    -------
    list[int]
        60 integer card IDs suitable for trainer.step() on step 0.
    """
    stage_col = "Stage (Pokémon)/Type (Energy and Trainer)"
    id_col    = "Card ID"

    # Search for the CSV in several locations (handles Kaggle src/ layouts)
    import os
    search_paths = [
        csv_path,
        "./EN_Card_Data.csv",
        "../EN_Card_Data.csv",
        os.path.join(os.path.dirname(__file__), "..", "EN_Card_Data.csv"),
        "/kaggle/working/EN_Card_Data.csv",
        "/kaggle/working/src/EN_Card_Data.csv",
        "/kaggle/input/pokemage-src/EN_Card_Data.csv",
    ]
    found_csv = next((p for p in search_paths if os.path.isfile(p)), None)

    try:
        if found_csv is None:
            raise FileNotFoundError(f"EN_Card_Data.csv not found; searched: {search_paths}")
        import pandas as pd  # type: ignore
        df = pd.read_csv(found_csv)

        basic_poke_ids = (
            df[df[stage_col] == "Basic Pokémon"][id_col]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        basic_energy_ids = (
            df[df[stage_col] == "Basic Energy"][id_col]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    except Exception as exc:
        logger.warning("build_valid_deck: CSV parse failed (%s) — using hardcoded IDs", exc)
        # Hardcoded fallback from EN_Card_Data.csv inspection:
        # Basic Energy: 1-8, first Basic Pokémon: 22
        basic_poke_ids  = [22, 24, 25, 27, 28]
        basic_energy_ids = [1, 2, 3, 4, 5, 6, 7, 8]

    if not basic_poke_ids:
        logger.error("No Basic Pokémon found in CSV — cabt will forfeit immediately!")
        basic_poke_ids = [22]
    if not basic_energy_ids:
        basic_energy_ids = [1]

    # 4× copies of the first Basic Pokémon (max 4 per non-energy card in TCG rules)
    n_poke = min(4, size)
    deck   = [basic_poke_ids[0]] * n_poke

    # Pad to 60 with Basic Energy (cycling through available types)
    energy_pool = basic_energy_ids * ((size // len(basic_energy_ids)) + 2)
    deck += energy_pool[: size - len(deck)]

    assert len(deck) == size, f"build_valid_deck produced {len(deck)} cards, expected {size}"
    logger.info(
        "Built deck: 4× Pokémon ID %d + %d× Energy (IDs %s)",
        basic_poke_ids[0], size - n_poke, basic_energy_ids[:4],
    )
    return deck


def all_card_ids(csv_path: str = "./EN_Card_Data.csv") -> list[int]:
    """Return all integer card IDs from EN_Card_Data.csv in order."""
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(csv_path)
        id_col = next(
            (c for c in df.columns if c.lower().replace(" ", "_") in {"card_id", "id"}),
            None,
        )
        if id_col:
            return df[id_col].dropna().astype(int).unique().tolist()
    except Exception as exc:
        logger.warning("all_card_ids failed: %s", exc)
    return list(range(1, 101))
