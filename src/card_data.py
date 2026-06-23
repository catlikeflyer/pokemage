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
