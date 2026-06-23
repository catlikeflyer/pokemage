"""
config.py
---------
Central hyperparameter registry for the Pokémon TCG RL pipeline.

All training knobs live here so that train.py's argparse can override them
without any changes to model or env code.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Vocabulary / card pool
# ---------------------------------------------------------------------------
CARD_VOCAB_SIZE: int = 2048   # ≥ 2,000 card pool; pad to next power-of-2
CARD_EMBED_DIM: int = 64      # per-card embedding size

# ---------------------------------------------------------------------------
# Network dimensions
# ---------------------------------------------------------------------------
STATE_DIM:   int = 512   # output dimension of StateEncoder
ACTION_DIM:  int = 128   # output dimension of ActionEncoder
HIDDEN_DIM:  int = 256   # intermediate MLP hidden size
NUM_HEADS:   int = 4     # unused by default dot-product head; reserved for MHA

# ---------------------------------------------------------------------------
# Board geometry constants
# ---------------------------------------------------------------------------
MAX_BENCH:   int = 5
PRIZE_CARDS: int = 6
HAND_MAX:    int = 10
DECK_MAX:    int = 60
NUM_ZONES:   int = 8     # HAND, BENCH, ACTIVE, DISCARD, PRIZE, DECK, STADIUM, LOOKING
NUM_TYPES:   int = 6     # Fire, Water, Grass, Lightning, Fighting, Psychic (simplified)
NUM_STATUS:  int = 5     # Burned, Poisoned, Confused, Paralyzed, Asleep

# Feature dimensions derived from the board constants above
CARD_SCALAR_FEATS: int = 1 + 1 + NUM_TYPES + NUM_STATUS  # hp_frac + is_ex + energies + status = 14

# Action encoding constants
NUM_ACTION_TYPES: int = 10   # attack, play_basic, evolve, retreat, use_trainer,
                              # attach_energy, use_ability, pass, take_prize, end_turn

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # ---- network ---
    card_vocab_size:  int   = CARD_VOCAB_SIZE
    card_embed_dim:   int   = CARD_EMBED_DIM
    state_dim:        int   = STATE_DIM
    action_dim:       int   = ACTION_DIM
    hidden_dim:       int   = HIDDEN_DIM

    # ---- RL ---
    algo:             str   = "reinforce"   # "reinforce" | "ppo"
    gamma:            float = 0.99
    lam:              float = 0.95          # GAE lambda (PPO only)
    lr:               float = 3e-4
    max_grad_norm:    float = 0.5
    entropy_coeff:    float = 0.01
    value_coeff:      float = 0.5           # PPO value loss weight
    ppo_clip:         float = 0.2
    ppo_epochs:       int   = 4
    ppo_minibatch:    int   = 64
    batch_games:      int   = 10            # episodes per gradient update

    # ---- execution ---
    num_games:        int   = 500           # games per process invocation
    flush_every:      int   = 50            # GC + checkpoint every N games
    env:              str   = "mock"        # "mock" | "live"
    deck:             str   = "dragapult_ex"

    # ---- paths ---
    outdir:           str   = "./checkpoints"
    checkpoint:       str   = ""           # resume path (empty = fresh start)
    card_csv:         str   = "./EN_Card_Data.csv"

    def __post_init__(self) -> None:
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
