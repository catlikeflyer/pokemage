"""
main.py — Kaggle cabt Pokémon TCG AI Battle Challenge submission entry point.

Kaggle requirement:
  The last 'def' in this file must accept (obs, config) and return an action.

This file is intentionally thin — all logic lives in agent.py and its
supporting modules (model.py, config.py, card_data.py, env_wrapper.py).

Checkpoint
----------
Upload checkpoints/latest.pth as a Kaggle dataset named "pokemage-weights"
and add it as an input to your submission notebook. It will appear at:
    /kaggle/input/pokemage-weights/latest.pth
"""
from __future__ import annotations

import os
import sys

# ── Make supporting modules importable regardless of cwd ───────────────────
if "__file__" in globals():
    _HERE = os.path.dirname(os.path.abspath(__file__))
else:
    _HERE = "/kaggle_simulations/agent"
    if not os.path.isdir(_HERE):
        _HERE = os.path.dirname(os.path.abspath(sys.argv[0])) if (globals().get("sys") and sys.argv) else os.getcwd()

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Import the fully self-contained agent (bootstraps model on first call)
from agent import agent as _agent_fn   # noqa: E402


def agent(obs, config):
    """
    Kaggle evaluator calls this once per turn.

    Parameters
    ----------
    obs    : cabt Observation object
    config : cabt environment config (unused by us, kept for API compat)

    Returns
    -------
    Step 0 : list[int]  — 60-card deck declaration
    Step 1+ : raw cabt action object chosen by the trained policy
    """
    return _agent_fn(obs, config)
