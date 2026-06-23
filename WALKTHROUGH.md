# Pokémon TCG RL Training Pipeline – Walkthrough

## Project at a Glance

A standalone, competition-ready policy-gradient training pipeline for the [Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge) built on PyTorch. Trains a dynamic-action-space attention network via REINFORCE or PPO against the Matsuo Institute's `cabt` engine, with structured workarounds for the `libcg.so` C++ memory leak and the 10-minute Kaggle execution constraint.

**Environment:** `conda activate pokemage` (Python 3.13 · PyTorch 2.12 · NumPy 2.5)  
**Accelerator:** CUDA → **MPS (Apple Silicon)** → CPU, auto-detected at startup

---

## File Map

```
pokemage/
├── src/               Core Python source package
│   ├── config.py      Central Config dataclass — all hyperparameters
│   ├── card_data.py   Card ID → int vocab; loads EN_Card_Data.csv or synthetic mock
│   ├── env_wrapper.py MockCabtEnv + LiveCabtEnv + state/action feature extractors
│   ├── model.py       PolicyNetwork (bmm attention + ValueHead)
│   ├── train.py       REINFORCE + PPO loop, RolloutBuffer, GC flush, get_device()
│   ├── eval.py        TimeGatedAgent — hard 30s time-budget fallback
│   ├── agent.py       Submission agent entrypoint
│   └── smoke_test.py  7-check verification suite (MPS-aware)
├── run_batched.sh     Bash orchestrator: kills + restarts per batch
├── make_submission.sh Bash packager: gathers files into submission/
└── requirements.txt   torch>=2.1.0, numpy>=1.24.0
```

---

## Architecture

### Forward pass — tensor flow

```
numpy obs (STATE_DIM=512,)          numpy actions (N, ACTION_DIM=128)
        │                                       │
        ▼                                       ▼
  StateEncoder                          ActionEncoder
  Linear(512→256) + GELU               Linear(128→256) + GELU
  Linear(256→512) + LayerNorm          Linear(256→512) + LayerNorm
  + residual skip                           │
        │                                   │
        ▼                                   ▼
  state_emb (B, 512)             action_keys (B, N, 512)
        │
        ▼  unsqueeze → (B, 512, 1)
  ┌─────────────────────────────────────┐
  │  torch.bmm(action_keys, query)      │  ← dynamic N per turn
  │  → (B, N, 1) → squeeze → (B, N)    │
  └─────────────────────────────────────┘
        │
        ▼  masked_fill(illegal, −1e9)
   logits (B, N)
        │
       ├──→ softmax  → probs     (B, N)
       ├──→ log_softmax → log_probs (B, N)
       └──→ − Σ p·log p → entropy  (B,)

  state_emb → ValueHead → value (B, 1)
```

### Memory leak mitigation — two layers

```
run_batched.sh  (outer, shell level)
│
│  while games_done < TOTAL_GAMES:
│      python src/train.py --num_games BATCH_SIZE   ← NEW process each batch
│      # process exit → OS reclaims full C++ heap (libcg.so leak flushed)
│
└──► train.py  (inner, Python level)
         │
         │  every flush_every games:
         │      save_checkpoint()
         │      gc.collect()
         │      torch.mps.empty_cache()   # or cuda / no-op
         │      buffer.clear()
```

### Time-budget guard

```
TimeGatedAgent.act()
│
├── time_remaining < 30s ?  ──► random.choice(legal_actions)  [instant]
├── elapsed > 570s ?        ──► random.choice(legal_actions)  [instant]
└── otherwise               ──► policy.forward() → multinomial sample
```

---

## Verified Smoke Test Output (MPS)

```
────────────────────────────────────────────────────────────
  1. Import checks
────────────────────────────────────────────────────────────
  ✓ torch  2.12.1
  ✓ numpy  2.5.0
  ✓ All project modules imported successfully
  ⚡ Accelerator : MPS
  ✓ MPS built   : True

────────────────────────────────────────────────────────────
  2. Card vocabulary
────────────────────────────────────────────────────────────
  ✓ card_id_to_idx('1') = 1  (vocab size: 1267)
  ✓ UNK lookup = 0

────────────────────────────────────────────────────────────
  3. PolicyNetwork tensor shape contract (bmm)  [MPS]
────────────────────────────────────────────────────────────
  ✓ logits     : (4, 12)
  ✓ probs      : (4, 12)  (sum over legal = 1.000000)
  ✓ log_probs  : (4, 12)
  ✓ value      : (4, 1)
  ✓ entropy    : (4,)
  ✓ Masked logits verified (< -1e8 for illegal actions)
  ✓ bmm: (B=4, N=12, D=512) × (B=4, D=512, 1) → (B=4, N=12) ✓

────────────────────────────────────────────────────────────
  4. MockCabtEnv reset / step
────────────────────────────────────────────────────────────
  ✓ reset(): state=(512,), actions=(5, 128), mask=(5,)
  ✓ step(0): reward=0.00, done=False

────────────────────────────────────────────────────────────
  5. 5-episode REINFORCE training run
────────────────────────────────────────────────────────────
  ✓ 5 episodes collected in 0.79s
  ✓ Loss: 0.1788
  ✓ Episode rewards: ['-0.5', '-0.5', '+0.0', '+0.0', '+0.0']
  ✓ Loss is finite and non-NaN

────────────────────────────────────────────────────────────
  6. Checkpoint save / load round-trip
────────────────────────────────────────────────────────────
  ✓ Checkpoint saved to latest.pth
  ✓ Loaded checkpoint at game_idx=5
  ✓ Model weights identical after round-trip

────────────────────────────────────────────────────────────
  7. TimeGatedAgent time-based fallback
────────────────────────────────────────────────────────────
  ✓ Normal turn → model used, action=3
  ✓ Low-time turn → fallback used, action=7
  ✓ Turns: 2 | Model: 1 (50%) | Fallback: 1 (50%) | Elapsed: 0.0s

────────────────────────────────────────────────────────────
  All checks passed ✓
────────────────────────────────────────────────────────────

  ╔══════════════════════════════════════════════════════╗
  ║   Pokémon TCG RL Pipeline – Smoke Test PASSED  ✓    ║
  ╚══════════════════════════════════════════════════════╝
```

---

## Key Design Decisions

### 1 — Dynamic action space via `torch.bmm`

The number of legal actions N varies every turn (4–20+ depending on board state). A fixed softmax head would require padding to a maximum size and masking, wasting memory and compute. Instead, both the state embedding and every action embedding are projected to the same `STATE_DIM`, then scored with a batched dot product:

```python
query  = state_emb.unsqueeze(2)           # (B, STATE_DIM, 1)
logits = torch.bmm(action_keys, query)    # (B, N, 1) → squeeze → (B, N)
logits = logits / sqrt(STATE_DIM)         # temperature scaling
logits = logits.masked_fill(mask, -1e9)   # illegal → −∞ before softmax
```

N expands and contracts without any network surgery. The gradient never flows through masked actions.

### 2 — MPS optimisation

| Concern | Solution |
|---|---|
| Device selection | `get_device()` in `train.py`: `cuda > mps > cpu` |
| Cache flushing | `torch.mps.empty_cache()` in `flush_memory()` |
| `multinomial` on MPS | Probs moved to CPU for 1-D sampling in `policy.act()` — negligible transfer cost, fully MPS forward/backward |
| Accurate timing | `torch.mps.synchronize()` before `perf_counter()` calls |
| Cross-device assertions | `.cpu()` normalisation in weight round-trip checks |

### 3 — REINFORCE vs PPO

| | REINFORCE (`--algo reinforce`) | PPO (`--algo ppo`) |
|---|---|---|
| Update frequency | Every `batch_games` episodes | Every `batch_games` episodes |
| Advantage | Discounted returns − baseline | GAE (λ=0.95) |
| Epochs per rollout | 1 | 4 mini-batch passes |
| Clip | None | ε=0.2 |
| Best for | Fast iteration, low overhead | Stable long-run training |

### 4 — `card_data.py` column normalisation

The real `EN_Card_Data.csv` uses `'Card ID'` (with a space). The loader now normalises any header with `.lower().replace(" ", "_")` before matching, accepting `card_id`, `Card ID`, `id`, `ID`, etc. without requiring any manual preprocessing.

---

## Usage

```bash
conda activate pokemage

# Verify the full pipeline on MPS
python src/smoke_test.py

# 5-game mock run — REINFORCE
python src/train.py --num_games 5 --env mock --algo reinforce

# 500-game mock run — PPO, save to ./checkpoints/
python src/train.py --num_games 500 --algo ppo

# Resume from last checkpoint
python src/train.py --num_games 500 --checkpoint ./checkpoints/latest.pth

# Full batched training (10,000 games, 500 per process — flushes C++ heap)
chmod +x run_batched.sh
./run_batched.sh

# Live cabt engine + PPO (requires kaggle-environments)
ALGO=ppo ENV=live TOTAL_GAMES=20000 BATCH_SIZE=500 ./run_batched.sh

# Evaluate a checkpoint (3 episodes, with timing summary)
python src/eval.py --checkpoint ./checkpoints/latest.pth --num_episodes 3
```

### Key CLI flags for `train.py`

| Flag | Default | Description |
|---|---|---|
| `--num_games` | 500 | Games per process invocation |
| `--algo` | `reinforce` | `reinforce` or `ppo` |
| `--env` | `mock` | `mock` (no SDK) or `live` (cabt) |
| `--deck` | `dragapult_ex` | Starter deck key |
| `--flush_every` | 50 | GC + checkpoint interval (games) |
| `--batch_games` | 10 | Episodes per gradient update |
| `--checkpoint` | _(none)_ | `.pth` path to resume from |
| `--lr` | 3e-4 | Adam learning rate |

### Key env vars for `run_batched.sh`

| Variable | Default | Description |
|---|---|---|
| `TOTAL_GAMES` | 10000 | Total games across all batches |
| `BATCH_SIZE` | 500 | Games per subprocess |
| `ALGO` | `reinforce` | Algorithm |
| `ENV` | `mock` | Environment backend |
| `DECK` | `dragapult_ex` | Deck selection |
| `OUTDIR` | `./checkpoints` | Checkpoint directory |

---

## Kaggle Deployment Checklist

1. **Real card data** — `EN_Card_Data.csv` auto-detected in the working directory; `card_data.py` normalises the `'Card ID'` header automatically.

2. **Switch to live cabt**:
   ```bash
   pip install kaggle-environments
   python src/train.py --env live --algo ppo --num_games 1000
   ```

3. **Update starter deck IDs** in `STARTER_DECKS` in [`env_wrapper.py`](file:///Users/dhnam/Desktop/Data%20Projects/pokemage/src/env_wrapper.py) with real 60-card competition lists.

4. **Wire the submission agent** in your `agent.py`:
   ```python
   from eval import TimeGatedAgent
   _agent = TimeGatedAgent.from_checkpoint("./checkpoints/latest.pth")
   def agent_fn(obs):
       return _agent.act_from_obs(obs)
   ```

5. **Self-play** — swap the `"random"` opponent string in `LiveCabtEnv.reset()` with a path to a previous checkpoint for competitive self-training.

---

## Change Log

| Round | Changes |
|---|---|
| **v1 – Initial build** | All 8 files created. 7-check smoke test passed (CPU, PyTorch 2.12). |
| **v2 – MPS optimisation** | `get_device()` (cuda > mps > cpu), `torch.mps.empty_cache()`, MPS-safe `multinomial` sampling (CPU round-trip), `torch.mps.synchronize()` for timing, device-agnostic checkpoint comparison. `card_data.py` column-name normalisation (`'Card ID'` support). All 7 checks re-verified on MPS (Apple Silicon). |
