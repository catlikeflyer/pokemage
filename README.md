# PokéMage: Pokémon TCG RL Training Pipeline

PokéMage is a standalone, competition-ready PyTorch reinforcement learning pipeline developed for the [Kaggle Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge). It trains a neural network agent to play the Pokémon Trading Card Game against the Matsuo Institute's `cabt` engine using REINFORCE or PPO.

## 🚀 Key Features

* **Dynamic Action-Space Attention Network**: Solves the variable action-space problem (actions changing from 4 to 20+ per turn) using batched matrix multiplication (`torch.bmm`) and action masks.
* **Dual-Layer Memory Leak Mitigation**: Flushes C++ heap leaks from `libcg.so` via regular process recycling orchestrated by `run_batched.sh` combined with Python garbage collection and GPU cache empty cycles.
* **Apple Silicon (MPS) & CUDA Acceleration**: Auto-detects and optimizes tensor operations for Apple Silicon (MPS) and NVIDIA CUDA (including MPS-safe multinomial sampling).
* **Time-Gated Execution**: Protects against game execution time limits in Kaggle via a budget-gated agent that switches to fallback strategies under low-time scenarios.
* **Complete Submission Tooling**: Integrates automated agent self-tests and packages dependencies/weights into a Kaggle-compliant submission zip.

---

## 📂 Repository Structure

* [src/config.py](file:///Users/dhnam/Desktop/Data%20Projects/pokemage/src/config.py) — Central training hyperparameters and configs.
* [src/card_data.py](file:///Users/dhnam/Desktop/Data%20Projects/pokemage/src/card_data.py) — Card database and vocabulary normalizer (processes `EN_Card_Data.csv`).
* [src/env_wrapper.py](file:///Users/dhnam/Desktop/Data%20Projects/pokemage/src/env_wrapper.py) — Environment wrappers for the simulator (`MockCabtEnv` and `LiveCabtEnv`).
* [src/model.py](file:///Users/dhnam/Desktop/Data%20Projects/pokemage/src/model.py) — Policy and Value Network architecture (Attention Q/K based).
* [src/train.py](file:///Users/dhnam/Desktop/Data%20Projects/pokemage/src/train.py) — Training loop supporting PPO and REINFORCE.
* [src/eval.py](file:///Users/dhnam/Desktop/Data%20Projects/pokemage/src/eval.py) — Agent testing, evaluation loop, and time-gated action fallback.
* [src/smoke_test.py](file:///Users/dhnam/Desktop/Data%20Projects/pokemage/src/smoke_test.py) — Test suite to verify environment, PyTorch models, and hardware acceleration.
* [run_batched.sh](file:///Users/dhnam/Desktop/Data%20Projects/pokemage/run_batched.sh) — Process orchestrator to reset C++ heap and train continuously.
* [make_submission.sh](file:///Users/dhnam/Desktop/Data%20Projects/pokemage/make_submission.sh) — Script to construct a clean Kaggle submission zip.

---

## ⚙️ Installation & Setup

1. **Environment Setup**:
   Create a Conda environment with Python 3.13:
   ```bash
   conda create -n pokemage python=3.13
   conda activate pokemage
   ```

2. **Dependencies**:
   Install the required PyTorch and NumPy packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(For training on live simulation environments, install `pip install kaggle-environments`)*

3. **Verify Installation**:
   Ensure that PyTorch, device acceleration, and the neural network architecture are fully functional:
   ```bash
   python src/smoke_test.py
   ```

---

## 🏋️ Training Guide

### Single Process Runs
For quick verification, debug runs, or testing different hyperparameters:
```bash
# Run a quick 5-game REINFORCE training run with the mock environment
python src/train.py --num_games 5 --env mock --algo reinforce

# Run 500 games using PPO
python src/train.py --num_games 500 --algo ppo
```

### Batch Mode (Mitigating C++ Memory Leaks)
Due to memory leaks in the underlying C++ simulator engine (`libcg.so`), long training sessions will consume all system RAM. To prevent crashes, use the shell orchestrator script `run_batched.sh` to periodically restart training from the latest checkpoint:
```bash
# Run batched training (runs 10,000 total games in 500-game batches)
chmod +x run_batched.sh
./run_batched.sh
```

You can configure the batched training behavior via environment variables:
```bash
ALGO=ppo ENV=live TOTAL_GAMES=20000 BATCH_SIZE=500 ./run_batched.sh
```

---

## 📈 Evaluation

Evaluate how your trained agent performs:
```bash
python src/eval.py --checkpoint ./checkpoints/latest.pth --num_episodes 10
```

---

## 📦 Building and Packaging for Kaggle

When ready to submit to the Kaggle Pokémon TCG competition:

1. **Build the Zip Package**:
   Run the submission builder script:
   ```bash
   ./make_submission.sh
   ```
   This will run a self-test on the submission agent, verify its compatibility, bundle Python files and model weights, and write the final file to `pokemage_agent.zip`.

2. **Submit to Kaggle**:
   - Go to the competition page and upload `pokemage_agent.zip`.
   - Ensure you upload your trained `latest.pth` to Kaggle datasets if you decide to upload model weights separately.

---

## 🔬 Architecture Details

For more granular details regarding the neural network attention mechanism, MPS optimization strategies, REINFORCE vs PPO hyperparameter settings, and Kaggle deployment details, refer to [WALKTHROUGH.md](file:///Users/dhnam/Desktop/Data%20Projects/pokemage/WALKTHROUGH.md).
