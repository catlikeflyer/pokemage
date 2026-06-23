#!/usr/bin/env bash
# =============================================================================
# run_batched.sh
# =============================================================================
# Orchestrates the Pokémon TCG RL training pipeline by running train.py in
# repeated short-lived processes.  Each invocation terminates completely after
# --num_games games, which fully flushes the C++ heap and eliminates the
# cumulative memory leak in libcg.so.
#
# Usage
# -----
#   chmod +x run_batched.sh
#   ./run_batched.sh                          # default: 10,000 games, 500/batch
#   TOTAL_GAMES=5000 BATCH_SIZE=250 ./run_batched.sh
#   ALGO=ppo ENV=live DECK=mega_lucario_ex ./run_batched.sh
#
# Environment variables (all optional, with defaults shown below)
# ---------------------------------------------------------------
#   TOTAL_GAMES   Total games to train                    (default: 10000)
#   BATCH_SIZE    Games per process invocation            (default: 500)
#   OUTDIR        Checkpoint output directory             (default: ./checkpoints)
#   ALGO          reinforce | ppo                         (default: reinforce)
#   ENV           mock | live                             (default: mock)
#   DECK          Starter deck key                        (default: dragapult_ex)
#   PYTHON        Python interpreter                      (default: python)
#   FLUSH_EVERY   GC flush interval (within a batch)      (default: 50)
#   SLEEP_BETWEEN Seconds to sleep between batches        (default: 2)
#   LOG_DIR       Directory for per-batch log files       (default: ./logs)
# =============================================================================
set -euo pipefail

# ── Configuration (override via env vars) ────────────────────────────────────
TOTAL_GAMES="${TOTAL_GAMES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-500}"
OUTDIR="${OUTDIR:-./checkpoints}"
ALGO="${ALGO:-reinforce}"
ENV="${ENV:-mock}"
DECK="${DECK:-dragapult_ex}"
PYTHON="${PYTHON:-python3}"
FLUSH_EVERY="${FLUSH_EVERY:-50}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-2}"
LOG_DIR="${LOG_DIR:-./logs}"

# ── Derived ──────────────────────────────────────────────────────────────────
NUM_BATCHES=$(( (TOTAL_GAMES + BATCH_SIZE - 1) / BATCH_SIZE ))
CHECKPOINT_PATH="${OUTDIR}/latest.pth"

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }

banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    printf "║  %-56s  ║\n" "$*"
    echo "╚══════════════════════════════════════════════════════════╝"
}

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUTDIR" "$LOG_DIR"

banner "Pokémon TCG RL – Batched Training Orchestrator"
log "Total games  : $TOTAL_GAMES"
log "Batch size   : $BATCH_SIZE"
log "Num batches  : $NUM_BATCHES"
log "Algorithm    : $ALGO"
log "Environment  : $ENV"
log "Deck         : $DECK"
log "Checkpoint   : $CHECKPOINT_PATH"
log "Output dir   : $OUTDIR"
log "Log dir      : $LOG_DIR"
echo ""

# ── Track progress with a state file ─────────────────────────────────────────
STATE_FILE="${OUTDIR}/.train_state"
games_done=0

if [[ -f "$STATE_FILE" ]]; then
    games_done=$(cat "$STATE_FILE")
    log "Resuming from state file: games_done=$games_done"
fi

# ── Main loop ─────────────────────────────────────────────────────────────────
batch_num=0

while [[ $games_done -lt $TOTAL_GAMES ]]; do
    batch_num=$(( batch_num + 1 ))
    remaining=$(( TOTAL_GAMES - games_done ))
    this_batch=$(( remaining < BATCH_SIZE ? remaining : BATCH_SIZE ))
    log_file="${LOG_DIR}/batch_$(printf '%04d' $batch_num).log"

    banner "Batch $batch_num / $NUM_BATCHES  ($this_batch games)"

    # Build resume argument
    resume_arg=""
    if [[ -f "$CHECKPOINT_PATH" ]]; then
        resume_arg="--checkpoint ${CHECKPOINT_PATH}"
        log "Resuming from checkpoint: $CHECKPOINT_PATH"
    else
        log "No checkpoint found – starting fresh."
    fi

    # ── Launch training subprocess ─────────────────────────────────────────
    # The subprocess exits after `this_batch` games, fully releasing the
    # C++ heap and killing any lingering libcg.so state.
    set +e
    "$PYTHON" src/train.py \
        --num_games    "$this_batch"    \
        --algo         "$ALGO"          \
        --env          "$ENV"           \
        --deck         "$DECK"          \
        --outdir       "$OUTDIR"        \
        --flush_every  "$FLUSH_EVERY"   \
        $resume_arg                     \
        2>&1 | tee "$log_file"
    exit_code=$?
    set -e

    # ── Handle subprocess result ───────────────────────────────────────────
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: src/train.py exited with code $exit_code (see $log_file)"
        log "Waiting 10s before retrying this batch..."
        sleep 10
        # Do NOT advance games_done – retry the same batch
        continue
    fi

    games_done=$(( games_done + this_batch ))

    # Update state file so we can resume from the shell-level too
    echo "$games_done" > "$STATE_FILE"

    log "✓ Batch $batch_num complete. Total games: $games_done / $TOTAL_GAMES"

    # Optional pause between batches (lets OS reclaim memory completely)
    if [[ $games_done -lt $TOTAL_GAMES && $SLEEP_BETWEEN -gt 0 ]]; then
        log "Sleeping ${SLEEP_BETWEEN}s between batches..."
        sleep "$SLEEP_BETWEEN"
    fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
banner "Training Complete"
log "Total games trained : $games_done"
log "Final checkpoint    : $CHECKPOINT_PATH"
log "Per-batch logs      : $LOG_DIR/"
echo ""

# Run a quick eval smoke-test on the final checkpoint
if [[ -f "$CHECKPOINT_PATH" ]]; then
    log "Running 3-episode evaluation smoke-test..."
    "$PYTHON" src/eval.py --checkpoint "$CHECKPOINT_PATH" --num_episodes 3 \
        2>&1 | tee "${LOG_DIR}/final_eval.log"
fi

log "Done. 🎉"
