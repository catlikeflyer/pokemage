#!/usr/bin/env bash
# make_submission.sh
# ------------------
# Assembles the Kaggle submission zip from the project source files.
#
# Usage
# -----
#   ./make_submission.sh                       # uses checkpoints/latest.pth
#   ./make_submission.sh --ckpt my.pth         # use a specific checkpoint
#   ./make_submission.sh --no-ckpt             # skip checkpoint (upload separately)
#
# Output
# ------
#   submission/          clean directory ready to inspect / test
#   pokemage_agent.zip   zip to upload to Kaggle as your submission
#
# Kaggle upload steps
# -------------------
#   1. Upload latest.pth to Kaggle > Datasets > "+ New Dataset"
#      Name it exactly:  pokemage-weights
#      After upload, note the dataset path shown (e.g. youruser/pokemage-weights)
#
#   2. Go to the competition page > "Submit Prediction"
#      Upload pokemage_agent.zip
#      In the notebook, attach your pokemage-weights dataset as an input
#
#   3. Kaggle will call agent(obs, config) for every turn.
#      The checkpoint will be at /kaggle/input/pokemage-weights/latest.pth

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ────────────────────────────────────────────────────────────────
OUTDIR="$SCRIPT_DIR/submission"
ZIPFILE="$SCRIPT_DIR/pokemage_agent.zip"
CKPT="$SCRIPT_DIR/checkpoints/latest.pth"
INCLUDE_CKPT=true

# ── Parse flags ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)     CKPT="$2";        shift 2 ;;
        --no-ckpt)  INCLUDE_CKPT=false; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Source files to copy ─────────────────────────────────────────────────────
SOURCE_FILES=(
    agent.py
    config.py
    card_data.py
    env_wrapper.py
    model.py
    train.py
    eval.py
)

# ── Clean output dir ─────────────────────────────────────────────────────────
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Pokémon TCG RL – Build Submission"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Copy Python source files ─────────────────────────────────────────────────
echo ""
echo "  Copying source files…"
for f in "${SOURCE_FILES[@]}"; do
    src="$SCRIPT_DIR/src/$f"
    if [[ -f "$src" ]]; then
        cp "$src" "$OUTDIR/$f"
        size=$(wc -c < "$src" | tr -d ' ')
        printf "  ✓  %-22s  %d bytes\n" "$f" "$size"
    else
        echo "  ✗  $f  NOT FOUND – aborting"
        exit 1
    fi
done

# Copy main.py from root
if [[ -f "$SCRIPT_DIR/main.py" ]]; then
    cp "$SCRIPT_DIR/main.py" "$OUTDIR/main.py"
    main_size=$(wc -c < "$SCRIPT_DIR/main.py" | tr -d ' ')
    printf "  ✓  %-22s  %d bytes\n" "main.py" "$main_size"
else
    echo "  ✗  main.py NOT FOUND – aborting"
    exit 1
fi


# ── Copy card data CSV ───────────────────────────────────────────────────────
if [[ -f "$SCRIPT_DIR/EN_Card_Data.csv" ]]; then
    cp "$SCRIPT_DIR/EN_Card_Data.csv" "$OUTDIR/EN_Card_Data.csv"
    csv_size=$(wc -c < "$SCRIPT_DIR/EN_Card_Data.csv" | tr -d ' ')
    printf "  ✓  %-22s  %d bytes\n" "EN_Card_Data.csv" "$csv_size"
else
    echo "  ⚠  EN_Card_Data.csv not found – agent will use synthetic vocab"
fi

# ── Copy checkpoint ───────────────────────────────────────────────────────────
echo ""
if $INCLUDE_CKPT; then
    if [[ -f "$CKPT" ]]; then
        cp "$CKPT" "$OUTDIR/latest.pth"
        ckpt_size=$(wc -c < "$CKPT" | tr -d ' ')
        printf "  ✓  %-22s  %d bytes\n" "latest.pth (bundled)" "$ckpt_size"
        echo ""
        echo "  NOTE: Checkpoint bundled inside zip."
        echo "        agent.py will load it from the submission directory."
        echo "        You do NOT need to upload a separate Kaggle dataset."
    else
        echo "  ⚠  Checkpoint not found at: $CKPT"
        echo "  ⚠  Submission will use random policy weights."
        echo "  ⚠  To fix: train first with:"
        echo "       conda run -n pokemage python train.py --num_games 500 --algo ppo"
        echo "     then re-run this script."
    fi
else
    echo "  ⏭  Checkpoint skipped (--no-ckpt)."
    echo "     Upload latest.pth as a Kaggle dataset named 'pokemage-weights'."
fi

# ── Self-test the agent in the submission directory ──────────────────────────
echo ""
echo "  Running self-test…"
if conda run -n pokemage python "$OUTDIR/agent.py" --selftest 2>&1 | \
        grep -E "(✓|PASSED|FAILED|Error|error)"; then
    echo "  ✓  Self-test passed"
else
    echo "  ⚠  Self-test produced warnings (see above) – submission may still work"
fi

# ── Create zip ───────────────────────────────────────────────────────────────
echo ""
echo "  Creating zip…"
rm -f "$ZIPFILE"
(cd "$OUTDIR" && zip -r "$ZIPFILE" . -x "*.pyc" -x "__pycache__/*") > /dev/null
zip_size=$(wc -c < "$ZIPFILE" | tr -d ' ')
printf "  ✓  %-22s  %d bytes\n" "pokemage_agent.zip" "$zip_size"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Submission ready!"
echo ""
echo "  Directory : $OUTDIR"
echo "  Zip file  : $ZIPFILE"
echo ""
echo "  Upload pokemage_agent.zip to:"
echo "    https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge/submissions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
