#!/usr/bin/env bash
# build_submission.sh
# ───────────────────
# Builds submission.zip ready for upload to the Kaggle cabt competition.
#
# Layout inside the zip:
#   main.py           ← entry point (last def = agent)
#   agent.py
#   config.py
#   card_data.py
#   env_wrapper.py
#   model.py
#   (EN_Card_Data.csv if present — avoids re-downloading at runtime)
#
# Checkpoint is NOT bundled here — upload checkpoints/latest.pth as a
# separate Kaggle dataset named "pokemage-weights".
#
# Usage:
#   bash build_submission.sh                 # from repo root
#   bash build_submission.sh --with-weights  # also bundle latest.pth
#
# Output: submission.zip (≈ a few KB without weights, ~50 MB with)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$REPO_ROOT/submission.zip"
TMP="$REPO_ROOT/.submission_tmp"

# Clean up any previous build
rm -rf "$TMP" "$OUT"
mkdir -p "$TMP"

echo "──────────────────────────────────────────"
echo "  Building Pokémage submission.zip"
echo "  Root: $REPO_ROOT"
echo "──────────────────────────────────────────"

# ── Core files ─────────────────────────────────────────────────────────────
cp "$REPO_ROOT/main.py"            "$TMP/main.py"
cp "$REPO_ROOT/src/agent.py"       "$TMP/agent.py"
cp "$REPO_ROOT/src/config.py"      "$TMP/config.py"
cp "$REPO_ROOT/src/card_data.py"   "$TMP/card_data.py"
cp "$REPO_ROOT/src/env_wrapper.py" "$TMP/env_wrapper.py"
cp "$REPO_ROOT/src/model.py"       "$TMP/model.py"

echo "  ✓ Core source files copied"

# ── Optional: card CSV ──────────────────────────────────────────────────────
if [ -f "$REPO_ROOT/EN_Card_Data.csv" ]; then
    cp "$REPO_ROOT/EN_Card_Data.csv" "$TMP/EN_Card_Data.csv"
    echo "  ✓ EN_Card_Data.csv included"
else
    echo "  ⚠ EN_Card_Data.csv not found — agent will use synthetic vocab at runtime"
fi

# ── Optional: bundle weights ────────────────────────────────────────────────
if [[ "${1-}" == "--with-weights" ]]; then
    CKPT="$REPO_ROOT/checkpoints/latest.pth"
    if [ -f "$CKPT" ]; then
        cp "$CKPT" "$TMP/latest.pth"
        echo "  ✓ latest.pth bundled ($(du -h "$CKPT" | cut -f1))"
        echo "  ℹ  Agent will load from ./latest.pth inside the zip"
    else
        echo "  ⚠ --with-weights: $CKPT not found — skipping"
    fi
fi

# ── Zip ─────────────────────────────────────────────────────────────────────
(cd "$TMP" && zip -qr "$OUT" .)
rm -rf "$TMP"

SIZE=$(du -h "$OUT" | cut -f1)
echo ""
echo "  ✓ submission.zip created ($SIZE)"
echo "  → $OUT"
echo ""
echo "  Next steps:"
echo "    1. Upload checkpoints/latest.pth as Kaggle dataset 'pokemage-weights'"
echo "    2. Upload submission.zip on the competition Submit page"
echo "──────────────────────────────────────────"
