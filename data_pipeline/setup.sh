#!/bin/bash
# One-time setup. Run on the LOGIN node (needs internet):   bash setup.sh
# Idempotent: safe to re-run; finished steps are skipped.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
mkdir -p "$WDC_DIR/logs"

echo "== 1/4 conda env '$CONDA_ENV'"
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
command -v conda >/dev/null 2>&1 || module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
command -v conda >/dev/null 2>&1 || { echo "conda not found; load your cluster's conda module first"; exit 1; }
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda env update -n "$CONDA_ENV" -f "$HERE/environment.yml" --prune
else
  conda env create -n "$CONDA_ENV" -f "$HERE/environment.yml"
fi
conda activate "$CONDA_ENV"
python -c "import pandas, pyarrow, numpy, ollama; print('python deps ok')"

echo "== 2/4 ollama binary"
if [ -x "$OLLAMA_BIN" ]; then
  echo "found $OLLAMA_BIN ($("$OLLAMA_BIN" --version 2>/dev/null | tail -1))"
else
  dest=$(dirname "$(dirname "$OLLAMA_BIN")")   # ~/ollama
  mkdir -p "$dest"
  if curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz -o /tmp/ollama_$USER.tgz; then
    tar -xzf /tmp/ollama_$USER.tgz -C "$dest"
  else  # newer releases ship .tar.zst
    curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst -o /tmp/ollama_$USER.tar.zst
    tar --use-compress-program=unzstd -xf /tmp/ollama_$USER.tar.zst -C "$dest"
  fi
  rm -f /tmp/ollama_$USER.tgz /tmp/ollama_$USER.tar.zst
  "$OLLAMA_BIN" --version
fi

echo "== 3/4 models ($LLM_MODEL, $EMBED_MODEL) -> $OLLAMA_MODELS"
mkdir -p "$OLLAMA_MODELS"
start_ollama          # temporary CPU-only server on the login node, just to download weights
require_model "$LLM_MODEL"
require_model "$EMBED_MODEL"
"$OLLAMA_BIN" list
kill $OLLAMA_PID 2>/dev/null || true

echo "== 4/4 download WDC data (random chunks, seed $SEED)"
bash "$HERE/00_download.sh"

echo
echo "Setup done. Next:  bash submit.sh"
