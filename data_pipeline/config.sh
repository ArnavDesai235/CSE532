# Shared settings for every script and SLURM job in data_pipeline/.
# Override any of these by exporting the variable before running, e.g.
#   WDC_DIR=/scratch/$USER/wdc bash setup.sh
# Do not hardcode usernames: everything resolves from $HOME / $USER.

export WDC_DIR=${WDC_DIR:-$HOME/cse532_wdc}            # data + outputs (needs ~10 GB)
export CONDA_ENV=${CONDA_ENV:-cse532_wdc}               # created by setup.sh from environment.yml
export OLLAMA_BIN=${OLLAMA_BIN:-$HOME/ollama/bin/ollama} # existing install is reused if present
export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/.ollama/models}  # ~5.5 GB for both models

export LLM_MODEL=${LLM_MODEL:-qwen2.5:7b}
export EMBED_MODEL=${EMBED_MODEL:-nomic-embed-text}

# Sampling (00_download.sh, 01_parse.py) - keep identical across runs for reproducibility
export N_CHUNKS=${N_CHUNKS:-10}             # random chunk files out of 1,658 (~150 MB each)
export SEED=${SEED:-532}
export PER_DOMAIN_CAP=${PER_DOMAIN_CAP:-2000}   # max products kept per website
export TARGET_N=${TARGET_N:-500000}            # final dataset size (fewer if not enough data)

# Activate the conda env in batch jobs, where ~/.bashrc conda init may not run.
activate_env() {
  if ! command -v conda >/dev/null 2>&1; then
    [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
    command -v conda >/dev/null 2>&1 || module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
  fi
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
}

# Start a private Ollama server for this job on a job-specific port, so two jobs
# (or two users) on the same node never collide on 11434.
start_ollama() {
  local port=$(( 20000 + ${SLURM_JOB_ID:-$$} % 20000 ))
  export OLLAMA_HOST=127.0.0.1:$port
  export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-8}
  "$OLLAMA_BIN" serve > "$WDC_DIR/logs/ollama_${SLURM_JOB_ID:-local}.log" 2>&1 &
  OLLAMA_PID=$!
  trap 'kill $OLLAMA_PID 2>/dev/null' EXIT
  for _ in $(seq 1 90); do
    curl -s "http://$OLLAMA_HOST" >/dev/null && { echo "ollama up on $OLLAMA_HOST"; return 0; }
    sleep 2
  done
  echo "ERROR: ollama did not start; see $WDC_DIR/logs/ollama_${SLURM_JOB_ID:-local}.log" >&2
  return 1
}

# Fail fast if a model is missing (compute nodes may have no internet; setup.sh pulls them).
require_model() {
  local have; have=$("$OLLAMA_BIN" list | awk 'NR>1{print $1}')
  if grep -qx -e "$1" -e "$1:latest" <<< "$have"; then echo "model $1 present"
  else echo "Model $1 missing; pulling (needs internet)..."; "$OLLAMA_BIN" pull "$1"; fi
}
