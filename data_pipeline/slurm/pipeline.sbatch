#!/bin/bash
#SBATCH -J wdc-pipeline
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 8 --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
# Steps 2b + 3 (GPU): classify every product into the frozen taxonomy, then embed.
# Both steps checkpoint; if the job hits the time limit, just resubmit and it resumes.
set -eo pipefail
HERE=${SLURM_SUBMIT_DIR:?submit from data_pipeline/ via submit.sh}
source "$HERE/config.sh"
activate_env
cd "$WDC_DIR"
[ -f taxonomy.json ] || { echo "taxonomy.json missing: run 'bash submit.sh' first"; exit 1; }
mkdir -p logs
start_ollama
require_model "$LLM_MODEL"
require_model "$EMBED_MODEL"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

[ -f products_cat.parquet ] || python "$HERE/02_categories.py" products.parquet products_cat.parquet
python "$HERE/03_embed.py" products_cat.parquet dataset/
echo "DONE: $WDC_DIR/dataset/{vectors.npy,attrs.parquet}"
