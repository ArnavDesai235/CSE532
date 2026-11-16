#!/bin/bash
#SBATCH -J wdc-taxonomy
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 8 --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
# Step 2a (GPU): LLM proposes the category taxonomy -> $WDC_DIR/taxonomy.json, then stops.
# A human reviews/edits taxonomy.json before step 2b+3 (pipeline.sbatch).
set -eo pipefail
HERE=${SLURM_SUBMIT_DIR:?submit from data_pipeline/ via submit.sh}
source "$HERE/config.sh"
activate_env
mkdir -p "$WDC_DIR/logs"
start_ollama
require_model "$LLM_MODEL"
cd "$WDC_DIR"
PROPOSE_ONLY=1 python "$HERE/02_categories.py" products.parquet products_cat.parquet
echo "Review $WDC_DIR/taxonomy.json, then: bash submit.sh pipeline"
