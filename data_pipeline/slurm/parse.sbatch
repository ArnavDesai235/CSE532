#!/bin/bash
#SBATCH -J wdc-parse
#SBATCH -p defq
#SBATCH -c 4 --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
# Step 1 (CPU only): raw N-Quads -> products.parquet (sampled, deduped).
# Submit via submit.sh (it sbatch-es from data_pipeline/, which this relies on).
set -eo pipefail
HERE=${SLURM_SUBMIT_DIR:?submit from data_pipeline/ via submit.sh}
source "$HERE/config.sh"
activate_env
cd "$WDC_DIR"
python "$HERE/01_parse.py" raw/ products.parquet
