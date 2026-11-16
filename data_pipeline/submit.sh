#!/bin/bash
# Submits the SLURM jobs from the right directory.
#   bash submit.sh            -> parse (CPU) then taxonomy proposal (GPU), chained
#   bash submit.sh pipeline   -> classification + embedding (GPU), after you reviewed taxonomy.json
set -eo pipefail
cd "$(dirname "$0")"
source config.sh
mkdir -p logs

case "${1:-first}" in
  first)
    p=$(sbatch --parsable slurm/parse.sbatch)
    t=$(sbatch --parsable --dependency=afterok:$p slurm/propose.sbatch)
    echo "submitted parse=$p, taxonomy=$t (runs after parse succeeds)"
    echo "watch: squeue -u $USER ; tail -f logs/wdc-*_{$p,$t}.out"
    echo "when done: review $WDC_DIR/taxonomy.json, then: bash submit.sh pipeline" ;;
  pipeline)
    [ -f "$WDC_DIR/taxonomy.json" ] || { echo "no taxonomy.json yet"; exit 1; }
    j=$(sbatch --parsable slurm/pipeline.sbatch)
    echo "submitted pipeline=$j ; tail -f logs/wdc-pipeline_$j.out" ;;
  *) echo "usage: bash submit.sh [pipeline]"; exit 1 ;;
esac
