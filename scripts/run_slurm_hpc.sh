#!/bin/bash
# ==============================================================================
# SLURM Job Submission Script for Distributed Hilbert B-Tree Vector Engine
# Course: CSE532 Database Systems, Stony Brook University
#
# Hybrid Parallelism Model:
#   - MPI across nodes/ranks: Range-partitioned distributed Hilbert curve
#   - OpenMP/Threads within each rank: Multi-core HNSW and SIMD distance scans
# ==============================================================================

#SBATCH --job-name=cse532_mpi_btree
#SBATCH --output=logs/mpi_%j.out
#SBATCH --error=logs/mpi_%j.err
#SBATCH --nodes=2                   # Number of cluster compute nodes (2 nodes = 80 cores; 4 nodes = 160 cores)
#SBATCH --ntasks-per-node=10        # 10 MPI ranks per node (10 * 4 = 40 physical cores per node)
#SBATCH --cpus-per-task=4           # 4 CPU hardware cores per MPI rank for OpenMP/SIMD
#SBATCH --time=01:00:00             # Wall clock execution limit (1 hour; max 4h on short-40core)
#SBATCH --partition=short-40core    # 40-core partition queue (e.g. Stony Brook SeaWulf)
#SBATCH --mem=120G                  # RAM allocation per node (leaves headroom out of 192GB total)

# Create log directory
mkdir -p logs

echo "=============================================================================="
echo "Starting Slurm Job: $SLURM_JOB_NAME (Job ID: $SLURM_JOB_ID)"
echo "Cluster Nodes Allocated: $SLURM_JOB_NODELIST"
echo "Total MPI Tasks (Ranks): $SLURM_NTASKS"
echo "CPUs per MPI Rank: $SLURM_CPUS_PER_TASK (Total Cores: $(( SLURM_NTASKS * SLURM_CPUS_PER_TASK )))"
echo "Submission Timestamp: $(date)"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 1. Environment & Module Loading (e.g. Stony Brook SeaWulf or generic cluster)
# ------------------------------------------------------------------------------
# Uncomment and adjust module names if your cluster uses environment modules:
# module purge
# module load openmpi/4.1.4
# module load anaconda/3

# ------------------------------------------------------------------------------
# 2. Virtual Environment / Conda Activation
# ------------------------------------------------------------------------------
# Initialize Conda in bash subshell if needed
if ! command -v conda &> /dev/null; then
    for conda_path in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniconda/etc/profile.d/conda.sh" \
        "/shared/miniconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -f "$conda_path" ]; then
            source "$conda_path"
            break
        fi
    done
fi

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
fi

# Activate environment if specified or if CONDA_DEFAULT_ENV is present
# (User assumes Python and Faiss are already installed in this environment)
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using currently active Conda environment: $CONDA_DEFAULT_ENV"
elif [ -n "$1" ]; then
    echo "Activating requested environment: $1"
    conda activate "$1" 2>/dev/null || true
elif [ -d ".venv" ]; then
    echo "Activating local .venv directory..."
    source .venv/bin/activate 2>/dev/null || true
fi

PYTHON_BIN=$(which python)
echo "Resolved Python Binary: $PYTHON_BIN"
$PYTHON_BIN -c "import sys, numpy; print('Python:', sys.version.split()[0], '| NumPy:', numpy.__version__)"
$PYTHON_BIN -c "import faiss; print('Faiss: Successfully loaded from', faiss.__file__)" || {
    echo "WARNING: Could not import faiss. Please activate the Conda environment where Faiss is installed."
}

# ------------------------------------------------------------------------------
# 3. Thread Affinity & OpenMP Configuration
# ------------------------------------------------------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------------------
# 4. Execute Distributed Benchmark via srun
# ------------------------------------------------------------------------------
# Parameters:
#   --n : Number of vectors (e.g. 1,000,000 or 10,000,000)
#   --m : Number of scalar attributes (e.g. 12 attributes)
#   --max-leaf-size : Leaf subgraph capacity (e.g. 20,000)
#   --nq : Number of test queries
srun --export=ALL "$PYTHON_BIN" prototype/bench_mpi.py \
    --n 1000000 \
    --m 12 \
    --max-leaf-size 20000 \
    --nq 50

EXIT_CODE=$?
echo "=============================================================================="
echo "Job Finished with Exit Code: $EXIT_CODE at $(date)"
echo "=============================================================================="
exit $EXIT_CODE
