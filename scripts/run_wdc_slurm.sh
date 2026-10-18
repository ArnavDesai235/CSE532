#!/bin/bash
# ==============================================================================
# SLURM Job Script: Resumable WDC Embedding Ingestion on SeaWulf A100 Nodes
# Course: CSE532 Database Systems, Stony Brook University
#
# Hardware Target:
#   - Stony Brook SeaWulf 'a100-long' queue (up to 48 hours, 1 node with 4x A100 80GB)
#   - or 'a100' queue (up to 8 hours, 1-2 nodes)
# ==============================================================================

#SBATCH --job-name=wdc_embed_a100
#SBATCH --output=logs/wdc_%j.out
#SBATCH --error=logs/wdc_%j.err
#SBATCH --partition=a100-long        # Use 'a100-long' (up to 48h) or 'a100' (up to 8h)
#SBATCH --nodes=1                   # 1 compute node
#SBATCH --ntasks=1                  # 1 task
#SBATCH --cpus-per-task=32          # 32 CPU cores on Ice Lake host
#SBATCH --gres=gpu:4                # Request all 4x NVIDIA A100 (80GB) GPUs on the node
#SBATCH --time=48:00:00             # Max walltime for a100-long (set 08:00:00 if using a100)
#SBATCH --mem=200G                  # Host RAM allocation (node has 256GB total)

# Create log directory
mkdir -p logs data/wdc_raw data/wdc_processed

echo "=============================================================================="
echo "Starting SeaWulf WDC Ingestion Job: $SLURM_JOB_NAME (Job ID: $SLURM_JOB_ID)"
echo "Node Allocated: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs Assigned: $CUDA_VISIBLE_DEVICES"
echo "CPUs Allocated: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 1. Environment & CUDA Modules
# ------------------------------------------------------------------------------
# SeaWulf A100 nodes use Milan/Ice Lake modules:
module purge
module load slurm
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true

# ------------------------------------------------------------------------------
# 2. Activate Conda Environment
# ------------------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    for conda_path in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/shared/miniconda3/etc/profile.d/conda.sh"; do
        if [ -f "$conda_path" ]; then
            source "$conda_path"
            break
        fi
    done
fi

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
fi

# Activate your project conda environment
# (Replace 'cse532-a100' with your conda environment name)
if conda info --envs | grep -q "cse532-a100"; then
    conda activate cse532-a100
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

PYTHON_BIN=$(which python)
echo "Resolved Python Binary: $PYTHON_BIN"
$PYTHON_BIN -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| GPU Count:', torch.cuda.device_count())"

# ------------------------------------------------------------------------------
# 3. Tuning & Thread Configuration
# ------------------------------------------------------------------------------
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------------------
# 4. Launch Ingestion Pipeline (Auto-Resuming from Checkpoint)
# ------------------------------------------------------------------------------
# Parameters:
#   --url-list     : List of WDC download URLs to process
#   --model-name   : Alibaba 1024-dim model (or BAAI/bge-large-en-v1.5)
#   --fp16         : Saves vectors as Float16 (halving disk usage to 6.14 TB for 3B vectors)
#   --batch-size   : 256 vectors per inference batch
#   --chunk-size   : 500,000 vectors per sealed binary file
#   --delete-raw   : Deletes temporary raw .json.gz shard to prevent quota overflow
$PYTHON_BIN scripts/process_wdc.py \
    --url-list data/wdc_urls.txt \
    --data-dir data/wdc_raw \
    --output-dir data/wdc_processed \
    --model-name "Alibaba-NLP/gte-large-en-v1.5" \
    --dim 1024 \
    --fp16 \
    --batch-size 256 \
    --chunk-size 500000 \
    --delete-raw

EXIT_CODE=$?
echo "=============================================================================="
echo "Job Execution Finished with Exit Code: $EXIT_CODE at $(date)"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 5. Optional Auto-Resubmit Loop
# ------------------------------------------------------------------------------
# If you want Slurm to automatically queue the next job when this 48h/8h window
# completes until all URLs in data/wdc_urls.txt are finished:
# Uncomment the line below:
# if [ $EXIT_CODE -eq 0 ]; then
#     sbatch scripts/run_wdc_slurm.sh
# fi

exit $EXIT_CODE
