# Distributed MPI Hilbert B-Tree Vector Engine on HPC & Slurm

## 1. Overview & Architecture

To scale vector search with multi-attribute filtering beyond single-node CPU boundaries, the Hilbert-Partitioned B-Tree is adapted for **Distributed High-Performance Computing (HPC) clusters** managed by **Slurm** job queues using the Message Passing Interface (**MPI** via `mpi4py`).

```
                    +------------------------------------+
                    |        Rank 0 (Coordinator)        |
                    |   - Receives & Quantizes Data      |
                    |   - Bit-Interleaved Hilbert Order  |
                    |   - Broadcasts Queries (bcast)     |
                    |   - Final Top-k Reduction (gather) |
                    +-----------------+------------------+
                                      |
         +----------------------------+----------------------------+
         | (bcast query)              | (bcast query)              | (bcast query)
         v                            v                            v
+------------------+         +------------------+         +------------------+
|      Rank 0      |         |      Rank 1      |         |   Rank (P - 1)   |
| Leaves [0 .. L0] |         | Leaves [L0 .. L1]|         | Leaves [.. Lend] |
| - Zone Map Check |         | - Zone Map Check |         | - Zone Map Check |
| - SIMD / HNSW    |         | - SIMD / HNSW    |         | - SIMD / HNSW    |
| - Local Top-k    |         | - Local Top-k    |         | - Local Top-k    |
+--------+---------+         +--------+---------+         +--------+---------+
         |                            |                            |
         +----------------------------+----------------------------+
                                      | (gather candidate lists)
                                      v
                    +------------------------------------+
                    |  Hierarchical Top-K Tree Reduce    |
                    |  Global Top-K Result Extraction    |
                    +------------------------------------+
```

### Key Distributed Mechanics
1. **Range-Partitioned Distributed Curve**:
   - The global dataset is sorted by the 60-bit Subspace Bit-Interleaved Hilbert key.
   - The contiguous key space is chunked into leaf partitions and scattered across $P$ MPI ranks.
   - Each MPI rank owns a disjoint, localized segment of the attribute space.
2. **Distributed Zone Map Pruning (Zero Network Chatter)**:
   - Each rank maintains local `attr_min` and `attr_max` bounding boxes for its assigned partitions.
   - When Rank 0 broadcasts a query box $[lo, hi]$, any rank whose bounding boxes are disjoint from the query box prunes itself immediately, contributing **zero distance computations** and sending back an empty candidate list without exploring memory.
3. **Sub-Millisecond Network Overhead**:
   - Only small query parameters $(q, lo, hi, k, ef)$ are broadcast downstream ($< 1 \text{ KB}$).
   - Each rank only sends back its local top-$k$ candidates (e.g. 10 IDs + 10 floats $\approx 120 \text{ bytes}$).
   - Even across 64 cluster nodes, total gathered data is $< 8 \text{ KB}$, completing across InfiniBand/Omni-Path in $< 10\,\mu\text{s}$.
4. **Hybrid MPI + OpenMP**:
   - **MPI** scales across compute nodes ($P = \text{SLURM\_NTASKS}$).
   - **OpenMP / ThreadPool** scales locally within each node across CPU cores ($\text{OMP\_NUM\_THREADS} = \text{SLURM\_CPUS\_PER\_TASK}$) for local leaf subgraphs.

---

## 2. HPC Environment Setup

### Step A: Load Cluster Modules (e.g. Stony Brook SeaWulf)
```bash
module purge
module load openmpi/4.1.4
module load python/3.10
# or gcc/11.2.0, intel-oneapi-mpi, etc.
```

### Step B: Create Cluster Conda Environment
Install `mpi4py` built against the cluster's native OpenMPI/MPI:
```bash
# Using conda-forge ensures mpi4py links cleanly with cluster MPI
conda create -n cse532-mpi python=3.10 -y
conda activate cse532-mpi

# Install MPI and Faiss packages
conda install -c conda-forge mpi4py faiss-cpu numpy -y
pip install hilbertcurve
```

Verify MPI installation:
```bash
python -c "from mpi4py import MPI; print('MPI initialized:', MPI.Is_initialized(), 'Size:', MPI.COMM_WORLD.Get_size())"
```

---

## 3. Running Distributed Jobs via Slurm

### A. Batch Job Submission (`sbatch`)
Submit the pre-configured Slurm script:
```bash
sbatch scripts/run_slurm_hpc.sh
```

Inspect job queue status:
```bash
squeue -u $USER
```

Monitor live output:
```bash
tail -f logs/mpi_*.out
```

### B. Interactive Slurm Allocation (`salloc`)
To run live interactive benchmarks across compute nodes:
```bash
# Request 2 nodes, 8 ranks per node, 4 cores per rank for 30 minutes
salloc --nodes=2 --ntasks-per-node=8 --cpus-per-task=4 --time=00:30:00 --partition=short

# Once allocated on compute nodes:
conda activate cse532-mpi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Launch distributed benchmark across the 16 ranks:
srun python prototype/bench_mpi.py --n 1000000 --m 12 --max-leaf-size 20000 --nq 50
```

---

## 4. Benchmark CLI Parameters

| Flag | Default | Description |
| :--- | :---: | :--- |
| `--n` | `100000` | Number of vectors to generate and index (e.g. `1000000` or `10000000`). |
| `--m` / `--n-attr` | `12` | Number of scalar attributes to potentially filter on. |
| `--max-leaf-size` | `20000` | Maximum vectors per leaf subgraph partition before splitting. |
| `--nq` | `20` | Number of test queries per selectivity level. |

---

## 5. Scaling to 10 Million Vectors ($N = 10{,}000{,}000$)

To scale to 10 million vectors on an HPC cluster:
- **Node Allocation**: 8 nodes $\times$ 8 tasks/node = 64 MPI ranks.
- **Vectors per Rank**: $\approx 156{,}250$ vectors per rank.
- **Leaf Partitions per Rank**: $\approx 8$ leaves of size $20{,}000$.
- **Submission Command**:
  ```bash
  srun -n 64 python prototype/bench_mpi.py --n 10000000 --m 12 --max-leaf-size 20000 --nq 100
  ```
Each rank retains its entire working set within local RAM/cache, sustaining sub-millisecond local compute times across massive dataset scales.
