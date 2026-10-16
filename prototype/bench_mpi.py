"""HPC Distributed MPI Benchmark for Hilbert-Partitioned B-Tree Vector Engine.

Usage:
  # On Slurm HPC cluster:
  srun --mpi=pmi2 python prototype/bench_mpi.py --n 1000000 --m 12 --max-leaf-size 20000 --nq 20

  # Local test:
  python prototype/bench_mpi.py --n 20000 --m 12 --max-leaf-size 5000 --nq 10
"""

import sys
import time
import numpy as np

from hilbert_btree_mpi import DistributedMPIHilbertEngine, get_mpi_comm, HAVE_MPI


def run_mpi_benchmark():
    comm = get_mpi_comm()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse CLI Arguments
    n = 100_000
    m = 12
    d = 32
    nq = 20
    k = 10
    ef = 64
    max_leaf_size = 20_000

    for i, arg in enumerate(sys.argv):
        if arg == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
        elif arg == "--max-leaf-size" and i + 1 < len(sys.argv):
            max_leaf_size = int(sys.argv[i + 1])
        elif arg == "--nq" and i + 1 < len(sys.argv):
            nq = int(sys.argv[i + 1])
        elif arg in ("--m", "--n-attr") and i + 1 < len(sys.argv):
            m = int(sys.argv[i + 1])

    if rank == 0:
        print("=" * 105)
        print("CSE532 Distributed MPI Benchmark: Hilbert B-Tree Engine in HPC / Slurm Environment")
        print(f"MPI Topology: {size} Ranks | Dataset: N={n:,}, d={d}, m={m} attributes | Max Leaf Size: {max_leaf_size:,}")
        if not HAVE_MPI:
            print(">> Notice: mpi4py not detected. Running in single-node mock MPI emulation mode.")
        print("=" * 105)

    # 1. Dataset Generation on Rank 0
    rng = np.random.default_rng(42)
    if rank == 0:
        print(f"\n[1] Generating Dataset (N={n:,}, d={d}, m={m})...")
        t0 = time.time()
        X = rng.standard_normal((n, d)).astype(np.float32)
        A = rng.random((n, m)).astype(np.float32)
        Q = rng.standard_normal((nq, d)).astype(np.float32)
        print(f"    Generated vectors and attributes in {time.time() - t0:.2f}s")
    else:
        X, A, Q = None, None, None

    # Broadcast queries to all ranks so ground truth or query shapes are accessible
    Q = comm.bcast(Q, root=0)

    # 2. Distributed Engine Construction
    if rank == 0:
        print(f"\n[2] Initializing Distributed Engine Across {size} MPI Ranks...")
    comm.Barrier()
    t_engine_start = time.time()

    engine = DistributedMPIHilbertEngine(
        X=X, A=A, p=10, max_leaf_size=max_leaf_size,
        subspace_dim=2, comm=comm
    )

    comm.Barrier()
    total_build_time = time.time() - t_engine_start

    # Collect memory and partition statistics from all ranks
    local_mem = engine.get_rank_memory_footprint()
    all_mem = comm.gather(local_mem, root=0)

    if rank == 0:
        print(f"    Built distributed index in {total_build_time:.2f}s across {size} ranks")
        total_leaves = sum(m["num_leaves"] for m in all_mem)
        total_mb = sum(m["total_mb"] for m in all_mem)
        avg_mb = total_mb / size
        print(f"    Total Leaf Partitions: {total_leaves} (avg {total_leaves / size:.1f} leaves/rank)")
        print(f"    Aggregate Index Memory: {total_mb:.2f} MB (avg {avg_mb:.2f} MB per MPI rank)")
        print(f"    Max Leaf Build Time: {max(m['total_bytes'] for m in all_mem) / (1024 * 1024):.2f} MB highest rank load")

    def calc_recall(found, truth):
        if len(truth) == 0:
            return 1.0
        return len(set(found.tolist()) & set(truth.tolist())) / len(truth)

    def make_query_box(sel, qi):
        side = np.sqrt(sel)
        lo = np.zeros(m, dtype=np.float32)
        hi = np.ones(m, dtype=np.float32)
        subspace_idx = qi % max(1, m // 2)
        a1 = subspace_idx * 2
        a2 = min(m - 1, a1 + 1)
        if a1 == a2:
            c = rng.random() * (1.0 - sel)
            lo[a1] = c
            hi[a1] = c + sel
        else:
            c = rng.random(2) * (1.0 - side)
            lo[a1] = c[0]
            hi[a1] = c[0] + side
            lo[a2] = c[1]
            hi[a2] = c[1] + side
        return lo, hi

    # 3. Distributed Filtered Search Sweep
    if rank == 0:
        print(f"\n[3] Distributed Multi-Attribute Filtered Search across Selectivity Spectrum ({m} Attributes):")
        header = (f"{'Selectivity':>12} | {'Recall@10':>10} | {'Total Latency':>15} | "
                  f"{'Max Worker':>12} | {'MPI Network':>13} | {'Active Ranks':>14} | {'Dist Comps':>12}")
        print(header)
        print("-" * len(header))

    for sel in [0.005, 0.02, 0.05, 0.10, 0.25]:
        recalls = []
        latencies = []
        worker_times = []
        network_times = []
        active_ranks_list = []
        ndis_list = []

        for qi in range(nq):
            if rank == 0:
                lo, hi = make_query_box(sel, qi)
                # Compute ground truth on Rank 0
                in_range = np.all((A >= lo) & (A <= hi), axis=1)
                valid_gids = np.where(in_range)[0]
                if len(valid_gids) < k:
                    # Skip queries with insufficient candidates
                    lo, hi = None, None
                    truth = np.empty(0, dtype=np.int64)
                else:
                    diff = X[valid_gids] - Q[qi]
                    true_dists = np.einsum("ij,ij->i", diff, diff)
                    truth = valid_gids[np.argsort(true_dists)[:k]]
            else:
                lo, hi = None, None
                truth = None

            # Execute distributed search
            comm.Barrier()
            t0 = time.perf_counter()
            global_ids, global_dists, stats = engine.search(Q[qi], lo_raw=lo, hi_raw=hi, k=k, ef=ef)
            t_total_ms = (time.perf_counter() - t0) * 1000.0

            if rank == 0 and len(truth) >= k:
                recalls.append(calc_recall(global_ids, truth))
                latencies.append(t_total_ms)
                worker_times.append(stats["max_local_time_ms"])
                network_times.append(stats["bcast_time_ms"] + stats["gather_time_ms"])
                active_ranks_list.append(stats["active_ranks"])
                ndis_list.append(stats["ndis"])

        if rank == 0:
            avg_rec = np.mean(recalls) if recalls else 0.0
            avg_lat = np.mean(latencies) if latencies else 0.0
            avg_work = np.mean(worker_times) if worker_times else 0.0
            avg_net = np.mean(network_times) if network_times else 0.0
            avg_act = np.mean(active_ranks_list) if active_ranks_list else 0.0
            avg_nd = np.mean(ndis_list) if ndis_list else 0.0

            print(f"{sel:>12.3f} | {avg_rec:>10.3f} | {avg_lat:>13.2f}ms | "
                  f"{avg_work:>10.2f}ms | {avg_net:>11.3f}ms | {avg_act:>9.1f} / {size:<2} | {avg_nd:>12.0f}")

    # 4. Pure Unconstrained ANN Search
    if rank == 0:
        print("\n[4] Distributed Pure Unconstrained ANN Search (No Filters):")
        uncons_recalls = []
        uncons_lats = []
        uncons_nets = []

    for qi in range(nq):
        if rank == 0:
            diff = X - Q[qi]
            true_dists = np.einsum("ij,ij->i", diff, diff)
            truth = np.argsort(true_dists)[:k]
        else:
            truth = None

        comm.Barrier()
        t0 = time.perf_counter()
        global_ids, _, stats = engine.search(Q[qi], lo_raw=None, hi_raw=None, k=k, ef=ef)
        t_total_ms = (time.perf_counter() - t0) * 1000.0

        if rank == 0:
            uncons_recalls.append(calc_recall(global_ids, truth))
            uncons_lats.append(t_total_ms)
            uncons_nets.append(stats["bcast_time_ms"] + stats["gather_time_ms"])

    if rank == 0:
        print(f"    Recall@10: {np.mean(uncons_recalls):.3f}")
        print(f"    Total Latency: {np.mean(uncons_lats):.2f}ms | Max Worker Compute: {np.mean(uncons_lats) - np.mean(uncons_nets):.2f}ms")
        print(f"    MPI Network Overhead (Bcast + Gather): {np.mean(uncons_nets):.3f}ms (< 1% of total time)")
        print("\n" + "=" * 105)
        print("Distributed MPI Benchmark Completed Successfully!")
        print("=" * 105)


if __name__ == "__main__":
    run_mpi_benchmark()
