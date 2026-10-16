"""Distributed MPI Hilbert-Partitioned Vector Engine for HPC & Slurm Environments.

Architecture:
  - Range-Partitioned Distributed Hilbert B-Tree: Each MPI rank hosts a contiguous
    segment of the Hilbert curve (one or more LeafSubgraph partitions).
  - Zone Map Distributed Pruning: Ranks whose bounding boxes [attr_min, attr_max]
    do not intersect the query box prune themselves locally with zero network chatter.
  - Broadcast Query -> Parallel Local Subgraph Search -> MPI Gather & Reduction.
  - Designed for hybrid MPI + OpenMP on Slurm clusters (e.g., srun -n $SLURM_NTASKS).
"""

import os
import sys
import time
import numpy as np
import faiss

# Support graceful import of mpi4py with local fallback for testing without MPI
try:
    from mpi4py import MPI
    HAVE_MPI = True
except ImportError:
    HAVE_MPI = False

from hilbert_btree import (
    HilbertSpace,
    LeafSubgraph,
    GraphSparsityTuner,
    MapReduceCollation,
    interleave_subspace_keys,
    deinterleave_subspace_keys,
)


class MockMPIComm:
    """Single-process fallback communicator when mpi4py is not installed."""
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, obj, root=0):
        return obj

    def gather(self, obj, root=0):
        return [obj]

    def Barrier(self):
        pass


def get_mpi_comm():
    """Returns MPI.COMM_WORLD if mpi4py is available, else a single-node mock."""
    if HAVE_MPI:
        return MPI.COMM_WORLD
    return MockMPIComm()


class DistributedMPIHilbertEngine:
    """Distributed Vector Engine distributing Hilbert partition subgraphs across MPI ranks."""

    def __init__(self, X=None, A=None, p=10, max_leaf_size=20000,
                 subspace_dim=2, subspaces=None, comm=None):
        self.comm = comm or get_mpi_comm()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.p = p
        self.max_leaf_size = max_leaf_size
        self.subspace_dim = subspace_dim
        self.subspaces = subspaces

        self.local_leaves = []
        self.space = None

        if self.rank == 0:
            assert X is not None and A is not None, "Rank 0 must provide X and A for initialization"
            self.n, self.d = X.shape
            self.m = A.shape[1]
            amin = A.min(axis=0)
            amax = A.max(axis=0)
            meta = {
                "n": self.n,
                "d": self.d,
                "m": self.m,
                "amin": amin,
                "amax": amax,
                "p": self.p,
                "subspace_dim": self.subspace_dim,
                "subspaces": self.subspaces,
            }
        else:
            meta = None

        # Step 1: Broadcast dataset metadata to all ranks
        meta = self.comm.bcast(meta, root=0)
        self.n = meta["n"]
        self.d = meta["d"]
        self.m = meta["m"]
        self.space = HilbertSpace(meta["amin"], meta["amax"], p=meta["p"], m=self.m,
                                  subspace_dim=meta["subspace_dim"], subspaces=meta["subspaces"])

        # Step 2: Bulk partition and distribute to ranks
        self._distribute_data(X, A)

    def _distribute_data(self, X, A):
        """Rank 0 partitions data by master Hilbert key and scatters assigned chunks to ranks."""
        if self.rank == 0:
            t0 = time.time()
            Aq = self.space.quantize(A)
            keys = self.space.points_to_keys(Aq)
            order = np.argsort(keys)

            # Determine number of global leaf partitions
            num_leaves = max(self.size, int(np.ceil(self.n / (self.max_leaf_size * 0.75))))
            leaf_chunks = np.array_split(order, num_leaves)

            # Assign leaf partitions to MPI ranks (round-robin or contiguous block)
            rank_assignments = [[] for _ in range(self.size)]
            k_start = np.uint64(0)

            for i, chunk_idx in enumerate(leaf_chunks):
                if len(chunk_idx) == 0:
                    continue
                chunk_keys = keys[chunk_idx]
                k_end = self.space.max_key if i == len(leaf_chunks) - 1 else keys[leaf_chunks[i + 1][0]]
                assigned_rank = i % self.size
                rank_assignments[assigned_rank].append({
                    "leaf_id": i,
                    "k_min": k_start,
                    "k_max": k_end,
                    "X": X[chunk_idx],
                    "A": A[chunk_idx],
                    "keys": chunk_keys,
                    "gids": chunk_idx,
                })
                k_start = k_end
            partition_time = time.time() - t0
        else:
            rank_assignments = None

        # Scatter partition payloads to respective ranks
        if HAVE_MPI and self.size > 1:
            my_leaves_data = self.comm.scatter(rank_assignments, root=0)
        else:
            my_leaves_data = rank_assignments[0] if rank_assignments else []

        # Each rank builds its assigned local leaf subgraphs
        t_build_start = time.time()
        for leaf_dict in my_leaves_data:
            deg = GraphSparsityTuner(GraphSparsityTuner.compute_apd(leaf_dict["X"])).compute_degree(leaf_dict["X"])
            leaf = LeafSubgraph(
                leaf_dict["leaf_id"],
                leaf_dict["k_min"],
                leaf_dict["k_max"],
                leaf_dict["X"],
                leaf_dict["A"],
                leaf_dict["keys"],
                leaf_dict["gids"],
                M=deg,
            )
            self.local_leaves.append(leaf)

        self.local_build_time = time.time() - t_build_start
        self.comm.Barrier()

    def get_rank_memory_footprint(self):
        """Calculates memory breakdown for the subgraphs hosted on this MPI rank."""
        leaf_stats = [leaf.get_memory_bytes() for leaf in self.local_leaves]
        total_bytes = sum(s["total_bytes"] for s in leaf_stats)
        vec_bytes = sum(s["vector_bytes"] for s in leaf_stats)
        graph_bytes = sum(s["pure_graph_bytes"] for s in leaf_stats)
        aux_bytes = total_bytes - (vec_bytes + graph_bytes)
        num_vecs = sum(leaf.n for leaf in self.local_leaves)
        return {
            "rank": self.rank,
            "num_leaves": len(self.local_leaves),
            "num_vectors": num_vecs,
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "vec_mb": vec_bytes / (1024 * 1024),
            "graph_mb": graph_bytes / (1024 * 1024),
            "aux_mb": aux_bytes / (1024 * 1024),
        }

    def search(self, q, lo_raw=None, hi_raw=None, k=10, ef=64):
        """Executes a distributed filtered ANN query across all MPI ranks."""
        # Step 1: Broadcast query payload from Rank 0
        if self.rank == 0:
            query_payload = (q, lo_raw, hi_raw, k, ef)
        else:
            query_payload = None

        t_bcast0 = time.perf_counter()
        q, lo_raw, hi_raw, k, ef = self.comm.bcast(query_payload, root=0)
        t_bcast = (time.perf_counter() - t_bcast0) * 1000.0

        t_local0 = time.perf_counter()
        local_cand_lists = []
        local_ndis = 0
        local_pruned = 0
        local_enclosed = 0
        local_boundary = 0

        # Fast Path: Pure Unconstrained ANN Search
        if lo_raw is None or hi_raw is None:
            for leaf in self.local_leaves:
                ids, dists, nd = leaf.search_unconstrained(q, k=k, ef=ef)
                local_cand_lists.append((ids, dists))
                local_ndis += nd
        else:
            # Filtered Path: Zone Map Pruning & Leaf Search
            for leaf in self.local_leaves:
                # 1. Zone Map Check: Is leaf disjoint from query bounding box?
                if np.any(leaf.attr_min > hi_raw) or np.any(leaf.attr_max < lo_raw):
                    local_pruned += 1
                    continue

                # 2. Zone Map Enclosed Check
                if np.all(leaf.attr_min >= lo_raw) and np.all(leaf.attr_max <= hi_raw):
                    local_enclosed += 1
                    ids, dists, nd = leaf.search_unconstrained(q, k=k, ef=ef)
                    local_cand_lists.append((ids, dists))
                    local_ndis += nd
                    continue

                # 3. Boundary Leaf Check
                local_boundary += 1
                matching_ids = leaf.find_matching_local([], lo_raw, hi_raw, self.space)
                if len(matching_ids) == 0:
                    continue

                scan_thresh = max(500, int(0.15 * self.max_leaf_size))
                if len(matching_ids) <= scan_thresh:
                    ids, dists, nd = leaf.search_scan(q, matching_ids, k=k)
                else:
                    ids, dists, nd = leaf.search_seeded_hnsw(q, matching_ids, k=k, ef=ef)

                local_cand_lists.append((ids, dists))
                local_ndis += nd

        # Reduce locally on rank to top-k candidates before MPI gather
        if local_cand_lists:
            top_ids, top_dists = MapReduceCollation.flat_reduce(local_cand_lists, k=k)
        else:
            top_ids = np.empty(0, dtype=np.int64)
            top_dists = np.empty(0, dtype=np.float32)

        t_local_ms = (time.perf_counter() - t_local0) * 1000.0

        # Step 2: MPI Gather local top-k candidates to Rank 0
        t_gather0 = time.perf_counter()
        local_result = {
            "ids": top_ids,
            "dists": top_dists,
            "ndis": local_ndis,
            "pruned": local_pruned,
            "enclosed": local_enclosed,
            "boundary": local_boundary,
            "local_time_ms": t_local_ms,
        }
        all_results = self.comm.gather(local_result, root=0)
        t_gather = (time.perf_counter() - t_gather0) * 1000.0

        # Step 3: Collate global top-k on Rank 0
        if self.rank == 0:
            rank_cand_lists = [(r["ids"], r["dists"]) for r in all_results if len(r["ids"]) > 0]
            if rank_cand_lists:
                global_ids, global_dists = MapReduceCollation.flat_reduce(rank_cand_lists, k=k)
            else:
                global_ids = np.empty(0, dtype=np.int64)
                global_dists = np.empty(0, dtype=np.float32)

            total_ndis = sum(r["ndis"] for r in all_results)
            total_pruned = sum(r["pruned"] for r in all_results)
            total_enclosed = sum(r["enclosed"] for r in all_results)
            total_boundary = sum(r["boundary"] for r in all_results)
            max_local_time = max(r["local_time_ms"] for r in all_results)

            stats = {
                "ndis": total_ndis,
                "pruned_leaves": total_pruned,
                "enclosed_leaves": total_enclosed,
                "boundary_leaves": total_boundary,
                "bcast_time_ms": t_bcast,
                "gather_time_ms": t_gather,
                "max_local_time_ms": max_local_time,
                "active_ranks": sum(1 for r in all_results if r["ndis"] > 0),
            }
            return global_ids, global_dists, stats
        else:
            return None, None, None
