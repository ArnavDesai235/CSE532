"""Head-to-head benchmark comparing Hilbert-Partitioned B-Tree against Traditional KNN Graph baselines.

Baselines evaluated:
  1. Traditional HNSW (Post-Filtering): Monolithic Faiss HNSW index, searches top-ef, filters matches.
  2. Traditional HNSW (Filtered Beam Search): Global HNSW graph traversal with in-graph predicate filtering.
  3. Exact Linear Scan (Pre-filtered SIMD Scan): Baseline scanning only matching attribute records.
  4. Hilbert-Partitioned B-Tree (Our Approach): Zone Map leaf pruning + parallel subgraphs + shared-state early exit.

Usage:
  python prototype/bench_compare_baselines.py [--corr]
"""

import heapq
import sys
import time
import numpy as np
import faiss
from hilbert_btree import HilbertHNSWEngine


def run_comparison():
    rng = np.random.default_rng(42)
    n, d, nq, k, ef = 40_000, 32, 50, 10, 64
    max_leaf_size = 10_000

    print("=" * 95)
    print("CSE532 Comparative Benchmark: Hilbert-Partitioned B-Tree vs. Traditional KNN Graphs")
    print("=" * 95)

    corr = "--corr" in sys.argv
    X = rng.standard_normal((n, d)).astype("float32")
    if corr:
        A = X[:, :2] + 0.4 * rng.standard_normal((n, 2))
        A = (A - A.min(0)) / (A.max(0) - A.min(0))
        print("Attribute Distribution: CORRELATED with vector embedding space")
    else:
        A = rng.random((n, 2)).astype("float32")
        print("Attribute Distribution: INDEPENDENT uniform attributes")

    Q = rng.standard_normal((nq, d)).astype("float32")

    # 1. Build Traditional Monolithic HNSW Index
    print(f"\n[1] Building Traditional Monolithic HNSW Graph (N={n}, d={d}, M=20, efConstruction=100)...")
    t0 = time.time()
    trad_index = faiss.IndexHNSWFlat(d, 20)
    trad_index.hnsw.efConstruction = 100
    trad_index.add(X)
    trad_build_time = time.time() - t0
    print(f"    Built monolithic HNSW index in {trad_build_time:.2f}s")

    # Extract graph adjacency for in-graph filtered beam search
    h = trad_index.hnsw
    nbrs = faiss.vector_to_array(h.neighbors)
    offs = faiss.vector_to_array(h.offsets)
    nb0 = h.nb_neighbors(0)

    # 2. Build Hilbert-Partitioned B-Tree Engine
    print(f"\n[2] Building Hilbert B-Tree of Subgraphs (N={n}, d={d}, max_leaf_size={max_leaf_size})...")
    t0 = time.time()
    engine = HilbertHNSWEngine(X, A, p=10, max_leaf_size=max_leaf_size,
                               M_min=12, M_max=32, M_base=20, scan_threshold=300)
    btree_build_time = time.time() - t0
    num_leaves = len(engine.router.leaves)
    degrees = [leaf.M for leaf in engine.router.leaves]
    print(f"    Built {num_leaves} partitioned leaf subgraphs in {btree_build_time:.2f}s")
    print(f"    Adaptive Sparsity Degrees across leaves: {degrees}")

    def calc_recall(found, truth):
        if len(truth) == 0:
            return 1.0
        return len(set(found.tolist()) & set(truth.tolist())) / len(truth)

    def dist_sq(q, idxs):
        diff = X[idxs] - q
        return np.einsum("ij,ij->i", diff, diff)

    def traditional_filtered_beam_search(q, valid_mask, k=10, ef=64):
        """Standard filtered greedy beam search on monolithic HNSW graph."""
        trad_index.hnsw.efSearch = 16
        _, I = trad_index.search(q[None], 4)
        seeds = [i for i in I[0] if i >= 0]
        if not seeds:
            seeds = [0]

        visited = np.zeros(n, dtype=bool)
        visited[seeds] = True
        cand = [(float(d), int(i)) for d, i in zip(dist_sq(q, seeds), seeds)]
        heapq.heapify(cand)
        res = [(-float(d), int(i)) for d, i in cand if valid_mask[i]]
        heapq.heapify(res)

        ndis = len(seeds) + 16 * 2
        while len(res) > ef:
            heapq.heappop(res)

        while cand:
            d, u = heapq.heappop(cand)
            if len(res) >= ef and d > -res[0][0]:
                break
            nb = nbrs[offs[u]: offs[u] + nb0]
            nb = nb[nb >= 0]
            nb = nb[~visited[nb]]
            if nb.size == 0:
                continue
            visited[nb] = True
            dn = dist_sq(q, nb)
            ndis += nb.size
            worst = -res[0][0] if len(res) >= ef else np.inf
            for dv, v in zip(dn, nb):
                if dv < worst:
                    heapq.heappush(cand, (float(dv), int(v)))
                    if valid_mask[v]:
                        heapq.heappush(res, (-float(dv), int(v)))
                        if len(res) > ef:
                            heapq.heappop(res)
                        worst = -res[0][0] if len(res) >= ef else np.inf

        out = sorted((-neg_d, i) for neg_d, i in res)[:k]
        return np.array([i for _, i in out], dtype=np.int64), ndis

    # 3. Sweep across Selectivity
    print("\n[3] Head-to-Head Performance Evaluation across Selectivity Spectrum:")
    header = (f"{'Sel':>6} | "
              f"{'Trad. Post Recall':>17} | {'Trad. Beam (Recall/Dist)':>24} | "
              f"{'Hilbert B-Tree (Recall/Dist/Lat)':>32} | {'Scan Dist':>9}")
    print(header)
    print("-" * len(header))

    for sel in [0.005, 0.02, 0.05, 0.10, 0.25]:
        r_post = []
        r_beam, nd_beam = [], []
        r_btree, nd_btree, lat_btree = [], [], []
        nd_scan = []

        for qi in range(nq):
            side = np.sqrt(sel)
            c = rng.random(2) * (1.0 - side)
            lo, hi = c, c + side

            # Exact ground truth
            in_range = np.all((A >= lo) & (A <= hi), axis=1)
            valid_gids = np.where(in_range)[0]
            if len(valid_gids) < k:
                continue

            diff = X[valid_gids] - Q[qi]
            true_dists = np.einsum("ij,ij->i", diff, diff)
            truth = valid_gids[np.argsort(true_dists)[:k]]
            valid_set = set(valid_gids.tolist())

            # 1. Baseline: Traditional HNSW Post-Filtering
            trad_index.hnsw.efSearch = ef
            _, I = trad_index.search(Q[qi][None], ef)
            post_matches = [i for i in I[0] if i in valid_set][:k]
            r_post.append(calc_recall(np.array(post_matches), truth))

            # 2. Baseline: Traditional In-Graph Filtered Beam Search
            beam_ids, nd_b = traditional_filtered_beam_search(Q[qi], in_range, k=k, ef=ef)
            r_beam.append(calc_recall(beam_ids, truth))
            nd_beam.append(nd_b)

            # 3. Baseline: Scan Cost
            nd_scan.append(len(valid_gids))

            # 4. Our Hilbert-Partitioned B-Tree Engine
            t0 = time.perf_counter()
            btree_ids, stats = engine.search(Q[qi], lo_raw=lo, hi_raw=hi, k=k, ef=ef, parallel=True)
            lat_btree.append((time.perf_counter() - t0) * 1000.0)
            r_btree.append(calc_recall(btree_ids, truth))
            nd_btree.append(stats["ndis"])

        avg_post = np.mean(r_post) if r_post else 0.0
        avg_r_beam = np.mean(r_beam) if r_beam else 0.0
        avg_nd_beam = np.mean(nd_beam) if nd_beam else 0.0
        avg_r_btree = np.mean(r_btree) if r_btree else 0.0
        avg_nd_btree = np.mean(nd_btree) if nd_btree else 0.0
        avg_lat_btree = np.mean(lat_btree) if lat_btree else 0.0
        avg_scan = np.mean(nd_scan) if nd_scan else 0.0

        beam_str = f"{avg_r_beam:.3f} / {avg_nd_beam:6.0f}"
        btree_str = f"{avg_r_btree:.3f} / {avg_nd_btree:6.0f} / {avg_lat_btree:5.2f}ms"
        print(f"{sel:>6.3f} | {avg_post:>17.3f} | {beam_str:>24} | {btree_str:>32} | {avg_scan:>9.0f}")

    # 4. Ingestion / Streaming Maintenance Comparison
    print("\n[4] Streaming Appends Maintenance Cost (8,000 Vectors Ingestion):")
    n_appends = 8_000
    X_new = rng.standard_normal((n_appends, d)).astype("float32")
    A_new = rng.random((n_appends, 2)).astype("float32")

    # Traditional HNSW sequential insertion
    t0 = time.time()
    trad_index.add(X_new)
    trad_add_time = time.time() - t0
    print(f"    Traditional Monolithic HNSW add({n_appends} vectors): {trad_add_time:.2f}s")
    print(f"      -> Required global graph insertion with O(N * M * log N) cross-graph rewiring.")

    # Hilbert B-Tree Temporal Append & Localized Compaction
    t0 = time.time()
    for i in range(n_appends):
        engine.insert(X_new[i], A_new[i], global_id=n + i)
    engine.flush()
    btree_add_time = time.time() - t0
    print(f"    Hilbert B-Tree L0 Buffer + Compaction ({n_appends} vectors): {btree_add_time:.2f}s")
    print(f"      -> ZERO cross-partition rewiring; executed localized median splits only on full leaves.")

    engine.close()
    print("\n" + "=" * 95)
    print("Comparative Benchmark Completed Successfully!")
    print("=" * 95)


if __name__ == "__main__":
    run_comparison()
