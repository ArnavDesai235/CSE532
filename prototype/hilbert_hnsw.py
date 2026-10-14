"""Hilbert-curve-assisted filtered HNSW (prototype).

The Hilbert curve over the *attribute* dimensions does three jobs:
  1. Selectivity estimate: a box predicate decomposes into Hilbert key intervals;
     binary search on the sorted keys gives an (over-)count in O(#intervals * log n).
  2. Scan path: matching ids are contiguous slices of `order` -> brute force for tiny sets.
  3. Graph path: the same slices give (a) a validity bitmap for filtered traversal and
     (b) a pool of *valid* entry points; we seed layer-0 beam search with the valid points
     closest to q, plus a few vector-near (unfiltered) points for navigation.
"""
import heapq
import numpy as np
import faiss
from hilbertcurve.hilbertcurve import HilbertCurve


class HilbertHNSW:
    def __init__(self, X, A, p=10, M=32, ef_construction=200):
        self.X = np.ascontiguousarray(X, dtype="float32")
        self.n, self.d = self.X.shape
        self.m = A.shape[1]
        assert self.m * p <= 63, "keys must fit in uint64"
        self.p = p
        self.hc = HilbertCurve(p, self.m)

        # Quantize attributes to the p-bit grid.
        self.amin, self.amax = A.min(0), A.max(0)
        self.Aq = self.quantize(A)
        keys = np.array(self.hc.distances_from_points(self.Aq.tolist()), dtype=np.uint64)
        self.order = np.argsort(keys, kind="stable")
        self.skeys = keys[self.order]

        # Standard HNSW; we reuse its graph and run our own layer-0 search.
        self.index = faiss.IndexHNSWFlat(self.d, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.add(self.X)
        h = self.index.hnsw
        self.nbrs = faiss.vector_to_array(h.neighbors)
        self.offs = faiss.vector_to_array(h.offsets)
        self.nb0 = h.nb_neighbors(0)  # level 0 occupies the first nb0 slots per node

    def quantize(self, A):
        span = np.where(self.amax > self.amin, self.amax - self.amin, 1)
        q = np.floor((A - self.amin) / span * (2**self.p - 1) + 0.5)
        return np.clip(q, 0, 2**self.p - 1).astype(np.int64)

    # ---------- Hilbert range decomposition ----------
    def decompose(self, lo, hi, max_depth=None):
        """Box [lo, hi] in quantized coords -> merged list of [start, end) key intervals.
        Aligned dyadic cubes are contiguous on the Hilbert curve, so each fully-covered
        cube is one interval. Cubes still partial at max_depth are included (over-approx)."""
        max_depth = self.p if max_depth is None else max_depth
        out = []

        def rec(level, corner):
            side = 1 << (self.p - level)
            cmax = corner + side - 1
            if np.any(cmax < lo) or np.any(corner > hi):
                return
            if np.all(corner >= lo) and np.all(cmax <= hi) or level == max_depth:
                size = 1 << (self.m * (self.p - level))
                start = (self.hc.distance_from_point(corner.tolist()) // size) * size
                out.append((start, start + size))
                return
            half = side >> 1
            for bits in range(1 << self.m):
                off = np.array([(bits >> j) & 1 for j in range(self.m)]) * half
                rec(level + 1, corner + off)

        rec(0, np.zeros(self.m, dtype=np.int64))
        out.sort()
        merged = []
        for s, e in out:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    def matching(self, lo_raw, hi_raw, max_depth=6):
        """Returns (valid_ids, estimated_count, n_intervals)."""
        lo, hi = self.quantize(np.asarray(lo_raw)), self.quantize(np.asarray(hi_raw))
        ivs = self.decompose(lo, hi, max_depth)
        sk = self.skeys
        slices = [(np.searchsorted(sk, np.uint64(s)), np.searchsorted(sk, np.uint64(e))) for s, e in ivs]
        est = sum(b - a for a, b in slices)
        if est == 0:
            return np.empty(0, dtype=np.int64), 0, len(ivs)
        cand = np.concatenate([self.order[a:b] for a, b in slices if b > a])
        # Exact refinement (only matters for over-approximated boundary cells).
        aq = self.Aq[cand]
        ok = np.all((aq >= lo) & (aq <= hi), axis=1)
        return cand[ok], est, len(ivs)

    # ---------- search ----------
    def _dist(self, q, ids):
        diff = self.X[ids] - q
        return np.einsum("ij,ij->i", diff, diff)

    def search(self, q, lo_raw, hi_raw, k=10, ef=128, scan_threshold=2000,
               n_seeds=8, seed_budget=512, nav_seeds=4, use_hilbert_seeds=True,
               two_hop=False, rng=None):
        rng = rng or np.random.default_rng(0)
        q = np.asarray(q, dtype="float32")
        valid_ids, est, _ = self.matching(lo_raw, hi_raw)
        stats = {"est": est, "valid": len(valid_ids), "ndis": 0}
        if len(valid_ids) == 0:
            return np.empty(0, dtype=np.int64), stats
        if len(valid_ids) <= scan_threshold:
            d = self._dist(q, valid_ids)
            stats.update(mode="scan", ndis=len(valid_ids))
            top = np.argsort(d)[:k]
            return valid_ids[top], stats

        valid = np.zeros(self.n, dtype=bool)
        valid[valid_ids] = True

        seeds = []
        if nav_seeds:  # vector-near points from the normal HNSW descent (may be invalid)
            self.index.hnsw.efSearch = max(nav_seeds, 16)
            _, I = self.index.search(q[None], nav_seeds)
            seeds += [i for i in I[0] if i >= 0]
            stats["ndis"] += self.index.hnsw.efSearch * 2  # rough
        if use_hilbert_seeds:  # valid points closest to q, sampled from the curve slices
            pool = valid_ids if len(valid_ids) <= seed_budget else rng.choice(valid_ids, seed_budget, replace=False)
            dp = self._dist(q, pool)
            stats["ndis"] += len(pool)
            seeds += pool[np.argsort(dp)[:n_seeds]].tolist()

        seeds = np.unique(seeds)
        if two_hop:
            ids, ndis = self._beam0_valid(q, seeds[valid[seeds]], valid, k, ef)
        else:
            ids, ndis = self._beam0(q, seeds, valid, k, ef)
        stats.update(mode="graph", ndis=stats["ndis"] + ndis)
        return ids, stats

    def _beam0_valid(self, q, seeds, valid, k, ef):
        """ACORN-style layer-0 search: distances are computed only for valid nodes; invalid
        neighbors are hopped over (their neighbors expanded). Entry points matter here,
        because the valid-induced subgraph can be poorly connected."""
        if len(seeds) == 0:
            return np.empty(0, dtype=np.int64), 0
        visited = np.zeros(self.n, dtype=bool)
        visited[seeds] = True
        ds = self._dist(q, seeds)
        ndis = len(seeds)
        cand = [(float(d), int(i)) for d, i in zip(ds, seeds)]
        heapq.heapify(cand)
        res = [(-d, i) for d, i in cand]
        heapq.heapify(res)
        while len(res) > ef:
            heapq.heappop(res)
        while cand:
            d, u = heapq.heappop(cand)
            if len(res) >= ef and d > -res[0][0]:
                break
            nb = self.nbrs[self.offs[u]: self.offs[u] + self.nb0]
            nb = nb[nb >= 0]
            hop = nb[~valid[nb]]
            if hop.size:  # 2-hop through invalid neighbors
                nb2 = np.concatenate([self.nbrs[self.offs[h]: self.offs[h] + self.nb0] for h in hop])
                nb = np.concatenate([nb, nb2[nb2 >= 0]])
            nb = np.unique(nb[valid[nb]])
            nb = nb[~visited[nb]]
            if nb.size == 0:
                continue
            visited[nb] = True
            dn = self._dist(q, nb)
            ndis += nb.size
            for dv, v in zip(dn, nb):
                if len(res) < ef or dv < -res[0][0]:
                    heapq.heappush(cand, (float(dv), int(v)))
                    heapq.heappush(res, (-float(dv), int(v)))
                    if len(res) > ef:
                        heapq.heappop(res)
        out = sorted((-nd, i) for nd, i in res)[:k]
        return np.array([i for _, i in out], dtype=np.int64), ndis

    def _beam0(self, q, seeds, valid, k, ef):
        """Layer-0 beam search: invalid nodes may be traversed (navigation) but never returned."""
        visited = np.zeros(self.n, dtype=bool)
        visited[seeds] = True
        ds = self._dist(q, seeds)
        ndis = len(seeds)
        cand = [(float(d), int(i)) for d, i in zip(ds, seeds)]
        heapq.heapify(cand)
        res = [(-float(d), int(i)) for d, i in zip(ds, seeds) if valid[i]]  # max-heap
        heapq.heapify(res)
        while len(res) > ef:
            heapq.heappop(res)
        while cand:
            d, u = heapq.heappop(cand)
            if len(res) >= ef and d > -res[0][0]:
                break
            nb = self.nbrs[self.offs[u]: self.offs[u] + self.nb0]
            nb = nb[nb >= 0]
            nb = nb[~visited[nb]]
            if nb.size == 0:
                continue
            visited[nb] = True
            dn = self._dist(q, nb)
            ndis += nb.size
            worst = -res[0][0] if len(res) >= ef else np.inf
            for dv, v in zip(dn, nb):
                if dv < worst:
                    heapq.heappush(cand, (float(dv), int(v)))
                    if valid[v]:
                        heapq.heappush(res, (-float(dv), int(v)))
                        if len(res) > ef:
                            heapq.heappop(res)
                        worst = -res[0][0] if len(res) >= ef else np.inf
        out = sorted((-nd, i) for nd, i in res)[:k]
        return np.array([i for _, i in out], dtype=np.int64), ndis
