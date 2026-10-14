"""Compare post-filtering vs filtered beam search with/without Hilbert entry points.
Usage: python bench.py [--corr]   (--corr makes attributes correlated with vectors)"""
import sys, time
import numpy as np
from hilbert_hnsw import HilbertHNSW

rng = np.random.default_rng(42)
n, d, nq, k, ef = 50_000, 32, 50, 10, 64
X = rng.standard_normal((n, d)).astype("float32")
if "--corr" in sys.argv:  # attributes partly determined by the embedding
    A = X[:, :2] + 0.5 * rng.standard_normal((n, 2))
    A = (A - A.min(0)) / (A.max(0) - A.min(0))
else:
    A = rng.random((n, 2))
Q = rng.standard_normal((nq, d)).astype("float32")

t = time.time(); idx = HilbertHNSW(X, A, p=10, M=16, ef_construction=100)
print(f"build {time.time()-t:.1f}s  ({'correlated' if '--corr' in sys.argv else 'independent'} attrs)")

def recall(found, truth):
    return len(set(found.tolist()) & set(truth.tolist())) / max(len(truth), 1)

variants = {  # name: (use_hilbert_seeds, two_hop)
    "beam": (False, False), "beam+H": (True, False),
    "2hop": (False, True), "2hop+H": (True, True)}
print("recall@10 / distance computations per query  (scan cost = |valid|, recall 1.0)")
print(f"{'sel':>6} {'post':>6} " + " ".join(f"{v:>15}" for v in variants) + f" {'scan':>7}")
for sel in [0.005, 0.02, 0.05, 0.1, 0.3]:
    r = {v: [] for v in ["post", *variants]}; nd = {v: [] for v in [*variants, "scan"]}
    for qi in range(nq):
        side = np.sqrt(sel)
        c = rng.random(2) * (1 - side)
        lo, hi = c, c + side
        valid, _, _ = idx.matching(lo, hi)
        if len(valid) < k:
            continue
        truth = valid[np.argsort(idx._dist(Q[qi], valid))[:k]]
        # post-filter: plain HNSW top-ef, then filter
        idx.index.hnsw.efSearch = ef
        _, I = idx.index.search(Q[qi][None], ef)
        vs = set(valid.tolist())
        r["post"].append(recall(np.array([i for i in I[0] if i in vs][:k]), truth))
        nd["scan"].append(len(valid))
        for key, (use, th) in variants.items():
            ids, st = idx.search(Q[qi], lo, hi, k=k, ef=ef, scan_threshold=0,
                                 use_hilbert_seeds=use, two_hop=th, nav_seeds=4)
            r[key].append(recall(ids, truth)); nd[key].append(st["ndis"])
    print(f"{sel:>6.3f} {np.mean(r['post']):>6.3f} " +
          " ".join(f"{np.mean(r[v]):>6.3f}/{np.mean(nd[v]):>8.0f}" for v in variants) +
          f" {np.mean(nd['scan']):>7.0f}")
