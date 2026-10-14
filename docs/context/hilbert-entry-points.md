# Hilbert-curve entry points for filtered HNSW — prototype findings

Code: `prototype/hilbert_hnsw.py`, benchmark: `prototype/bench.py`
Run: `pip install faiss-cpu hilbertcurve numpy && python prototype/bench.py [--corr]`
Date: 2026-10-14

## How the integration works
1. **Range decomposition.** Point IDs are sorted by Hilbert key over the quantized
   attribute dimensions (p = 10 bits per dimension). A box predicate is decomposed
   into aligned dyadic cubes. Every aligned cube is one contiguous Hilbert key interval,
   so each interval maps to a slice of the sorted ID array by binary search. Cubes still
   only partially inside the box at the max depth are included, and an exact attribute
   check refines them. This one pass yields:
   - the selectivity estimate (sum of slice lengths),
   - the scan path (brute force over the slices),
   - the validity bitmap for graph search.
2. **Seeding.** Sample up to 512 valid IDs from the slices, keep the 8 closest to the
   query, and add 4 vector-near points from the normal (unfiltered) HNSW descent.
3. **Restricted layer-0 search (ACORN-style two-hop).** Distances are computed only for
   valid nodes. When a neighbor is invalid, its own neighbors are expanded instead.
   The HNSW graph itself is unchanged; it is built with FAISS `IndexHNSWFlat`, and the
   layer-0 search is our own code over FAISS's neighbor arrays.

## Results
Synthetic data: 50k vectors, d = 32, 2 uniform attributes, 50 queries, k = 10, ef = 64.
Cells show recall@10 / distance computations per query. Scan cost = |valid| at recall 1.0.

| sel | post-filter | beam (any node) | 2-hop, no H seeds | 2-hop + H seeds | scan |
|---|---|---|---|---|---|
| 0.5% | 0.05 | 1.00 / 42.5k | 0.03 / 36 | 0.95 / 401 | 253 |
| 2% | 0.12 | 1.00 / 25.3k | 0.06 / 58 | 0.97 / 978 | 1,019 |
| 5% | 0.31 | 1.00 / 15.1k | 0.12 / 154 | 0.95 / 1,564 | 2,514 |
| 10% | 0.60 | 0.99 / 9.6k | 0.39 / 804 | 0.97 / 2,465 | 5,027 |
| 30% | 0.80 | 0.95 / 4.1k | 0.61 / 3,011 | 0.98 / 5,348 | 15,052 |

With attributes correlated to the vectors (`--corr`) the pattern is the same: 2-hop + H
seeds reaches 0.93–0.98 recall, and the scan is cheaper only at 0.5%.

## Takeaways
- **Seeding an unrestricted search is useless.** A beam search that may traverse any
  node already reaches valid points, but at 0.5% selectivity it computes 42k of 50k
  distances: a near-full scan with extra steps. Hilbert seeds change nothing there.
- **Restricted traversal is cheap but strands the search.** Without valid entry points,
  the two-hop search's recall collapses (0.03 at 0.5%).
- **Hilbert seeds fix that.** Recall rises from 0.03 to 0.95 at about 1/100th the distance
  computations of the unrestricted search. This is the mechanism to feature in the paper.
- **Routing crossover is about 2% selectivity.** Below it, the scan is cheaper (253 vs 401
  distances). Above it, two-hop + H seeds beats the scan by 1.6–2.8×.

## Caveats
- The "2-hop, no H seeds" baseline is weak. Its only seeds are the vector-near points,
  which are rarely valid. A real ACORN baseline descends through filtered upper layers
  and will do better. Compare against FAISS's ACORN index before claiming a win.
- Seed sampling costs up to 512 distances, about half the total at low selectivity.
  Next optimization: precomputed representative points per Hilbert cell instead of a
  random pool.
- Pure Python on synthetic data at 50k points. Distance counts are meaningful;
  wall-clock time is not yet.

## Next steps
- Add FAISS ACORN as a baseline.
- Replace random seed pools with per-cell representatives.
- Replace the fixed scan threshold with a cost model calibrated on the measured crossover.
- Sweep attribute dimensionality (1/2/4/8) and scale beyond 50k points.
