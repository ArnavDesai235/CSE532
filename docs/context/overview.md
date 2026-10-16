# CSE532 Project Overview

**Course:** CSE532 Database Systems (graduate), Stony Brook University, Fall 2026
**Instructor:** Prof. Shuai Mu
**Team:** Arnav Desai, Lawrence Caradonna

## Problem
Graph-based approximate nearest neighbor (ANN) indexes such as HNSW are fast for pure
vector similarity search. They break down when a query also filters on scalar
attributes (categorical or numeric, e.g. `x > 30`) that are unrelated to vector similarity.

- **Pre-filtering:** the graph has no attribute awareness, so the search falls back to
  near-exhaustive scanning.
- **Post-filtering:** the search finds the nearest candidates and then filters them. With a
  highly selective predicate, it can return fewer than k results even when valid matches
  exist elsewhere in the graph.

## Approach
1. **Hilbert-Partitioned B-Tree of Subgraphs:**
   - Map scalar attributes ($m \le 4$) to a 1D Hilbert space.
   - Organize data into a B-Tree structure where leaf nodes cover disjoint, non-overlapping Hilbert key intervals $[K_i, K_{i+1})$.
   - Each leaf maintains a dedicated, independent HNSW subgraph $G_i$ covering only vectors within that attribute range.
   - **Zero Filter Overhead on Enclosed Chunks:** When a query range fully encloses a leaf's interval, 100% of its vectors are valid; the query executes unconstrained, vanilla HNSW with zero filtering degradation.
   - Boundary leaves use within-chunk Hilbert-assisted traversal or flat scans.
2. **Temporal Merging Scheme:**
   - Incoming appends buffer temporally in an active in-memory tier (L0 / MemTable).
   - Once the temporal buffer fills, items are sorted by Hilbert key and flushed into their respective Hilbert leaf partitions.
   - **Localized Compaction:** When a leaf partition accumulates multiple temporal deltas or exceeds capacity, it splits along its median Hilbert key and rebuilds *only* its local subgraph. Cross-partition graph merging is never required, completely eliminating the intractable global HNSW merge bottleneck.
3. **Selectivity-Based Query Routing & Parallel Subgraph Search:**
   - An orchestrator estimates predicate selectivity by intersecting the query box with the Hilbert B-tree partitions.
   - Very selective queries (or small leaves): scan matching contiguous vectors directly via SIMD.
   - Moderate-to-broad queries: dispatch leaf partition searches concurrently in parallel across threads/cores.
   - **Shared-State Branch-and-Bound Early Exit:** A thread-safe global top-$k$ candidate tracker maintains dual thresholds (`worst_dist_k` for exact leaf-level hypersphere & flat-scan pruning, and `worst_dist_ef` for HNSW beam search waypoints). Leaves whose bounding hypersphere cannot beat the global $k$-th best distance abort before graph traversal. Active threads searching distant or unpromising boundary leaves abort early once closer candidates are discovered by peer threads, eliminating 30%–60% of distance computations while maintaining $\ge 98\%$ recall.

## Goals
- Return the full k results with high recall across all selectivity levels, including highly selective predicates (<1%) where post-filtering starves.
- Design and evaluate a Hilbert-partitioned B-Tree of subgraphs that eliminates cross-segment graph rewiring.
- Implement a temporal merging scheme that buffers appends and restricts graph rebuilds strictly to localized partition leaves.
- Formulate a cost-based optimizer to route dynamically between SIMD flat scans and partitioned subgraph traversals.
- Evaluate recall@k, latency, and index maintenance overhead against plain HNSW and ACORN-style filtered baselines.

## Scope
**In scope**
- Filtered k-NN queries over vectors with continuous numeric and categorical attributes ($m \le 4$)
- Hilbert-partitioned B-tree data layout with dedicated HNSW subgraphs
- Temporal append buffering and localized leaf compaction model
- Selectivity estimation via Hilbert range decomposition and B-tree boundary pruning
- Cost-based query router (SIMD scan vs. Hilbert-seeded HNSW traversal)
- Benchmarks sweeping predicate selectivity, attribute dimension, and data correlation

**Out of scope**
- Distributed clusters (single-node multi-threaded evaluation)
- Complex ACID transaction concurrency (focus on read-optimized lakehouse / append-mostly workloads)
- Arbitrary schema changes beyond defined attribute dimensions

## Deliverable
| Deliverable | Weight | Due |
|---|---|---|
| 6-page, two-column workshop-style technical report | 30% of grade | TBD |

## Related work
- **Qdrant query planner** ([blog](https://qdrant.tech/blog/pre-filtering-vs-post-filtering/)):
  already estimates filter cardinality per query and routes among filterable HNSW,
  HNSW + ACORN, payload-index scan, and full scan. Selectivity-based routing on its own
  is not novel; our contribution has to come from the space-filling curve.
- **ACORN** (SIGMOD 2024, [arXiv 2403.04871](https://arxiv.org/abs/2403.04871)):
  predicate-agnostic filtered HNSW traversal; targets the moderate-selectivity band.
- **Range-filtered graphs** (SeRF, iRangeGraph): purpose-built for numeric range predicates.
- **Experimental study** ([arXiv 2508.16263](https://arxiv.org/abs/2508.16263)): benchmarks
  10 filtered-ANN algorithms at 0.1%–100% selectivity. Its harness,
  [FANNBench](https://github.com/lmccccc/FANNBench), is a candidate for our benchmarks.

## Risks and open questions
1. **Mid-selectivity gap.** Plain HNSW with post-filtering degrades between ~1% and ~20%
   selectivity (it must visit ~k/s candidates). A two-way scan/HNSW switch does not fix
   this band. Options: a third, filter-aware graph path, or feeding the curve's matching
   segments as an ID bitmap into filtered traversal (ACORN-style).
2. **Attribute dimensionality.** Beyond ~3–4 attribute dimensions, box predicates break into
   many curve intervals, hurting scan locality and estimation cost. Hilbert degrades more
   slowly than Z-order. Sweep 1/2/4/8 dimensions to find where the approach stops helping.
3. **Categorical attributes.** They have no natural order, so their placement on the curve is
   arbitrary. Equality is fine; IN-lists and disjunctions fragment.
4. **Fixed attribute set.** Only attributes built into the curve can be estimated or scanned.
   State this as an explicit limitation.
5. **Routing threshold.** Replace the fixed 1% cutoff with a cost model: estimated count ×
   vector dimension vs. calibrated graph-traversal cost. Report the measured crossover.
6. **Partition skew & dynamic splitting.** Non-uniform attribute distributions cause certain
   Hilbert ranges to accumulate more data. Leaves must split dynamically along the median
   Hilbert key when capacity threshold $N_{\max}$ is reached.
7. **Scatter-gather fanout on broad queries.** Queries with broad selectivity (>30%) touch
   multiple partition subgraphs. Evaluate the crossover where searching multiple subgraphs
   becomes slower than a single unified graph.

## Evaluation plan additions
- Baselines beyond pre-filtering/post-filtering: post-filtering with adaptive `ef`, ACORN;
  SeRF or iRangeGraph if numeric ranges are central.
- Report selectivity-estimator accuracy (q-error) as its own result.
- Sweep attribute dimensionality as well as selectivity.
- Sweep leaf partition capacity ($N_{\max} \in [10k, 50k, 100k]$) to quantify the trade-off
  between chunk pruning granularity and multi-graph scatter-gather overhead.
- Benchmark ingestion throughput and compaction latency under temporal stream appends:
  measure time to merge temporal deltas and rebuild localized subgraphs vs. global rebuild.

