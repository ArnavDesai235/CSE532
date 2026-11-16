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
Organize data along a space-filling curve over the attribute dimensions. An orchestrator
estimates predicate selectivity by intersecting the filter's range with the curve's ranges.

- Very selective queries (e.g. <1% of the data): scan the matching curve segments directly.
- Otherwise: run standard HNSW traversal.

## Goals
- Return the full k results with high recall under filtered queries across all selectivity
  levels, including highly selective predicates where post-filtering fails.
- Estimate selectivity cheaply before execution using the space-filling curve layout.
- Route each query automatically to curve-segment scanning or graph traversal based on
  that estimate.
- Evaluate recall@k and query throughput against pre-filtering and post-filtering baselines.

## Scope
**In scope**
- Filtered k-NN queries over vectors with categorical and numeric attributes
- A space-filling-curve data layout and selectivity estimator
- An orchestrator that picks the execution plan
- An HNSW-based prototype
- Benchmarks sweeping predicate selectivity

**Out of scope**
- Distributed or disk-resident indexes
- Online updates and deletes
- Production-grade engine integration

## Dataset
Real-world benchmark data comes from the Web Data Commons Schema.org Product corpus
(Oct 2024). Pipeline and run instructions: `data_pipeline/README.md`.
- Sample: 10 random corpus chunks (seed 532), at most 2,000 products per website,
  500k products total.
- Vectors: `nomic-embed-text` (768-d) over name + specs + description.
- Filter attributes: `cat_id` (LLM-proposed, human-reviewed taxonomy of 20–40 classes,
  assigned by `qwen2.5:7b` through Ollama), `price_usd`, and `brand_id`. They are not
  embedded, so vectors and filters stay independent.
- Runs as SLURM jobs (`defq`); Lawrence runs it.
- Before benchmarking: the prototype assumes uniform numeric attributes in [0,1]. `cat_id`
  is categorical (risk 3), `price_usd` is skewed and has NaNs (needs a rank transform and a
  missing-value bucket), and `brand_id` is high-cardinality (better as a post-check than a
  curve dimension).

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

## Evaluation plan additions
- Baselines beyond pre-filtering/post-filtering: post-filtering with adaptive `ef`, ACORN;
  SeRF or iRangeGraph if numeric ranges are central.
- Report selectivity-estimator accuracy (q-error) as its own result.
- Sweep attribute dimensionality as well as selectivity.
