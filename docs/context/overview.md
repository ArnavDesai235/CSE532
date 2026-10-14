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

## Deliverable
| Deliverable | Weight | Due |
|---|---|---|
| 6-page, two-column workshop-style technical report | 30% of grade | TBD |
