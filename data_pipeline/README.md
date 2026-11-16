# WDC Product → filtered-ANN dataset

Builds the benchmark dataset for the project: product **vectors** for HNSW plus **filter
attributes** (category, price, brand) from the
[Web Data Commons Schema.org Product corpus](https://webdatacommons.org/structureddata/2024-12/stats/schema_org_subsets.html)
(Oct 2024 crawl: 280M product pages, 3.3M websites, 243 GB of N-Quads).

**Output** (in `$WDC_DIR/dataset/`, default `~/cse532_wdc/dataset/`):

| file | content |
|---|---|
| `vectors.npy` | N × 768 float32, L2-normalized (cosine = dot product). Row *i* is product `id` *i* |
| `attrs.parquet` | `id`, `cat_id` (int16), `price_usd` (float32, NaN = missing or non-USD), `brand_id` (int32, -1 = none) |
| `meta.json` | embedding model, dimension, row count, taxonomy (`cat_id` → name) |

Defaults give N ≈ 500k products from ~20k websites.

## How to run (SLURM cluster, `defq` partition)

Everything runs as **you** (`$USER`, `$HOME`); nothing is hardcoded to one account.

```bash
git clone https://github.com/ArnavDesai235/cse532.git
cd cse532/data_pipeline

# 1. One-time setup on the LOGIN node (needs internet, ~15-30 min)
bash setup.sh
#    creates conda env `cse532_wdc`, installs Ollama to ~/ollama if missing,
#    downloads the two models, downloads 10 random WDC chunks (~1.5 GB)

# 2. Parse + propose taxonomy (two chained jobs: CPU, then GPU)
bash submit.sh
squeue -u $USER                      # wait for both to finish (~1-3 h total)

# 3. Review the category list the LLM proposed (see "Reviewing the taxonomy")
nano ~/cse532_wdc/taxonomy.json

# 4. Classify + embed (GPU job, ~3-8 h)
bash submit.sh pipeline
```

Logs: `data_pipeline/logs/<job-name>_<jobid>.out`; Ollama server logs: `$WDC_DIR/logs/`.

## What each step does

| step | file | where | does |
|---|---|---|---|
| setup | `setup.sh` | login node | conda env from `environment.yml`, Ollama binary, models, then calls `00_download.sh` |
| 0 | `00_download.sh` | login node | picks `N_CHUNKS` random chunk files (fixed `SEED`) from WDC's lookup table → `chunks.txt`, downloads them to `raw/` (resumable) |
| 1 | `01_parse.py` via `slurm/parse.sbatch` | CPU job | N-Quads → one row per product: name, description, specs (model/color/material/size + spec table), raw category, brand, price, currency, GTIN, domain. Dedups (same page twice; same GTIN across stores), keeps English, keeps ≤ `PER_DOMAIN_CAP` random products per website, then `TARGET_N` random overall → `products.parquet`. Prints field coverage |
| 2a | `02_categories.py` (`PROPOSE_ONLY=1`) via `slurm/propose.sbatch` | GPU job | LLM reads 3,000 sampled raw category strings, proposes categories batch by batch, merges them into 20–40 classes + `Other` → `taxonomy.json` (frozen) |
| 2b | `02_categories.py` via `slurm/pipeline.sbatch` | GPU job | maps each **unique** raw category string to a taxonomy class (1 LLM call per unique string), then classifies products with no category from name + description → `products_cat.parquet` with `cat_id` |
| 3 | `03_embed.py` via `slurm/pipeline.sbatch` | same GPU job | embeds `name. specs. description[:400]` with `nomic-embed-text` → `dataset/` |

Design choices worth keeping in the report:
- **Random chunks + per-domain cap**: WDC chunks group all products of a website together, so
  hand-picking chunks samples *websites*, and one big store can dominate. Random chunks give
  ~20k websites; the cap stops any single one from exceeding 2,000 products.
- **Category, price, and brand are not in the embedded text**, so vectors and filter attributes
  stay independent. Otherwise "nearest neighbors within category X" would be trivially easy.
- **GTIN dedup**: the same product sold by several stores would give near-duplicate vectors
  and inflate recall.
- **Only USD prices** are kept (no currency conversion); others are NaN.

## Reviewing the taxonomy (step 3)

`taxonomy.json` is a JSON list of category names; the last entry must stay `"Other"`.
A 7B model sometimes produces overlapping or uneven classes. Before running step 4:
- merge overlaps (e.g. "Kitchen" and "Home Appliances"),
- drop entries that are too vague, aiming for 20–40 classes,
- keep the order stable after you submit step 4, because `cat_id` is the list index.

To have the LLM propose a fresh list, delete `taxonomy.json` and run
`sbatch slurm/propose.sbatch` from `data_pipeline/`.
Editing the file after step 4 started invalidates the classification cache (it is detected and
redone automatically).

## Configuration

All settings are in `config.sh` and can be overridden per run, e.g. `TARGET_N=200000 bash submit.sh`.

| variable | default | meaning |
|---|---|---|
| `WDC_DIR` | `~/cse532_wdc` | data and outputs (needs ~10 GB) |
| `CONDA_ENV` | `cse532_wdc` | env created by `setup.sh` |
| `OLLAMA_BIN` | `~/ollama/bin/ollama` | reused if it already exists |
| `OLLAMA_MODELS` | `~/.ollama/models` | model weights (~5.5 GB) |
| `LLM_MODEL` / `EMBED_MODEL` | `qwen2.5:7b` / `nomic-embed-text` | |
| `N_CHUNKS` / `SEED` | `10` / `532` | random chunk sample; same seed = same data |
| `PER_DOMAIN_CAP` / `TARGET_N` | `2000` / `500000` | sampling limits in step 1 |

The SLURM resource requests (`-p defq`, 1 GPU, time limits) are in `slurm/*.sbatch`. If the
partition name differs, edit the `#SBATCH -p` lines or submit with `sbatch -p <name>`.

## Resuming and troubleshooting

- **Job hit the time limit**: run `bash submit.sh pipeline` again. Step 2b resumes from
  `category_cache.json` (saved every 2,000 labels); step 3 resumes from `dataset/progress.json`.
- **`No chunk names found in Product_lookup.csv`**: WDC changed the lookup format. Check
  `head ~/cse532_wdc/Product_lookup.csv` and adjust the `grep` pattern in `00_download.sh`.
- **`Model ... missing; pulling` inside a job fails**: compute nodes have no internet. Run
  `bash setup.sh` again on the login node (it skips finished steps).
- **`ollama did not start`**: see `$WDC_DIR/logs/ollama_<jobid>.log`. Each job uses its own port
  (20000 + job id % 20000), so it doesn't conflict with other Ollama servers on the node.
- **`conda: command not found` in a job**: `config.sh` sources `~/.bashrc` and tries
  `module load anaconda3`/`miniconda3`. If your cluster names the module differently, add it to
  `activate_env` in `config.sh`.
- **Fewer than `TARGET_N` products**: step 1 prints a note. Raise `N_CHUNKS`, delete
  `$WDC_DIR/chunks.txt`, then re-run `bash 00_download.sh` and `bash submit.sh`.
- **Start over completely**: `rm -rf ~/cse532_wdc` (keeps the conda env, Ollama, and models).

## Checking the result

```python
import numpy as np, pandas as pd, json
v = np.load("dataset/vectors.npy", mmap_mode="r"); a = pd.read_parquet("dataset/attrs.parquet")
meta = json.load(open("dataset/meta.json"))
print(v.shape, len(a))                                              # same N
print(a.cat_id.map(dict(enumerate(meta["taxonomy"]))).value_counts(normalize=True))  # selectivity per category
```

Known limitations:
- The parser assumes WDC writes all quads of one page consecutively. This is expected but not
  verified on the real files. If it's wrong, products get split, and the parse log will show
  many products with no name or price.
- The pipeline was tested end to end only on synthetic data with a mocked Ollama. The first
  real run is the validation, so check the field-coverage printout in the parse log.
