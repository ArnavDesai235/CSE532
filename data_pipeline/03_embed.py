"""Embed every product + write the final ANN dataset.
Usage: python 03_embed.py products_cat.parquet out_dir/
Outputs in out_dir/:
  vectors.npy    N x D float32, L2-normalized (D=768 for nomic-embed-text); row i <-> attrs.id == i
  attrs.parquet  id, cat_id, price_usd (NaN = missing/non-USD), brand_id (-1 = none)
  meta.json      model, taxonomy (cat_id -> name), row count, creation time
Checkpoints every ~25k rows (progress.json); if the job dies, resubmitting resumes."""
import sys, os, json, time
import numpy as np, pandas as pd, ollama

MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
BATCH = 256
df = pd.read_parquet(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
assert (df.id.values == np.arange(len(df))).all(), "ids must be 0..N-1 in row order"

# category / price / brand deliberately NOT embedded: keeps vectors independent of filter attributes
specs = df.specs.fillna("")
texts = ("search_document: " + df.name + ". " + specs.where(specs == "", specs + ". ")
         + df.description.fillna("").str[:400]).tolist()

def embed(batch, tries=5):
    for a in range(tries):
        try:
            e = np.asarray(ollama.embed(model=MODEL, input=batch).embeddings, dtype=np.float32)
            return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        except Exception as ex:
            print(f"embed retry {a + 1}: {ex}", flush=True); time.sleep(5 * (a + 1))
    raise RuntimeError("embedding failed repeatedly; check the ollama log")

vec_path, prog_path = f"{out}/vectors.npy", f"{out}/progress.json"
start = 0
if os.path.exists(vec_path) and os.path.exists(prog_path):
    p = json.load(open(prog_path))
    if p.get("n") == len(df) and p.get("model") == MODEL:
        start = p["done"]; vecs = np.load(vec_path, mmap_mode="r+")
        print(f"resuming at row {start:,}")
if start == 0:
    dim = embed(texts[:1]).shape[1]
    vecs = np.lib.format.open_memmap(vec_path, mode="w+", dtype=np.float32, shape=(len(df), dim))

t0 = time.time()
for i in range(start, len(texts), BATCH):
    e = embed(texts[i:i + BATCH])
    vecs[i:i + len(e)] = e
    done = i + len(e)
    if (done // BATCH) % 100 == 0 or done == len(texts):
        vecs.flush()
        json.dump({"n": len(df), "model": MODEL, "done": done}, open(prog_path, "w"))
        print(f"{done:,}/{len(texts):,}  {(done - start) / (time.time() - t0):.0f} rows/s", flush=True)
vecs.flush()

# filterable attributes: only USD prices are comparable; others -> NaN
attrs = pd.DataFrame({
    "id": df.id,
    "cat_id": df.cat_id,
    "price_usd": df.price.where(df.currency.str.upper().eq("USD")).astype("float32"),
    "brand_id": df.brand.str.lower().str.strip().astype("category").cat.codes.astype("int32"),  # -1 = none
})
attrs.to_parquet(f"{out}/attrs.parquet")
tax = json.load(open("taxonomy.json")) if os.path.exists("taxonomy.json") else None
json.dump({"embed_model": MODEL, "rows": len(df), "dim": int(vecs.shape[1]),
           "taxonomy": tax, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
          open(f"{out}/meta.json", "w"), indent=1)
print(attrs.describe())
print(f"price_usd coverage {attrs.price_usd.notna().mean():.1%}, brand coverage {(attrs.brand_id >= 0).mean():.1%}")
