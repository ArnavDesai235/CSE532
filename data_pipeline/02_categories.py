"""Phase A: LLM proposes a taxonomy from the data (once; frozen to taxonomy.json).
Phase B: map products to it. LLM is called once per UNIQUE raw category,
plus once per product that has no category at all.
Usage: python 02_categories.py products.parquet products_cat.parquet
       PROPOSE_ONLY=1 python 02_categories.py ...   -> write taxonomy.json and stop (to review/edit it)"""
import sys, os, json, time, hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, ollama

MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
WORKERS = int(os.environ.get("OLLAMA_NUM_PARALLEL", 8))   # match the server's parallel slots
OPTS = {"temperature": 0, "num_ctx": 8192}
TAX_FILE = "taxonomy.json"
N_MIN, N_MAX = 20, 40
LIST_SCHEMA = {"type": "object", "properties": {"categories": {"type": "array", "items": {"type": "string"}}},
               "required": ["categories"]}

def ask_json(prompt, schema):
    r = ollama.chat(model=MODEL, format=schema, options=OPTS,
                    messages=[{"role": "user", "content": prompt}])
    return json.loads(r.message.content)

def build_pool(df, sample=3000):
    """Half raw category labels (most common + random tail), half product names. Names matter
    because only ~6% of real WDC products carry a category; the rest are classified by name."""
    raw_counts = df.category_raw.dropna().str.strip().value_counts()
    raw_counts = raw_counts[raw_counts.index != ""]
    k = min(sample // 2, len(raw_counts))
    top = raw_counts.index[:k // 2].tolist()
    rest = raw_counts.index[k // 2:].to_series()
    cats = top + rest.sample(min(len(rest), k - len(top)), random_state=0).tolist()
    names = df.name.drop_duplicates()
    names = names.sample(min(len(names), sample - len(cats)), random_state=0).str[:80].tolist()
    print(f"taxonomy pool: {len(cats):,} raw category labels + {len(names):,} product names")
    pool = cats + names
    return pd.Series(pool).sample(frac=1, random_state=0).tolist()   # mix labels and names per batch

def propose_taxonomy(pool, batch=150):
    # A1: each batch of labels/names -> up to 15 proposed categories; count how often each is proposed
    proposals = Counter()
    for i in range(0, len(pool), batch):
        try:
            cats = ask_json("Below are e-commerce category labels and product names from many websites.\n"
                            "Propose up to 15 broad product categories (2-4 words each) that cover them.\n\n"
                            + "\n".join(pool[i:i + batch]), LIST_SCHEMA)["categories"]
            proposals.update(c.strip().title() for c in cats if c.strip())
        except Exception as e:
            print("propose batch failed:", e)
    print(f"{len(proposals)} distinct proposals")

    # A2: merge near-duplicates into one flat, mutually exclusive list
    cand = "\n".join(f"{c} ({n})" for c, n in proposals.most_common(200))
    final = ask_json("Candidate product categories, with how often each was proposed:\n\n" + cand +
                     f"\n\nMerge duplicates and near-synonyms into ONE flat list of {N_MIN}-{N_MAX} broad, "
                     "mutually exclusive categories for general e-commerce. Do not include 'Other'.",
                     LIST_SCHEMA)["categories"]
    final = list(dict.fromkeys(c.strip().title() for c in final if c.strip()))
    if not N_MIN <= len(final) <= N_MAX:
        print(f"WARNING: got {len(final)} categories; review {TAX_FILE}")
    return final + ["Other"]

def classify(text):
    try:
        r = ollama.chat(model=MODEL, format=SCHEMA, options={"temperature": 0},
            messages=[{"role": "user", "content":
                f"Pick the best category for this product.\nOptions: {', '.join(TAXONOMY)}\n\n{text[:600]}"}])
        c = json.loads(r.message.content)["category"]
        return c if c in TAXONOMY else "Other"
    except Exception:
        return "Other"

CACHE_FILE = "category_cache.json"   # checkpoint: survives job time-outs, resubmit resumes

def load_cache():
    tax_hash = hashlib.md5(json.dumps(TAXONOMY).encode()).hexdigest()
    if os.path.exists(CACHE_FILE):
        c = json.load(open(CACHE_FILE))
        if c.get("taxonomy") == tax_hash:
            print(f"resuming: {len(c['labels']):,} cached labels")
            return c
        print("taxonomy.json changed since last run; discarding category cache")
    return {"taxonomy": tax_hash, "labels": {}}

def save_cache(cache):
    tmp = CACHE_FILE + ".tmp"
    json.dump(cache, open(tmp, "w"))
    os.replace(tmp, CACHE_FILE)   # atomic: a kill mid-write never corrupts the cache

def run(keys, texts, cache, chunk=2000):
    labels = cache["labels"]
    todo = [(k, t) for k, t in zip(keys, texts) if k not in labels]
    print(f"{len(keys) - len(todo):,} cached, {len(todo):,} to classify", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            for (k, _), lab in zip(part, ex.map(classify, [t for _, t in part])):
                labels[k] = lab
            save_cache(cache)
            done = i + len(part)
            print(f"  {done:,}/{len(todo):,}  {done / (time.time() - t0):.1f}/s", flush=True)
    return [labels[k] for k in keys]

df = pd.read_parquet(sys.argv[1])

# Phase A: build taxonomy once, then reuse the frozen file (reproducible across reruns)
if os.path.exists(TAX_FILE):
    TAXONOMY = json.load(open(TAX_FILE))
else:
    TAXONOMY = propose_taxonomy(build_pool(df))
    json.dump(TAXONOMY, open(TAX_FILE, "w"), indent=1)
print(f"{len(TAXONOMY)} categories:", TAXONOMY)
if os.environ.get("PROPOSE_ONLY"):
    sys.exit(0)
SCHEMA = {"type": "object", "properties": {"category": {"type": "string", "enum": TAXONOMY}},
          "required": ["category"]}

# Phase B
# 1) unique raw categories -> taxonomy
cache = load_cache()
uniq = [c for c in df.category_raw.dropna().str.strip().unique().tolist() if c]
print(f"{len(uniq):,} unique raw categories")
cat_map = dict(zip(uniq, run(["raw::" + c for c in uniq], [f"Category: {c}" for c in uniq], cache)))
df["category"] = df.category_raw.str.strip().map(cat_map)

# 2) products with no category -> classify from name/description
miss = df.category.isna()
print(f"{miss.sum():,} products without category")
m = df.loc[miss]
df.loc[miss, "category"] = run(("prod::" + m.url + "::" + m.name).tolist(),
                               (m.name + ". " + m.description.fillna("").str[:300]).tolist(), cache)

df["cat_id"] = df.category.map({c: i for i, c in enumerate(TAXONOMY)}).astype("int16")
df.to_parquet(sys.argv[2])
print(df.category.value_counts(normalize=True).round(3))   # selectivity per category
