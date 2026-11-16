"""N-Quads -> one row per Product.  Usage: python 01_parse.py raw/ products.parquet
Assumes quads of one page are contiguous (WDC groups by page URL); flushes on URL change.
Sampling (env vars, defaults in config.sh): at most PER_DOMAIN_CAP random products per website,
then TARGET_N random products overall, both with fixed SEED -> identical output on every run."""
import gzip, json, re, sys, glob, os
from collections import defaultdict, Counter
from urllib.parse import urlparse
import pandas as pd

QUAD = re.compile(r'^(\S+)\s+<([^>]+)>\s+(.+?)\s+<([^>]*)>\s*\.\s*$')
LIT = re.compile(r'^"(.*)"(?:@([\w-]+)|\^\^<[^>]+>)?$')
TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

def short(p):  # http(s)://schema.org/name -> name
    return p.rsplit("/", 1)[-1]

def parse_obj(o):
    if o.startswith('"'):
        m = LIT.match(o)
        if not m: return ("lit", o, None)
        try: v = json.loads('"' + m.group(1) + '"')
        except Exception: v = m.group(1)
        return ("lit", v.strip(), m.group(2))
    return ("node", o.strip("<>"), None)

def first_lit(node, key):
    for kind, v, lang in node.get(key, []):
        if kind == "lit" and v: return v, lang
    return None, None

def to_price(s):
    if s is None: return None
    s = re.sub(r"[^\d.,]", "", s).replace(",", "")
    try: return float(s)
    except ValueError: return None

SPEC_KEYS = ("model", "color", "material", "size")
GTIN_KEYS = ("gtin", "gtin13", "gtin12", "gtin14", "gtin8")

def collect_specs(n, nodes, cap=300):
    parts = []
    for k in SPEC_KEYS:
        v, _ = first_lit(n, k)
        if v is None:  # may link to a node with a name (e.g. model -> ProductModel)
            for kind, ref, _ in n.get(k, []):
                if kind == "node" and ref in nodes:
                    v, _ = first_lit(nodes[ref], "name")
                    if v: break
        if v: parts.append(f"{k}: {v}")
    for kind, ref, _ in n.get("additionalProperty", []):  # PropertyValue nodes = spec table
        if kind == "node" and ref in nodes:
            pn, _ = first_lit(nodes[ref], "name")
            pv, _ = first_lit(nodes[ref], "value")
            if pn and pv: parts.append(f"{pn}: {pv}")
    return "; ".join(dict.fromkeys(parts))[:cap] or None

def get_gtin(n):
    for k in GTIN_KEYS:
        v, _ = first_lit(n, k)
        if v:
            d = re.sub(r"\D", "", v)
            if 8 <= len(d) <= 14 and d.strip("0"): return d.zfill(14)  # normalize to GTIN-14
    return None

def flush(url, nodes, out):
    for nid, n in nodes.items():
        if not any(k == "node" and short(v) == "Product" for k, v, _ in n.get("type", [])):
            continue
        name, lang = first_lit(n, "name")
        if not name: continue
        desc, _ = first_lit(n, "description")
        cat, _ = first_lit(n, "category")
        brand, _ = first_lit(n, "brand")
        if brand is None:  # brand -> Brand/Organization node
            for k, v, _ in n.get("brand", []):
                if k == "node" and v in nodes: brand, _ = first_lit(nodes[v], "name")
        price = cur = None
        for k, v, _ in n.get("offers", []):
            if k == "node" and v in nodes:
                o = nodes[v]
                price = to_price(first_lit(o, "price")[0] or first_lit(o, "lowPrice")[0])
                cur = first_lit(o, "priceCurrency")[0]
                if price is not None: break
        out.append(dict(url=url, name=name, lang=lang, description=desc,
                        category_raw=cat, brand=brand, price=price, currency=cur,
                        specs=collect_specs(n, nodes), gtin=get_gtin(n)))

import py3langid
py3langid.set_languages(["en", "fr", "de", "es", "it", "pt", "nl", "pl", "cs", "ru", "uk", "sv", "da", "no",
                         "fi", "tr", "ja", "zh", "ko", "ar", "id", "vi", "ro", "hu", "el", "he", "th"])

def is_english(lang, name, desc):
    """Most WDC pages carry no language tag (~80% in a real chunk), so detect it from the text.
    Short ASCII-only texts (e.g. 'Nike Air Max 90') are too short for language ID: keep them."""
    if lang: return lang.lower().startswith("en")
    text = (name + ". " + (desc or "")[:300]).strip()
    if len(text) < 40: return text.isascii()
    return py3langid.classify(text)[0] == "en"

def domain_of(url):
    d = urlparse(url).netloc.lower().split(":")[0]
    return d[4:] if d.startswith("www.") else d

SEED = int(os.environ.get("SEED", 532))
PER_DOMAIN_CAP = int(os.environ.get("PER_DOMAIN_CAP", 2000))
TARGET_N = int(os.environ.get("TARGET_N", 500000))

def main(src, dst):
    rows, nodes, cur_url = [], defaultdict(lambda: defaultdict(list)), None
    per_dom = Counter()
    pre_cap = 3 * PER_DOMAIN_CAP   # memory bound while parsing; slack for later dedup/filters

    def emit(url, nodes):
        d = domain_of(url)
        if per_dom[d] >= pre_cap: return
        before = len(rows)
        flush(url, nodes, rows)
        for r in rows[before:]: r["domain"] = d
        per_dom[d] += len(rows) - before

    for fp in sorted(glob.glob(f"{src}/*")):
        op = gzip.open if fp.endswith(".gz") else open
        try:
            with op(fp, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = QUAD.match(line)
                    if not m: continue
                    s, p, o, g = m.groups()
                    if g != cur_url:
                        if cur_url: emit(cur_url, nodes)
                        nodes, cur_url = defaultdict(lambda: defaultdict(list)), g
                    key = "type" if p == TYPE else short(p)
                    nodes[s][key].append(parse_obj(o))
        except (EOFError, OSError) as e:   # truncated download: keep what was read, warn
            print(f"WARNING: {fp} truncated/corrupt ({e}); re-run 00_download.sh to resume it", flush=True)
        print(fp, f"{len(rows):,} rows so far", flush=True)
    if cur_url: emit(cur_url, nodes)
    if not rows: sys.exit("no products parsed; check the raw/ files")

    df = pd.DataFrame(rows)
    print(f"{len(df):,} raw products from {df.domain.nunique():,} domains")
    df = df.drop_duplicates(["url", "name"])                       # microdata + json-ld dupes
    n0 = len(df)
    df = df[[is_english(l, n, d) for l, n, d in zip(df.lang, df.name, df.description)]]
    print(f"language filter kept {len(df):,} of {n0:,} (English)")
    n0 = len(df)
    df = df[df.gtin.isna() | ~df.gtin.duplicated()]                 # same product sold by several stores
    print(f"gtin dedup removed {n0 - len(df):,} cross-store duplicates")

    # sampling: random <= PER_DOMAIN_CAP per website, then random TARGET_N overall (fixed seed)
    df = df.sample(frac=1, random_state=SEED).groupby("domain", sort=False).head(PER_DOMAIN_CAP)
    print(f"{len(df):,} after per-domain cap of {PER_DOMAIN_CAP:,}")
    if len(df) > TARGET_N:
        df = df.sample(n=TARGET_N, random_state=SEED)
    else:
        print(f"NOTE: only {len(df):,} products available (< TARGET_N={TARGET_N:,}); raise N_CHUNKS for more")
    df = df.sort_values(["domain", "url"]).reset_index(drop=True); df["id"] = df.index
    print(f"{df.domain.nunique():,} domains; largest domain share {df.domain.value_counts(normalize=True).iloc[0]:.2%}")
    df.to_parquet(dst)
    print(f"{len(df):,} products")
    print(df[["description", "category_raw", "brand", "price", "specs", "gtin"]].notna().mean().round(3))  # field coverage

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
