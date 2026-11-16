#!/bin/bash
# Downloads a random, reproducible subset of the WDC Schema.org Product corpus (Oct 2024).
# Run on the LOGIN node (compute nodes may not have internet). Called by setup.sh.
# Each chunk holds ALL products of ~2,000 websites; we pick N_CHUNKS of 1,658 at random.
# Re-running keeps the same chunks.txt and skips finished downloads; delete chunks.txt to re-sample.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
BASE=https://data.dws.informatik.uni-mannheim.de/structureddata/2024-12/quads/classspecific/Product
mkdir -p "$WDC_DIR/raw" && cd "$WDC_DIR"

wget -q -nc "$BASE/Product_sample.txt"
wget -q -nc "$BASE/Product_lookup.csv"          # domain -> chunk file
wget -q -nc "$BASE/Product_domain_stats.csv"    # per-domain counts

if [ ! -f chunks.txt ]; then
  grep -oE '[A-Za-z0-9_.-]+\.(gz|nq|txt)' Product_lookup.csv | grep -v '^Product_' | sort -u > all_chunks.txt
  [ -s all_chunks.txt ] || { echo "No chunk names found in Product_lookup.csv; check: head Product_lookup.csv"; exit 1; }
  shuf -n "$N_CHUNKS" --random-source=<(openssl enc -aes-256-ctr -pass pass:"$SEED" -nosalt </dev/zero 2>/dev/null) \
    all_chunks.txt > chunks.txt
  echo "Picked $(wc -l < chunks.txt) of $(wc -l < all_chunks.txt) chunks (seed $SEED)"
fi
cat chunks.txt

while read -r f; do
  [ -z "$f" ] && continue
  echo "downloading $f"
  wget -q -c -P raw "$BASE/$f"     # -c resumes partial files
done < chunks.txt
du -sh raw
