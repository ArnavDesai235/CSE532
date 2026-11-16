#!/bin/bash
# Downloads a random, reproducible subset of the WDC Schema.org Product corpus (Oct 2024).
# Run on the LOGIN node (compute nodes may not have internet). Called by setup.sh.
# Each chunk (90-140 MB) holds ALL products of a few thousand websites; we pick N_CHUNKS of 1,658 at random.
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
  # The directory listing has all 1,658 chunks; Product_lookup.csv only maps ~69 of them.
  wget -qO- "$BASE/" | grep -oE 'part_[0-9]+\.gz' | sort -u > all_chunks.txt || true
  [ -s all_chunks.txt ] || cut -d, -f3 Product_lookup.csv | grep -E '^part_[0-9]+\.gz$' | sort -u > all_chunks.txt
  [ -s all_chunks.txt ] || { echo "Could not list chunk files from $BASE/"; exit 1; }
  # Seeded, portable shuffle: order chunks by md5(SEED + name), take the first N.
  while read -r f; do printf '%s %s\n' "$(printf '%s%s' "$SEED" "$f" | md5sum | cut -c1-16)" "$f"; done < all_chunks.txt \
    | sort | head -n "$N_CHUNKS" | cut -d' ' -f2 > chunks.txt
  echo "Picked $(wc -l < chunks.txt) of $(wc -l < all_chunks.txt) chunks (seed $SEED)"
fi
cat chunks.txt

while read -r f; do
  [ -z "$f" ] && continue
  echo "downloading $f"
  wget -q -c -P raw "$BASE/$f"     # -c resumes partial files
done < chunks.txt
du -sh raw
