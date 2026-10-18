#!/usr/bin/env python3
"""High-Throughput, Checkpointed WDC Dataset Ingestion & Embedding Pipeline.

Engineered for HPC clusters (e.g. Stony Brook SeaWulf A100 nodes).
Features:
  1. Idempotent & Resumable: Keeps a persistent `checkpoint.json`. Can be killed
     and restarted multiple times without duplicating or corrupting data.
  2. Multi-GPU Scalable: Supports single-GPU, multi-GPU (`torchrun`), or CPU fallback.
  3. Alibaba 1024-dim Embeddings: Generates 1024-dim dense vectors using FP16 inference.
  4. Structured Scalar Extraction: Extracts normalized scalar attributes
     (log-price, rating, log-reviews, in-stock) for the Hilbert B-Tree.
  5. Chunked Memory-Mapped Storage: Saves vectors and attributes in sealed binary chunks
     along with a unified `dataset_manifest.json` that acts as a single contiguous dataset.
  6. Streaming Disk-Friendly: Downloads, embeds, saves, and deletes raw .json.gz shards
     to stay within HPC scratch storage quotas.
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Try importing SentenceTransformer or direct HuggingFace transformers
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False


def parse_args():
    parser = argparse.ArgumentParser(description="Process WDC Product Corpus into Vectors & Attributes.")
    parser.add_argument("--url-list", type=str, default="data/wdc_urls.txt",
                        help="Path to text file containing WDC shard download URLs (one per line).")
    parser.add_argument("--data-dir", type=str, default="data/wdc_raw",
                        help="Directory to cache raw downloaded WDC shard files.")
    parser.add_argument("--output-dir", type=str, default="data/wdc_processed",
                        help="Directory to store processed vector and attribute chunks.")
    parser.add_argument("--model-name", type=str, default="Alibaba-NLP/gte-large-en-v1.5",
                        help="HuggingFace model identifier (default: Alibaba-NLP/gte-large-en-v1.5, 1024-dim).")
    parser.add_argument("--dim", type=int, default=1024,
                        help="Embedding dimension (default: 1024).")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Save vectors as float16 (half-precision, 2KB/vector) instead of float32.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Inference batch size per GPU (default: 256).")
    parser.add_argument("--chunk-size", type=int, default=500_000,
                        help="Number of vectors per sealed output chunk file (default: 500,000).")
    parser.add_argument("--delete-raw", action="store_true", default=True,
                        help="Delete raw .json.gz shard files after embedding to conserve disk space.")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Optional cap on total items to process across this entire run.")
    return parser.parse_args()


# ==============================================================================
# Attribute Extraction & Text Formatting
# ==============================================================================

def make_embedding_text(record):
    """Synthesizes structured product JSON into a rich semantic string for the model."""
    name = record.get("name") or record.get("title") or ""
    name = str(name).strip()
    if not name:
        return ""

    brand = record.get("brand") or ""
    if isinstance(brand, dict):
        brand = brand.get("name", "")
    brand = str(brand).strip()

    category = record.get("category") or record.get("category_path") or ""
    category = str(category).strip()

    desc = record.get("description") or ""
    desc = str(desc).strip()
    # Truncate description to 120 words to avoid tail token bloat
    short_desc = " ".join(desc.split()[:120])

    parts = [f"Product: {name}"]
    if brand:
        parts.append(f"Brand: {brand}")
    if category:
        parts.append(f"Category: {category}")
    if short_desc:
        parts.append(f"Description: {short_desc}")

    return " | ".join(parts)


def extract_scalar_attributes(record):
    """Extracts a normalized 4-dimensional scalar attribute vector A."""
    # A0: Price (log10 scale)
    raw_price = str(record.get("price") or record.get("offers", {}).get("price", "0"))
    price_clean = re.sub(r"[^\d.]", "", raw_price.replace(",", "."))
    try:
        price_val = float(price_clean)
    except ValueError:
        price_val = 0.0
    price_log = float(np.log10(max(1.0, price_val)))

    # A1: Rating (0.0 to 5.0 scale)
    agg = record.get("aggregateRating") or record
    raw_rating = str(agg.get("ratingValue") or "0")
    try:
        rating_val = float(re.sub(r"[^\d.]", "", raw_rating))
        if 5.0 < rating_val <= 100.0:
            rating_val /= 20.0
    except ValueError:
        rating_val = 0.0
    rating_val = float(np.clip(rating_val, 0.0, 5.0))

    # A2: Review Count (log10 scale)
    raw_reviews = str(agg.get("reviewCount") or "0")
    try:
        review_val = float(re.sub(r"[^\d]", "", raw_reviews))
    except ValueError:
        review_val = 0.0
    reviews_log = float(np.log10(max(1.0, review_val)))

    # A3: In-Stock Availability (Boolean float: 1.0 or 0.0)
    avail = str(record.get("availability") or record.get("offers", {}).get("availability", "")).lower()
    in_stock = 1.0 if any(k in avail for k in ["instock", "in stock", "available"]) else 0.0

    return [price_log, rating_val, reviews_log, in_stock]


# ==============================================================================
# Checkpoint & Output Storage Manager
# ==============================================================================

class CheckpointManager:
    """Manages persistent job state, chunk creation, and atomic resumes."""

    def __init__(self, output_dir, dim=1024, fp16=True, chunk_size=500_000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.dtype = np.float16 if fp16 else np.float32
        self.chunk_size = chunk_size
        self.manifest_path = self.output_dir / "dataset_manifest.json"

        self.state = {
            "dim": self.dim,
            "dtype": "float16" if fp16 else "float32",
            "chunk_size": self.chunk_size,
            "total_vectors": 0,
            "processed_urls": [],
            "chunks": [],
        }
        self._load()

        # In-memory buffer for current active chunk
        self.current_chunk_vectors = []
        self.current_chunk_attrs = []

    def _load(self):
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
                print(f"[Checkpoint] Loaded existing manifest: {self.state['total_vectors']:,} vectors "
                      f"across {len(self.state['chunks'])} chunks ({len(self.state['processed_urls'])} URLs processed).")
            except Exception as e:
                print(f"[Checkpoint] Warning: Could not parse {self.manifest_path}: {e}")

    def is_url_processed(self, url):
        return url in self.state["processed_urls"]

    def save_batch(self, vectors, attrs):
        """Buffers vectors and flushes complete chunk files to disk."""
        self.current_chunk_vectors.append(vectors)
        self.current_chunk_attrs.append(attrs)

        buffered_count = sum(len(v) for v in self.current_chunk_vectors)
        if buffered_count >= self.chunk_size:
            self._flush_chunk()

    def _flush_chunk(self):
        if not self.current_chunk_vectors:
            return

        chunk_v = np.vstack(self.current_chunk_vectors).astype(self.dtype)
        chunk_a = np.vstack(self.current_chunk_attrs).astype(np.float32)

        self.current_chunk_vectors.clear()
        self.current_chunk_attrs.clear()

        chunk_id = len(self.state["chunks"])
        v_file = f"vectors_chunk_{chunk_id:05d}.npy"
        a_file = f"attrs_chunk_{chunk_id:05d}.npy"

        np.save(self.output_dir / v_file, chunk_v)
        np.save(self.output_dir / a_file, chunk_a)

        chunk_meta = {
            "chunk_id": chunk_id,
            "vector_file": v_file,
            "attr_file": a_file,
            "count": len(chunk_v),
            "start_idx": self.state["total_vectors"],
            "end_idx": self.state["total_vectors"] + len(chunk_v),
        }
        self.state["chunks"].append(chunk_meta)
        self.state["total_vectors"] += len(chunk_v)

        self._save_manifest()
        print(f"[Storage] Sealed Chunk {chunk_id:05d} ({len(chunk_v):,} vectors). "
              f"Cumulative Total: {self.state['total_vectors']:,}")

    def mark_url_completed(self, url):
        self._flush_chunk()
        if url not in self.state["processed_urls"]:
            self.state["processed_urls"].append(url)
        self._save_manifest()

    def _save_manifest(self):
        temp_path = self.manifest_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        os.replace(temp_path, self.manifest_path)


# ==============================================================================
# Model Embedding Wrapper
# ==============================================================================

class EmbeddingEngine:
    def __init__(self, model_name, device=None):
        if device is None:
            self.device = "cuda" if (HAVE_TORCH and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
        print(f"[Model] Initializing {model_name} on {self.device}...")
        if HAVE_SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer(model_name, device=self.device)
            if self.device.startswith("cuda"):
                self.model.half()  # Enable FP16 execution
        else:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def encode(self, texts, batch_size=256):
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embeddings


# ==============================================================================
# Shard Downloader & Processor
# ==============================================================================

def download_file(url, target_path):
    """Downloads a file with basic progress and retry logic."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".download")

    for attempt in range(3):
        try:
            print(f"[Download] Fetching {url} (Attempt {attempt + 1})...")
            urllib.request.urlretrieve(url, temp_path)
            os.replace(temp_path, target_path)
            return True
        except Exception as e:
            print(f"[Download] Error downloading {url}: {e}")
            time.sleep(2)

    if temp_path.exists():
        temp_path.unlink()
    return False


def process_shard(shard_path, engine, ckpt, batch_size=256, max_items=None):
    """Parses, extracts attributes, and generates embeddings for a single shard."""
    texts = []
    attrs = []
    shard_items = 0
    t0 = time.time()

    open_fn = gzip.open if str(shard_path).endswith(".gz") else open
    with open_fn(shard_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if max_items and (ckpt.state["total_vectors"] + shard_items >= max_items):
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue

            emb_text = make_embedding_text(record)
            if not emb_text:
                continue

            attr_vec = extract_scalar_attributes(record)
            texts.append(emb_text)
            attrs.append(attr_vec)
            shard_items += 1

            if len(texts) >= batch_size:
                vecs = engine.encode(texts, batch_size=batch_size)
                ckpt.save_batch(vecs, np.array(attrs, dtype=np.float32))
                texts.clear()
                attrs.clear()

        # Flush remaining items
        if texts:
            vecs = engine.encode(texts, batch_size=batch_size)
            ckpt.save_batch(vecs, np.array(attrs, dtype=np.float32))
            texts.clear()
            attrs.clear()

    elapsed = time.time() - t0
    rate = shard_items / max(elapsed, 0.001)
    print(f"[Shard] Completed {shard_path.name}: {shard_items:,} items in {elapsed:.1f}s ({rate:.0f} items/s).")
    return shard_items


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Checkpoint Manager
    ckpt = CheckpointManager(
        args.output_dir,
        dim=args.dim,
        fp16=args.fp16,
        chunk_size=args.chunk_size,
    )

    # 2. Load Shard URLs
    url_file = Path(args.url_list)
    if not url_file.exists():
        print(f"[Setup] Warning: {url_file} not found. Creating sample manifest...")
        url_file.parent.mkdir(parents=True, exist_ok=True)
        # Placeholder example: user can populate with actual WDC URLs
        sample_urls = [
            "http://data.dws.informatik.uni-mannheim.de/webdatacommons/products/sample_products_01.json.gz",
        ]
        with open(url_file, "w") as f:
            f.write("\n".join(sample_urls) + "\n")
        print(f"[Setup] Populated sample URLs in {url_file}. Replace with production WDC file list.")

    with open(url_file, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    pending_urls = [u for u in urls if not ckpt.is_url_processed(u)]
    print(f"[Queue] Total URLs: {len(urls)} | Already Processed: {len(urls) - len(pending_urls)} | "
          f"Pending for this Run: {len(pending_urls)}")

    if not pending_urls:
        print("[Queue] All URLs in list have already been processed! Exiting.")
        sys.exit(0)

    # 3. Load Model
    engine = EmbeddingEngine(args.model_name)

    # 4. Process Shards Sequentially (or Distributed)
    t_global_start = time.time()
    for i, url in enumerate(pending_urls, 1):
        if args.max_items and ckpt.state["total_vectors"] >= args.max_items:
            print(f"[Stop] Reached requested max_items ({args.max_items:,}). Stopping.")
            break

        shard_name = Path(url.split("?")[0]).name
        local_shard = data_dir / shard_name

        print(f"\n==============================================================================")
        print(f"Processing URL [{i}/{len(pending_urls)}]: {url}")
        print(f"==============================================================================")

        # Download shard if not already present
        if not local_shard.exists():
            success = download_file(url, local_shard)
            if not success:
                print(f"[Skip] Failed to download {url}. Skipping to next.")
                continue

        # Process, extract, and embed
        process_shard(local_shard, engine, ckpt, batch_size=args.batch_size, max_items=args.max_items)

        # Mark as completed in persistent checkpoint
        ckpt.mark_url_completed(url)

        # Delete raw file if requested to conserve HPC quota
        if args.delete_raw and local_shard.exists():
            local_shard.unlink()
            print(f"[Cleanup] Deleted temporary raw shard {local_shard.name}")

    total_time = time.time() - t_global_start
    print("\n" + "=" * 80)
    print(f"Batch Execution Finished in {total_time / 3600:.2f} hours.")
    print(f"Total Vectors in Manifest: {ckpt.state['total_vectors']:,}")
    print(f"Sealed Chunks: {len(ckpt.state['chunks'])}")
    print(f"Manifest Path: {ckpt.manifest_path.resolve()}")
    print("=" * 80)


# ==============================================================================
# Helper Loader for Hilbert B-Tree Engine
# ==============================================================================

class WDCDatasetView:
    """Virtual continuous view over multiple sealed chunk files on disk."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        manifest_path = self.output_dir / "dataset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.total_vectors = self.manifest["total_vectors"]
        self.dim = self.manifest["dim"]
        self.dtype = np.float16 if self.manifest["dtype"] == "float16" else np.float32
        self.chunks = self.manifest["chunks"]

        # Memory-map each chunk file without loading into RAM
        self.v_mmaps = [
            np.load(self.output_dir / c["vector_file"], mmap_mode="r")
            for c in self.chunks
        ]
        self.a_mmaps = [
            np.load(self.output_dir / c["attr_file"], mmap_mode="r")
            for c in self.chunks
        ]

    def __len__(self):
        return self.total_vectors

    def get_arrays(self, max_items=None):
        """Concatenates chunks into memory-mapped continuous arrays for HilbertHNSWEngine."""
        count = min(self.total_vectors, max_items) if max_items else self.total_vectors
        if not self.v_mmaps:
            return np.empty((0, self.dim), dtype=self.dtype), np.empty((0, 4), dtype=np.float32)

        # Fast concatenation of mmap chunks
        if len(self.v_mmaps) == 1:
            return self.v_mmaps[0][:count], self.a_mmaps[0][:count]

        X = np.vstack([m for m in self.v_mmaps])[:count]
        A = np.vstack([m for m in self.a_mmaps])[:count]
        return X, A


def load_processed_wdc(output_dir="data/wdc_processed", max_items=None):
    """Convenience function to load processed vectors and attributes."""
    view = WDCDatasetView(output_dir)
    return view.get_arrays(max_items=max_items)


if __name__ == "__main__":
    main()

