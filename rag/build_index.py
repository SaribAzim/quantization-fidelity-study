"""
Phase 4 - Step 2: embed every passage once and save the result (the "index").

After this, searching is just: embed the query, take dot products with all
passage embeddings, return the highest scores. With a few tens of thousands
of passages that is a single matrix multiplication, so no special search
library is needed.

Run from the project root:   python rag/build_index.py
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedder import Embedder

CORPUS = "rag/data/corpus.jsonl"
INDEX = "rag/data/embeddings.pt"
# The extra core-topics library is embedded separately, so adding it never forces re-embedding the first one.
CORPUS_EXTRA = "rag/data/corpus_extra.jsonl"
INDEX_EXTRA = "rag/data/embeddings_extra.pt"

if __name__ == "__main__":
    if "--extra" in sys.argv:
        CORPUS, INDEX = CORPUS_EXTRA, INDEX_EXTRA
    passages = [json.loads(line) for line in open(CORPUS, encoding="utf-8")]
    print(f"Embedding {len(passages)} passages on the CPU...", flush=True)
    embedder = Embedder()
    start = time.time()
    vectors = embedder.encode([p["text"] for p in passages], batch_size=32)
    torch.save(vectors, INDEX)
    print(f"Saved {tuple(vectors.shape)} embeddings to {INDEX} in {time.time() - start:.0f}s")
