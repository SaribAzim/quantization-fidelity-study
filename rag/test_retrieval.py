"""
Sanity test for the library and search: well-known questions must find the
article that obviously answers them. Not a benchmark, just a check that the
embedder, index and retriever are wired together correctly.

Run from the project root:   python rag/test_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedder import Embedder
from rag.retriever import Retriever

# (query, a word that should appear in the title of one of the top-3 passages)
CASES = [
    ("How does binary search find an element in a sorted array?", "binary search"),
    ("What is a hash table collision?", "hash"),
    ("Explain how public key cryptography works", "public-key"),
    ("What does the TCP three-way handshake do?", "transmission control"),
    ("How does gradient descent minimize a loss function?", "gradient descent"),
    ("What is a buffer overflow vulnerability?", "buffer overflow"),
]

if __name__ == "__main__":
    retriever, embedder = Retriever(), Embedder()
    print(f"{len(retriever.passages)} passages, {tuple(retriever.vectors.shape)} embeddings")

    # Search as a short QUESTION (with the bge query prefix) ...
    hits = retriever.search(embedder.encode([q for q, _ in CASES], is_query=True), k=3)
    found = 0
    for (query, expect), result in zip(CASES, hits):
        titles = [p["title"] for p, _ in result]
        ok = any(expect in t.lower() for t in titles)
        found += ok
        print(f"{'OK  ' if ok else 'MISS'} {query!r} -> {titles}  (top score {result[0][1]:.2f})")
    assert found >= 5, f"only {found}/{len(CASES)} queries found their article, something is wrong"

    # ...and as a made-up HyDE-style PASSAGE (no prefix), which should land in the same place.
    passage = ("Binary search is an algorithm that repeatedly halves a sorted list: it compares the target "
               "with the middle element and discards the half that cannot contain it, so it runs in logarithmic time.")
    top = retriever.search(embedder.encode([passage], is_query=False), k=3)[0]
    print("HyDE-style passage ->", [p["title"] for p, _ in top])
    assert any("binary search" in p["title"].lower() for p, _ in top)
    print(f"\nRetrieval test passed ({found}/{len(CASES)} questions found their article).")
