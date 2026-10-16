"""
Finds the passages most relevant to a query: the "R" in RAG.

Search = one matrix multiplication. Each passage embedding and the query
embedding have length 1, so their dot product is their similarity (1.0 =
same meaning, near 0 = unrelated). We keep the top-k scores.

The library comes in up to two files (the category pass and the extra
core-topics pass); both are searched together as one library.
"""

import json
import os

import torch

LIBRARY_FILES = [
    ("rag/data/corpus.jsonl", "rag/data/embeddings.pt"),
    ("rag/data/corpus_extra.jsonl", "rag/data/embeddings_extra.pt"),
]


class Retriever:
    def __init__(self, library_files=LIBRARY_FILES):
        self.passages, vectors = [], []
        for corpus_path, index_path in library_files:
            if not (os.path.exists(corpus_path) and os.path.exists(index_path)):
                continue  # the extra file is optional
            passages = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]
            part = torch.load(index_path)
            assert len(passages) == part.size(0), f"{corpus_path} and its index are out of sync, rebuild the index"
            self.passages += passages
            vectors.append(part)
        assert self.passages, "no library found, run rag/build_corpus.py and rag/build_index.py"
        self.vectors = torch.cat(vectors)

    def search(self, query_vectors, k=3):
        """query_vectors: (n, 384). Returns, for each query, a list of k (passage, similarity) pairs."""
        scores = query_vectors @ self.vectors.T  # (n, number of passages)
        top_scores, top_ids = scores.topk(k, dim=-1)
        return [[(self.passages[i], s) for i, s in zip(ids.tolist(), sc.tolist())]
                for ids, sc in zip(top_ids, top_scores)]
