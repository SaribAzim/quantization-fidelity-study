"""
Turns text into a list of 384 numbers (an "embedding") so we can search by
MEANING instead of by exact words.

How to think about it: an embedding model reads a piece of text and places it
as a point in a 384-dimensional space, so that texts with similar meaning land
close together. "How does quicksort pick a pivot?" and a paragraph explaining
quicksort end up near each other, even if they share few words.

Model: BAAI/bge-small-en-v1.5, a small free model (about 130 MB) that runs fine
on the CPU, which leaves the GPU's memory to the language model.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge was trained so that a short SEARCH QUERY gets this prefix, while passages
# (and HyDE's made-up passages) are embedded as they are.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL).to(device).eval()

    @torch.no_grad()
    def encode(self, texts, is_query=False, batch_size=32):
        """Returns a (len(texts), 384) tensor; every row has length 1 so a dot product is a similarity."""
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        out = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state
            vectors = hidden[:, 0]  # bge summarizes a text in its first ([CLS]) position
            out.append(F.normalize(vectors, dim=-1).cpu())
        return torch.cat(out)
