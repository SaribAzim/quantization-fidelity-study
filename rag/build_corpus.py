"""
Phase 4 - Step 1: build the knowledge library the RAG system will search.

We download the plain text of a few hundred computer-science Wikipedia
articles (free public API), cut each into overlapping passages of about 200
words ("chunks"), and save them. A passage is small enough to be a focused
piece of evidence and big enough to contain a full idea.

CONTAMINATION CHECK: if the library secretly contained the exam questions,
RAG would look brilliant for the wrong reason (copying answers). So we
remove any passage that shares an 8-word run with any of the 412 exam
questions or their answer options.

Run from the project root:   python rag/build_corpus.py
Output: rag/data/corpus.jsonl  (one passage per line; not committed to git)
"""

import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.mcq import load_questions

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "quantization-fidelity-study/1.0 (student research project)"}
OUT = "rag/data/corpus.jsonl"
OUT_EXTRA = "rag/data/corpus_extra.jsonl"
CACHE_DIR = "rag/data/raw"  # one text file per downloaded article, so a restart does not re-download

# Wikipedia categories that match the four exam subjects.
CATEGORIES = [
    "Algorithms", "Sorting algorithms", "Search algorithms", "Graph algorithms", "Data structures",
    "Computational complexity theory", "Theory of computation", "Programming language concepts",
    "Computer architecture", "Operating systems", "Computer networking", "Database management systems",
    "Software engineering", "Computer security", "Cryptography", "Computer security exploits",
    "Machine learning", "Machine learning algorithms", "Artificial neural networks", "Statistical classification",
    "Supervised learning", "Unsupervised learning", "Computer programming", "Boolean algebra",
    "Number systems", "Internet protocols", "Data compression", "Parallel computing",
]
PER_CATEGORY = 25
CHUNK_WORDS, OVERLAP_WORDS, MIN_WORDS = 200, 40, 60
STOP_SECTIONS = re.compile(r"\n== *(See also|References|External links|Further reading|Notes|Bibliography|Sources) *==.*", re.S)


MIN_SECONDS_BETWEEN_REQUESTS = 1.0  # Wikimedia asks API users to be gentle
_last_request = [0.0]


def get(params):
    """One polite API call: spaced out, and backs off when Wikipedia says 'too many requests' (HTTP 429)."""
    for attempt in range(8):
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.time() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.time()
        try:
            response = requests.get(API, headers=HEADERS, params={"format": "json", **params}, timeout=60)
            if response.status_code == 429:
                delay = int(response.headers.get("Retry-After", 0)) or min(120, 15 * (attempt + 1))
                print(f"  rate limited, waiting {delay}s", flush=True)
                time.sleep(delay)
                continue
            return response.json()
        except Exception as error:
            print(f"  request problem ({error}), retrying", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Wikipedia API did not answer")


def category_titles(category, limit):
    data = get({"action": "query", "generator": "categorymembers", "gcmtitle": f"Category:{category}",
                "gcmnamespace": 0, "gcmlimit": limit})
    return sorted(p["title"] for p in data.get("query", {}).get("pages", {}).values())


def fetch_text(title):
    """Full plain text of one article (the API hands out full text one page at a time). Cached on disk."""
    cache_file = os.path.join(CACHE_DIR, re.sub(r"[^A-Za-z0-9]+", "_", title)[:120] + ".txt")
    if os.path.exists(cache_file):
        return open(cache_file, encoding="utf-8").read()
    data = get({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "redirects": 1})
    text = ""
    for page in data.get("query", {}).get("pages", {}).values():
        text = page.get("extract", "")
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def chunk_text(title, text):
    """Overlapping ~200-word passages; each starts with the article title for context."""
    text = STOP_SECTIONS.sub("", "\n" + text)
    text = re.sub(r"==+ *(.*?) *==+", r"\1.", text)  # section headings -> plain sentences
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = []
    for start in range(0, max(len(words), 1), step):
        piece = words[start:start + CHUNK_WORDS]
        if len(piece) >= MIN_WORDS:
            chunks.append(f"{title}: " + " ".join(piece))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def ngrams(text, n=8):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


# A second pass with hand-picked core topics from a standard computer-science curriculum. The category pass
# above takes only the first 25 articles per category, which misses many textbook topics (e.g. TCP, gradient
# descent). These titles come from the curriculum, NOT from the exam questions, so nothing is tuned to the test.
CORE_TOPICS = """Hash table|Binary search tree|AVL tree|Red–black tree|B-tree|Heap (data structure)|Priority queue|Linked list|Stack (abstract data type)|Queue (abstract data type)|Array (data structure)|Graph (abstract data type)|Tree (data structure)|Trie|Depth-first search|Breadth-first search|Dijkstra's algorithm|Quicksort|Merge sort|Heapsort|Bubble sort|Insertion sort|Selection sort|Dynamic programming|Greedy algorithm|Divide-and-conquer algorithm|Recursion (computer science)|Big O notation|Time complexity|NP-completeness|P versus NP problem|Halting problem|Turing machine|Finite-state machine|Regular expression|Context-free grammar|Pushdown automaton|Church–Turing thesis|Minimum spanning tree|Kruskal's algorithm|Topological sorting|Bloom filter|Memoization|Binary search|Object-oriented programming|Inheritance (object-oriented programming)|Polymorphism (computer science)|Encapsulation (computer programming)|Abstraction (computer science)|Pointer (computer programming)|Garbage collection (computer science)|Compiler|Interpreter (computing)|Type system|Variable (computer science)|Subroutine|Scope (computer science)|Closure (computer programming)|Exception handling|Bitwise operation|Integer overflow|Floating-point arithmetic|Two's complement|Binary number|Hexadecimal|Boolean algebra|Logic gate|ASCII|Unicode|Endianness|CPU cache|Instruction set architecture|Instruction pipelining|Memory hierarchy|Virtual memory|Memory paging|Process (computing)|Thread (computing)|Deadlock (computer science)|Semaphore (programming)|Mutual exclusion|Scheduling (computing)|Context switch|File system|Interrupt|Von Neumann architecture|Reduced instruction set computer|Processor register|Assembly language|Machine code|Race condition|Kernel (operating system)|System call|Transmission Control Protocol|User Datagram Protocol|Internet Protocol|IP address|Domain Name System|HTTP|Router (computing)|OSI model|Ethernet|Subnetwork|Network address translation|Packet switching|Transport Layer Security|Firewall (computing)|Dynamic Host Configuration Protocol|Public-key cryptography|Symmetric-key algorithm|RSA cryptosystem|Advanced Encryption Standard|Cryptographic hash function|Digital signature|Diffie–Hellman key exchange|Man-in-the-middle attack|SQL injection|Cross-site scripting|Phishing|Malware|Ransomware|Denial-of-service attack|Buffer overflow|Authentication|Access control|Public key certificate|Password cracking|Zero-day vulnerability|Social engineering (security)|Cross-site request forgery|Brute-force attack|One-time pad|Block cipher|Stream cipher|Kerberos (protocol)|Virtual private network|Intrusion detection system|Principle of least privilege|Gradient descent|Stochastic gradient descent|Linear regression|Logistic regression|Overfitting|Regularization (mathematics)|Bias–variance tradeoff|Cross-validation (statistics)|Decision tree learning|Random forest|Support vector machine|K-nearest neighbors algorithm|K-means clustering|Naive Bayes classifier|Principal component analysis|Convolutional neural network|Recurrent neural network|Backpropagation|Activation function|Loss function|Bayes' theorem|Maximum likelihood estimation|Expectation–maximization algorithm|Hidden Markov model|Reinforcement learning|Ensemble learning|Boosting (machine learning)|Feature selection|Curse of dimensionality|Confusion matrix|Receiver operating characteristic|Perceptron|Multilayer perceptron|Hyperparameter optimization|Relational database|SQL|Database normalization|ACID|Database index|Database transaction|Software testing|Agile software development|Unit testing|Version control|Software design pattern""".split("|")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    extra = "--extra" in sys.argv
    out_path = OUT_EXTRA if extra else OUT

    titles = []
    if extra:
        titles = sorted(set(CORE_TOPICS))
        print(f"Extra pass: {len(titles)} core curriculum topics", flush=True)
    else:
        print("Collecting article titles...", flush=True)
        for category in CATEGORIES:
            found = category_titles(category, PER_CATEGORY)
            titles += found
            print(f"  {category}: {len(found)}", flush=True)
        titles = sorted(set(titles))
    print(f"{len(titles)} unique articles", flush=True)

    # Exam text, used only to throw away contaminated passages.
    banned = set()
    for q in load_questions():
        banned |= ngrams(q["question"])
        for choice in q["choices"]:
            banned |= ngrams(choice)
    print(f"{len(banned)} exam 8-grams loaded for the contamination check", flush=True)

    # In the extra pass, skip articles the category pass already collected.
    already = set()
    if extra and os.path.exists(OUT):
        already = {json.loads(line)["title"] for line in open(OUT, encoding="utf-8")}
    titles = [t for t in titles if t not in already]
    print(f"{len(titles)} not already in the library", flush=True)

    kept, dropped, short = [], 0, 0
    for i, title in enumerate(titles):
        text = fetch_text(title)
        chunks = chunk_text(title, text)
        if not chunks:
            short += 1
        for chunk in chunks:
            if ngrams(chunk) & banned:
                dropped += 1
            else:
                kept.append({"id": len(kept), "title": title, "text": chunk})
        if (i + 1) % 50 == 0:
            print(f"  fetched {i + 1}/{len(titles)} articles, {len(kept)} passages so far", flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"\nSaved {len(kept)} passages from {len(titles) - short} articles to {out_path}")
    print(f"Removed {dropped} passages that overlapped exam text (contamination check)")


if __name__ == "__main__":
    main()
