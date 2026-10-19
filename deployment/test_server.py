"""
End-to-end check of the live backend. Start the server first (deployment/run_live.ps1), then:
    python deployment/test_server.py

Checks: health and model list answer; a generation streams tokens for one
model; a second model loads (swap) and also streams; GPU memory does not grow
across the swap; limits reject an oversized prompt and an unknown model.
"""

import json
import subprocess
import time

import requests

BASE = "http://127.0.0.1:8000"
checks = []  # (check, result) pairs, saved to results/server_test.json for the reports


def gpu_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


def generate(model, prompt, n=24):
    events = []
    with requests.post(f"{BASE}/api/generate", json={"model": model, "prompt": prompt, "maxNewTokens": n}, stream=True, timeout=600) as r:
        assert r.status_code == 200, r.text
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                events.append(json.loads(line[6:]))
    kinds = [e["type"] for e in events]
    text = "".join(e["text"] for e in events if e["type"] == "token")
    done = next(e for e in events if e["type"] == "done")
    return kinds, text, done


health = requests.get(f"{BASE}/api/health", timeout=10).json()
print("health:", health)
checks.append(("Health endpoint", f"answers; GPU: {health['gpu']}"))
models = [m["id"] for m in requests.get(f"{BASE}/api/models", timeout=10).json()]
print("runnable models:", models)
assert len(models) >= 2
checks.append(("Model list", f"{len(models)} runnable models found on disk"))

t0 = time.time()
kinds, text, done = generate(models[0], "Explain what a hash table is in one sentence.")
print(f"{models[0]}: events {sorted(set(kinds))}, {done['tokens']} tokens at {done['tokensPerSecond']} tokens/s, {time.time() - t0:.0f}s total")
print("  answer:", text[:160].replace("\n", " "))
assert done["tokens"] > 0 and "token" in kinds
checks.append((f"Streaming ({models[0]})", f"{done['tokens']} word pieces streamed at {done['tokensPerSecond']} per second, {time.time() - t0:.0f} s including loading"))
mem_first = gpu_mb()

kinds, text, done = generate(models[1], "Explain what a hash table is in one sentence.")
assert "loading" in kinds and "ready" in kinds, "second model should trigger a swap"
mem_second = gpu_mb()
print(f"{models[1]}: swapped and streamed {done['tokens']} tokens; GPU memory {mem_first} MB -> {mem_second} MB")
assert mem_second < mem_first + 1500, "GPU memory grew too much across a swap (old model not released?)"
checks.append(("Swapping models", f"{models[1]} loaded in place of {models[0]}; GPU memory {mem_first} MB before, {mem_second} MB after"))

r = requests.post(f"{BASE}/api/generate", json={"model": models[0], "prompt": "x" * 2001}, timeout=30)
assert r.status_code == 422, r.status_code
r = requests.post(f"{BASE}/api/generate", json={"model": "original", "prompt": "hello"}, timeout=30)
assert r.status_code == 404, r.status_code
print("limits: oversized prompt rejected (422), non-runnable model rejected (404)")
checks.append(("Limits", "a 2,001 character prompt is refused (422); the original, which cannot run here, is refused (404)"))
preflight = requests.options(f"{BASE}/api/generate", headers={
    "Origin": "https://saribazim.github.io", "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "content-type", "Access-Control-Request-Private-Network": "true"}, timeout=10)
assert preflight.status_code == 200 and preflight.headers.get("access-control-allow-private-network") == "true", preflight.text
stranger = requests.get(f"{BASE}/api/health", headers={"Origin": "https://example.com"}, timeout=10)
assert "access-control-allow-origin" not in stranger.headers
print("browser access: the published site is allowed (including Chrome's private network check); other sites are not")
checks.append(("Browser access", "the published site's address passes the browser check, including Chrome's private network question; any other site is not allowed"))
json.dump({"checks": checks, "bugs": [["The published website would have been refused by the server in Chrome.",
                                        "Chrome asks a special question before an internet page may talk to a server on your own PC; the web framework refuses it unless told otherwise, and my own extra header came too late.",
                                        "Found by the automated preflight check; switched on the framework's own private network setting and added the check to the test."]]},
          open("results/server_test.json", "w"), indent=1)
print("\nAll live backend checks passed.")
