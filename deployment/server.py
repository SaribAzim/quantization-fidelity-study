"""
The local live backend for the website's arena.

The public website replays recorded answers. When this server runs on the
PC with the GPU, the same page finds it and unlocks live prompts: the chosen
models answer one after another, streamed token by token.

Only one 4 bit model fits on the 8 GB GPU at a time, so a ModelManager keeps
the current model loaded and swaps when a different one is requested
(loading takes about 20 seconds). Requests run one at a time.

The server listens on 127.0.0.1 only: it is reachable from this PC's browser,
never from the internet.

Run from the project root:   python -m uvicorn deployment.server:app --host 127.0.0.1 --port 8000
(or deployment/run_live.ps1)

Endpoints
  GET  /api/health     is the server up, which model is loaded
  GET  /api/models     runnable models on this PC
  POST /api/generate   {"model": id, "prompt": text, "maxNewTokens": n}
                       streams Server Sent Events: queued, loading, ready, token, done, error
"""

import asyncio
import gc
import json
import os
import sys
import threading
import time

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # model folders are relative to the project root

from deployment.catalog import CATALOG, by_id
from deployment.generation import stream_answer

MAX_PROMPT_CHARS = 2000
MAX_NEW_TOKENS = 160
SITE_ORIGINS = ["https://saribazim.github.io", "http://localhost:5173", "http://127.0.0.1:5173",
                "http://localhost:4173", "http://127.0.0.1:4173"]


def runnable_models():
    return [m for m in CATALOG if m["runnable"] and os.path.isdir(m["folder"])]


class ModelManager:
    """Keeps at most one model on the GPU and swaps it on request."""

    def __init__(self):
        self.model_id, self.model, self.tokenizer = None, None, None

    def unload(self):
        self.model = self.tokenizer = None
        self.model_id = None
        gc.collect()
        torch.cuda.empty_cache()

    def get(self, model_id):
        if self.model_id == model_id:
            return self.model, self.tokenizer
        from quantization.packed import load_lora_state, load_packed_model
        self.unload()
        entry = by_id(model_id)
        model, tokenizer = load_packed_model(entry["folder"])
        if entry.get("adapters"):
            load_lora_state(model, torch.load(entry["adapters"]))
        self.model_id, self.model, self.tokenizer = model_id, model, tokenizer
        return model, tokenizer


manager = ModelManager()
gpu_lock = asyncio.Lock()
waiting = 0

app = FastAPI(title="Quantization Fidelity Study, live backend")
# Browsers only let the website call this server if it agrees (CORS). allow_private_network answers Chrome's
# extra question before an https page on the internet may talk to a server on this PC (127.0.0.1).
app.add_middleware(CORSMiddleware, allow_origins=SITE_ORIGINS, allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"], allow_private_network=True)


class GenerateRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    maxNewTokens: int = Field(default=96, ge=1, le=MAX_NEW_TOKENS)


@app.get("/api/health")
def health():
    return {"ok": True, "loaded": manager.model_id, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "busy": gpu_lock.locked(), "waiting": waiting}


@app.get("/api/models")
def models():
    return [{"id": m["id"], "label": m["label"], "family": m["family"]} for m in runnable_models()]


def sse(event):
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/generate")
async def generate(req: GenerateRequest, request: Request):
    if req.model not in {m["id"] for m in runnable_models()}:
        return JSONResponse({"error": f"model '{req.model}' is not runnable on this PC"}, status_code=404)

    async def events():
        global waiting
        if gpu_lock.locked():
            waiting += 1
            yield sse({"type": "queued", "ahead": waiting})
        async with gpu_lock:
            waiting = max(0, waiting - 1) if waiting else 0
            loop = asyncio.get_running_loop()
            if manager.model_id != req.model:
                yield sse({"type": "loading", "model": req.model})
                t0 = time.time()
                try:
                    await loop.run_in_executor(None, manager.get, req.model)
                except Exception as error:  # report, do not crash the server
                    yield sse({"type": "error", "message": f"could not load the model: {error}"})
                    return
                yield sse({"type": "ready", "model": req.model, "loadSeconds": round(time.time() - t0, 1)})

            queue: asyncio.Queue = asyncio.Queue()
            stop = threading.Event()

            def worker():
                try:
                    for token, text, seconds in stream_answer(manager.model, manager.tokenizer, req.prompt, req.maxNewTokens, stop.is_set):
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "token", "text": text, "t": seconds})
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                except Exception as error:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(error)})
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            count, last_t = 0, 0.0
            while True:
                event = await queue.get()
                if event is None:
                    break
                if await request.is_disconnected():  # the page was closed: stop generating
                    stop.set()
                    break
                if event["type"] == "token":
                    count, last_t = count + 1, event["t"]
                yield sse(event)
            thread.join()
            yield sse({"type": "done", "tokens": count, "seconds": last_t,
                       "tokensPerSecond": round(count / last_t, 2) if last_t else None})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
