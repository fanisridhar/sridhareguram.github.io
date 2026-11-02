"""SIMULATED OpenAI-compatible LLM server with continuous batching.

NOT a real engine and NOT vLLM/TensorRT-LLM. It exists so the load driver,
metrics, plots and quality tooling can be validated on a machine without a GPU,
and to demonstrate WHY latency rises under load in a continuous-batching server:

  * `max_num_seqs` slots decode together; excess requests wait in a queue
    (queueing delay -> TTFT grows once concurrency > slots)
  * each decode step costs  base_ms + per_seq_ms * batch   (bigger batch -> slower tokens)
  * prefill of newly admitted prompts blocks the step loop
    (long prompts stall everyone's decode -> TPOT spikes)

Two parameter profiles (A/B) stand in for "two engines". Their numbers are
invented for the simulation and say NOTHING about real vLLM vs TensorRT-LLM.
"""
import argparse
import asyncio
import json
import random
import zlib

from aiohttp import web

PROFILES = {
    "sim-a": dict(max_num_seqs=32, base_ms=9.0, per_seq_ms=0.55, prefill_ms_per_word=0.020),
    "sim-b": dict(max_num_seqs=32, base_ms=6.5, per_seq_ms=0.40, prefill_ms_per_word=0.014),
}


class Engine:
    def __init__(self, p, quant_flip=0.0):
        self.p, self.quant_flip = p, quant_flip
        self.waiting, self.active = [], []
        self.wake = asyncio.Event()

    def submit(self, prompt_words, max_tokens, rid_seed):
        req = {"pw": prompt_words, "left": max_tokens, "q": asyncio.Queue(), "i": 0, "seed": rid_seed}
        self.waiting.append(req)
        self.wake.set()
        return req

    async def loop(self):
        p = self.p
        while True:
            if not self.waiting and not self.active:
                self.wake.clear()
                await self.wake.wait()
            # admit
            admitted = []
            while self.waiting and len(self.active) < p["max_num_seqs"]:
                r = self.waiting.pop(0)
                self.active.append(r)
                admitted.append(r)
            if admitted:  # prefill blocks the loop
                await asyncio.sleep(sum(r["pw"] for r in admitted) * p["prefill_ms_per_word"] / 1000)
            await asyncio.sleep((p["base_ms"] + p["per_seq_ms"] * len(self.active)) / 1000)
            for r in list(self.active):
                tok = self._token(r)
                r["q"].put_nowait(tok)
                r["i"] += 1
                r["left"] -= 1
                if r["left"] <= 0:
                    r["q"].put_nowait(None)
                    self.active.remove(r)

    def _token(self, r):
        # deterministic "reference" text from prompt seed; quantised variant flips a fraction of tokens
        base = random.Random(r["seed"] * 100003 + r["i"]).randrange(1000)
        if self.quant_flip and random.Random(r["seed"] * 7919 + r["i"] * 31).random() < self.quant_flip:
            base = (base + 1) % 1000
        return f"w{base} "


async def completions(request):
    eng: Engine = request.app["engine"]
    body = await request.json()
    pw = len(body["prompt"].split())
    req = eng.submit(pw, body.get("max_tokens", 64), zlib.crc32(body["prompt"].encode()) & 0xFFFF)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    while True:
        tok = await req["q"].get()
        if tok is None:
            await resp.write(b"data: [DONE]\n\n")
            break
        await resp.write(f"data: {json.dumps({'choices': [{'text': tok}]})}\n\n".encode())
    return resp


async def on_start(app):
    app["task"] = asyncio.create_task(app["engine"].loop())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, default="sim-a")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--quant-flip", type=float, default=0.0,
                    help="fraction of tokens perturbed, to emulate a lossy quantised variant")
    a = ap.parse_args()
    app = web.Application()
    app["engine"] = Engine(PROFILES[a.profile], a.quant_flip)
    app.router.add_post("/v1/completions", completions)
    app.on_startup.append(on_start)
    web.run_app(app, port=a.port, print=None)


if __name__ == "__main__":
    main()
