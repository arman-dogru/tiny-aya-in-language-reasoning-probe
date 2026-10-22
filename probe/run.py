"""Generate reasoning traces against a local llama.cpp server (CPU).

One JSONL record per (model, arm, item). The run is resumable: records already
present in the output file are skipped. Decoding is greedy (temperature 0) so a
rerun with the same GGUF, llama.cpp build and thread count reproduces the
traces; the model cards recommend temperature 0.6 / top-p 0.95, see README
"Limitations".

Arms
- direct : no reasoning. Answer line is prefilled ("#### ") and only a few
           tokens are generated.  Baseline for "does reasoning help at all?".
- l2     : "think in the same language as the question" + the reasoning is
           prefilled with an in-language step-by-step phrase (probe/langs.py).
- en     : "think in English" + reasoning prefilled with "Step-by-Step Answer:".
- native : (thinking models only) no language instruction or prefill; the
           model's default reasoning language.

Budget: reasoning is capped at --max-think tokens. If the cap is hit before an
answer line appears, the answer is budget-forced by appending "\n#### " (or
"\n</think>\n\n#### " for <think> models) and decoding a few more tokens. Cap
hits are recorded per trace.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from probe.langs import EN_OPENER, LANGS

FMT = {
    "math": 'write the final answer (a number only) on its own last line as "#### <number>".',
    "piqa": 'write the final answer (one letter: A, B, C or D) on its own last line as "#### <letter>".',
}
DIRECT_FMT = {
    "math": 'Give only the final answer (a number only) on one line as "#### <number>".',
    "piqa": 'Give only the final answer (one letter: A, B, C or D) on one line as "#### <letter>".',
}


def user_message(item: dict, arm: str) -> str:
    q, t = item["question"], item["task"]
    if arm == "direct":
        return f"{q}\n\n{DIRECT_FMT[t]}"
    if arm == "l2":
        return f"{q}\n\nThink step by step in the same language as the question, then {FMT[t]}"
    if arm == "en":
        return f"{q}\n\nThink step by step in English, then {FMT[t]}"
    if arm == "native":
        return f"{q}\n\nThink step by step, then {FMT[t]}"
    raise ValueError(arm)


def prefill(item: dict, arm: str, think_tags: bool, think_open: str = "<think>\n", think_close: str = "</think>") -> str:
    pre = think_open if think_tags else ""
    if arm == "direct":
        return (f"{think_open}\n{think_close}\n\n" if think_tags else "") + "#### "
    if arm == "l2":
        return pre + LANGS[item["lang"]]["opener"]
    if arm == "en":
        return pre + EN_OPENER
    if arm == "native":
        return pre
    raise ValueError(arm)


class Server:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.load(r)

    def props(self) -> dict:
        with urllib.request.urlopen(self.url + "/props", timeout=60) as r:
            return json.load(r)

    def template(self, msg: str) -> str:
        return self.post("/apply-template", {"messages": [{"role": "user", "content": msg}]})["prompt"]

    def complete(self, prompt: str, n: int, stop: list[str] | None = None) -> dict:
        body = {"prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1, "seed": 0,
                "cache_prompt": True, "stop": stop or []}
        return self.post("/completion", body)


def has_answer(text: str) -> bool:
    return "####" in text


def run_one(srv: Server, item: dict, arm: str, model: str, think_tags: bool, max_think: int,
            think_open: str = "<think>\n", think_close: str = "</think>") -> dict:
    msg = user_message(item, arm)
    prompt = srv.template(msg)
    pre = prefill(item, arm, think_tags, think_open, think_close)
    t0 = time.time()
    if arm == "direct":
        o = srv.complete(prompt + pre, 12, stop=["\n"])
        gen, n_tok, cap_hit, forced = o["content"], o["timings"]["predicted_n"], False, False
    else:
        # Stop as soon as the answer line is complete (keeps CPU cost down).
        o = srv.complete(prompt + pre, max_think)
        gen, n_tok = o["content"], o["timings"]["predicted_n"]
        cap_hit = bool(o.get("stop_type") == "limit" or (o.get("truncated") and not o.get("stopped_eos")))
        cap_hit = cap_hit or (n_tok >= max_think)
        forced = False
        if not has_answer(gen.split(think_close)[-1] if think_tags else gen) or (think_tags and think_close not in gen):
            if cap_hit:
                tail = (f"\n{think_close}\n\n#### " if think_tags and think_close not in gen else "\n#### ")
                o2 = srv.complete(prompt + pre + gen + tail, 12, stop=["\n"])
                gen = gen + tail + o2["content"]
                n_tok += o2["timings"]["predicted_n"]
                forced = True
    return dict(id=item["id"], task=item["task"], lang=item["lang"], gold=item["gold"], model=model, arm=arm,
                user_message=msg, prefill=pre, completion=gen, gen_tokens=n_tok, cap_hit=cap_hit,
                budget_forced=forced, max_think=max_think, wall_s=round(time.time() - t0, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--model", required=True, help="label, e.g. tiny-aya-global-q4km")
    ap.add_argument("--think-tags", action="store_true", help="model reasons inside <think></think>")
    ap.add_argument("--think-open", default="<think>\n",
                    help="text prefilled to open the reasoning block ('' if the chat template already opens it, e.g. Tiny Aya thinkers' <|START_THINKING|>)")
    ap.add_argument("--think-close", default="</think>", help="e.g. <|END_THINKING|> for Tiny Aya thinkers")
    ap.add_argument("--arms", default="direct,l2,en")
    ap.add_argument("--slice", default="data/slice.jsonl")
    ap.add_argument("--tasks", default="math,piqa")
    ap.add_argument("--langs", default="")
    ap.add_argument("--limit-per-cell", type=int, default=0, help="use only the first N items per (task,lang)")
    ap.add_argument("--max-think", type=int, default=768)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.slice, encoding="utf-8")]
    items = [i for i in items if i["task"] in a.tasks.split(",")]
    if a.langs:
        items = [i for i in items if i["lang"] in a.langs.split(",")]
    if a.limit_per_cell:
        seen: dict = {}
        keep = []
        for i in items:
            k = (i["task"], i["lang"])
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= a.limit_per_cell:
                keep.append(i)
        items = keep
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["id"], r["arm"]))
    srv = Server(a.server)
    props = srv.props()
    meta = dict(model_path=os.path.basename(props.get("model_path", "")), build=props.get("build_info", ""),
                n_ctx=props.get("default_generation_settings", {}).get("n_ctx"))
    # Order: round-robin over (task, lang) cells, all arms of an item back to
    # back, so a partial run is still balanced across cells and paired across arms.
    rank: dict = {}
    order = []
    for i in items:
        k = (i["task"], i["lang"])
        rank[k] = rank.get(k, -1) + 1
        order.append((rank[k], i))
    order.sort(key=lambda x: x[0])
    jobs = [(i, arm) for _, i in order for arm in a.arms.split(",") if (i["id"], arm) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done) meta={meta}", flush=True)
    lock = threading.Lock()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t_start = time.time()
    n_done = [0]

    def work(job):
        item, arm = job
        rec = run_one(srv, item, arm, a.model, a.think_tags, a.max_think, a.think_open, a.think_close)
        rec.update(meta)
        with lock:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done[0] += 1
            el = time.time() - t_start
            print(f"[{n_done[0]}/{len(jobs)}] {rec['id']} {arm} tok={rec['gen_tokens']} cap={rec['cap_hit']} "
                  f"{rec['wall_s']}s elapsed={el/60:.1f}m", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))


if __name__ == "__main__":
    main()
