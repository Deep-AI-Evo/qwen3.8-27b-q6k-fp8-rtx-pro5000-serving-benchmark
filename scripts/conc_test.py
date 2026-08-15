import json, time, urllib.request, threading, sys

BASE = sys.argv[1]
BASE = sys.argv[1]
MODEL = sys.argv[2]
N_STREAMS = int(sys.argv[3])
MAX_TOK = int(sys.argv[4])
LABEL = sys.argv[5]

UNITS = [
    "The quick brown fox jumps over the lazy dog and runs across the green meadow. ",
    "In a quiet village by the sea, the fishermen wake early each morning. ",
    "Modern machine learning models process vast amounts of information daily. ",
    "The ancient library contained thousands of scrolls and manuscripts. ",
]
results = [None] * N_STREAMS

def worker(i):
    text = UNITS[i % len(UNITS)] * (8000 // 12 + 20)   # ~8K tokens
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": text}],
               "max_tokens": MAX_TOK, "temperature": 0.2,
               "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t_req = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=900)
    first_t = None; n_out = 0; buf = b""; usage = None
    for raw in resp:
        buf += raw
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"): continue
            data = line[5:].strip()
            if data == b"[DONE]": break
            obj = json.loads(data)
            if obj.get("usage"): usage = obj["usage"]; continue
            for c in obj.get("choices") or []:
                delta = c.get("delta") or {}
                d = delta.get("content") or delta.get("reasoning_content") or ""
                if d:
                    if first_t is None: first_t = time.perf_counter()
                    n_out += 1
    t_end = time.perf_counter()
    c_tok = usage["completion_tokens"] if usage else n_out
    results[i] = {"req_wait_s": round(first_t - t_req, 3),
                  "decode_tps": round((c_tok - 1) / (t_end - first_t), 2) if c_tok > 1 else 0,
                  "tokens": c_tok, "first_t": first_t, "end_t": t_end,
                  "prompt_tokens": usage["prompt_tokens"] if usage else None}

threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_STREAMS)]
t_start = time.perf_counter()
for t in threads: t.start()
for t in threads: t.join()
wall = time.perf_counter() - t_start

tot_tok = sum(r["tokens"] - 1 for r in results)
first_all = min(r["first_t"] for r in results)
end_all = max(r["end_t"] for r in results)
agg = tot_tok / (end_all - first_all)
avg = sum(r["decode_tps"] for r in results) / N_STREAMS

print(f"=== {LABEL}: {N_STREAMS}-way concurrent ({BASE}) ===")
for i, r in enumerate(results):
    print(f"  stream {i}: TTFT_wait={r['req_wait_s']}s  decode={r['decode_tps']} t/s  tokens={r['tokens']}  prompt={r['prompt_tokens']}")
print(f"  wall time: {wall:.2f}s")
print(f"  aggregate decode: {agg:.2f} t/s")
print(f"  average per-stream decode: {avg:.2f} t/s")
print("ALL_DONE")
