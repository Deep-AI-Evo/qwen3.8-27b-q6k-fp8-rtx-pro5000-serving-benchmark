import json, time, urllib.request, random, sys

BASE, MODEL, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
sizes = [("2K", 2000, 3), ("32K", 32000, 2), ("200K", 200000, 1)]

def unique_text(n_tok):
    rnd = random.randint(10**9, 10**10)
    unit = f"The merchant guild records the number {rnd} in its ledger today. "
    return unit * (n_tok // 19 + 15)

def ttft_test(text, max_tokens=8):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": text}],
               "max_tokens": max_tokens, "temperature": 0.2,
               "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=1200)
    except urllib.error.HTTPError as e:
        print("HTTPERR", e.code, e.read().decode()[:300], flush=True)
        raise
    first_t = None; buf = b""; usage = None
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
                d = (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content") or ""
                if d and first_t is None:
                    first_t = time.perf_counter()
    ttft = first_t - t0
    return usage["prompt_tokens"], ttft, ttft

for tag, n_tok, rounds in sizes:
    for r in range(rounds):
        p_tok, ttft, _ = ttft_test(unique_text(n_tok))
        print(f"{LABEL} {tag} r{r+1}: prompt={p_tok} TTFT={ttft:.2f}s prefill={p_tok/ttft:.1f} t/s", flush=True)
print("ALL_DONE")
