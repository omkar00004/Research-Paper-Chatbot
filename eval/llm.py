"""Single shared LLM client for every stage (freellmapi gateway, OpenAI-compatible).

Content-hash cache, quota-aware retries, budget guards, call ledger. No secrets are logged.
Exit codes: 3 = daily quota / outage (resume later), 4 = budget hit, 5 = frozen-model drift.
"""
import json, os, random, re, threading, time
from collections import defaultdict
from dataclasses import dataclass
import httpx
from dotenv import load_dotenv

from eval.common import CACHE, ROOT, PROVIDER, RESULTS, atomic_write, kv, read_json, sha, write_json

load_dotenv(ROOT / ".env")
BASE_URL = os.getenv("FREELLMAPI_BASE_URL", "http://127.0.0.1:31415/v1").rstrip("/")
LEDGER = CACHE / "ledger.jsonl"
FROZEN = CACHE / "frozen_models.json"
MAX_CONCURRENCY = 4
RPM = float(os.getenv("EVAL_RPM", "36"))   # 10% under the 40 RPM limit of the NVIDIA route
RUN = os.getenv("EVAL_RUN", "full")   # "dry" for dry runs: separates ledger rows, shares the cache
STAGE = "misc"
def set_stage(s):
    global STAGE; STAGE = s


class QuotaExit(Exception):
    def __init__(self, provider, reason, wait=None):
        super().__init__(f"{provider}: {reason}"); self.provider, self.reason, self.wait = provider, reason, wait
class BudgetExceeded(Exception): pass
class ModelDrift(Exception): pass
class LLMError(Exception): pass


@dataclass
class Result:
    text: str
    finish_reason: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    truncated_retry: bool = False


class Ledger:
    lock = threading.Lock()
    @classmethod
    def log(cls, kind, **kw):
        row = {"t": round(time.time(), 2), "run": RUN, "stage": STAGE, "provider": PROVIDER, "kind": kind, **kw}
        with cls.lock, open(LEDGER, "a") as f:
            f.write(json.dumps(row) + "\n")

def read_ledger():
    if not LEDGER.exists(): return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]

def ledger_table(run="full"):
    """Aggregate by (stage, provider, model)."""
    agg = defaultdict(lambda: defaultdict(float))
    for r in read_ledger():
        if r["run"] != run: continue
        a = agg[(r["stage"], r["provider"], r.get("model", "-"))]
        k = r["kind"]
        if k == "call":
            a["calls"] += 1; a["prompt_tokens"] += r.get("prompt_tokens", 0); a["completion_tokens"] += r.get("completion_tokens", 0)
            a["latency"] += r.get("latency", 0); a["wall_s"] += r.get("latency", 0)
        elif k == "cache_hit": a["cache_hits"] += 1
        elif k == "retry": a["retries"] += 1
        elif k == "429": a["http_429"] += 1
        elif k == "5xx": a["http_5xx"] += 1
        elif k == "truncation": a["truncations"] += 1
        elif k == "failed": a["failed"] += 1
        elif k == "invalid_json": a["invalid_json"] += 1
    return agg

def render_ledger_md(path=RESULTS / "call_ledger.md"):
    out = ["# Call ledger\n"]
    for run in ("full", "dry"):
        agg = ledger_table(run)
        if not agg: continue
        out.append(f"\n## run = {run}\n")
        out.append("| stage | provider | model | HTTP calls | cache hits | hit rate | retries | 429s | 5xx | truncations (length retries) | failed | invalid JSON | prompt tok | completion tok | mean latency s |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        tot = defaultdict(float)
        for (st, pr, mo), a in sorted(agg.items()):
            lookups = a["calls"] + a["cache_hits"]
            rate = f"{a['cache_hits'] / lookups:.0%}" if lookups else "-"
            lat = f"{a['latency'] / a['calls']:.2f}" if a["calls"] else "-"
            out.append(f"| {st} | {pr} | {mo} | {int(a['calls'])} | {int(a['cache_hits'])} | {rate} | {int(a['retries'])} | {int(a['http_429'])} | {int(a['http_5xx'])} | {int(a['truncations'])} | {int(a['failed'])} | {int(a['invalid_json'])} | {int(a['prompt_tokens'])} | {int(a['completion_tokens'])} | {lat} |")
            for k, v in a.items(): tot[k] += v
        out.append(f"| **total** | | | {int(tot['calls'])} | {int(tot['cache_hits'])} | | {int(tot['retries'])} | {int(tot['http_429'])} | {int(tot['http_5xx'])} | {int(tot['truncations'])} | {int(tot['failed'])} | {int(tot['invalid_json'])} | {int(tot['prompt_tokens'])} | {int(tot['completion_tokens'])} | |")
    out.append("\nCost: all models are served by freellmapi free tiers; dollar cost = $0.00 (prices configurable via PRICE_IN_PER_M / PRICE_OUT_PER_M).\n")
    atomic_write(path, "\n".join(out) + "\n")
    return "\n".join(out)


def _extract_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("no JSON object found")


def retry_hint(text, headers):
    """Seconds until the provider says we may retry: Retry-After header, gateway retryAtMs, or 'reset ~4m' text. None if absent."""
    ra = headers.get("retry-after")
    if ra and re.fullmatch(r"\d+(\.\d+)?", ra): return float(ra)
    m = re.search(r'"retryAtMs"\s*:\s*(\d+)', text)
    if m: return max(0.0, int(m.group(1)) / 1000 - time.time())
    m = re.search(r"reset[^0-9~]*~?\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)\b", text)
    if m: return float(m.group(1)) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[m.group(2)]
    return None


def _base(m):
    """Normalise a returned model string so provider aliases of one model compare equal: drop provider prefix, '-latest', trailing date/version."""
    return re.sub(r"-(latest|\d{4,8})$", "", m.lower().split("/")[-1])


class LLM:
    def __init__(self, max_calls=None, max_cost=None):
        key = os.getenv("FREELLMAPI_API_KEY", "")
        if not key:
            raise SystemExit("FREELLMAPI_API_KEY is not set (put it in .env)")
        self.http = httpx.Client(base_url=BASE_URL, headers={"Authorization": f"Bearer {key}"}, timeout=180)
        self.sem = threading.Semaphore(MAX_CONCURRENCY)
        self.max_calls, self.max_cost = max_calls, max_cost
        self.calls = 0; self.cost = 0.0; self.lock = threading.Lock()
        self.consec_fail = 0; self.reset_at = 0.0; self.next_slot = 0.0; self.pace_lock = threading.Lock()
        self.p_in = float(os.getenv("PRICE_IN_PER_M", "0")); self.p_out = float(os.getenv("PRICE_OUT_PER_M", "0"))
        self.frozen = {k: ([v] if isinstance(v, str) else v) for k, v in read_json(FROZEN, {}).items()}

    # ---- http with retries -------------------------------------------------
    def _budget(self):
        with self.lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise BudgetExceeded(f"--max-calls {self.max_calls} reached for {PROVIDER}")
            if self.max_cost is not None and self.cost >= self.max_cost:
                raise BudgetExceeded(f"--max-cost {self.max_cost} reached for {PROVIDER}")
            self.calls += 1

    def _pace(self):
        """Global client-side pacing under the provider limit (NVIDIA NIM: 40 requests/min). Every HTTP attempt counts."""
        with self.pace_lock:
            now = time.time(); wait = self.next_slot - now
            self.next_slot = max(now, self.next_slot) + 60.0 / RPM
        if wait > 0: time.sleep(wait)

    def _gate(self, model):
        self._pace()
        # shared cooldown (set after a 429/5xx) so concurrent threads wait instead of burning tries; never wait > 15 min
        w = self.cool.get(model, 0) - time.time()
        if w > 0: time.sleep(w + random.random())
        w = self.reset_at - time.time()
        if w > 0 and self.remaining is not None and self.remaining <= 1:
            if w > 900: raise QuotaExit(PROVIDER, f"rate-limit window resets in {int(w)}s", w)
            time.sleep(w + 0.5)
    remaining = None
    cool = {}

    def _post(self, body, model):
        last, slept = "", 0.0
        for attempt in range(5):
            self._gate(model); self._budget()
            t0 = time.time()
            try:
                r = self.http.post("/chat/completions", json=body)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = f"transport: {type(e).__name__}"; Ledger.log("retry", model=model, why=last)
                time.sleep(min(60, 2 * 2 ** attempt) + random.random()); continue
            lat = time.time() - t0
            try:
                self.remaining = int(r.headers.get("x-ratelimit-remaining", "")); self.reset_at = float(r.headers.get("x-ratelimit-reset", 0))
            except ValueError:
                pass
            if r.status_code == 200:
                self.consec_fail = 0
                return r, lat
            text = r.text[:1500]; last = f"HTTP {r.status_code}"
            if r.status_code in (429, 500, 502, 503, 504, 408):
                Ledger.log("429" if r.status_code == 429 else "5xx", model=model, status=r.status_code, why=text[:400])
                hint = retry_hint(text, r.headers)
                daily = re.search(r"daily[_ ]quota|per[- ]day|quota.*exhaust|exhausted.*daily", text, re.I) and "upstream_error" not in text
                # the gateway labels short cooldowns 'daily_quota_exhausted' too: trust an advertised reset <= 15 min, exit otherwise
                if (hint is not None and hint > 900) or (daily and hint is None):
                    raise QuotaExit(PROVIDER, f"{'daily quota exhausted' if daily else 'reset in %ds' % hint}: {text[:240]}", hint)
                wait = (hint + 1 if hint is not None else min(60, 2 * 2 ** attempt)) + random.random() * 2
                if slept + wait > 900: raise QuotaExit(PROVIDER, f"cumulative wait > 15 min on {model}: {text[:200]}", wait)
                self.cool[model] = time.time() + wait; slept += wait
                Ledger.log("retry", model=model, why=last, wait=round(wait, 1))
                continue
            raise LLMError(f"HTTP {r.status_code}: {text[:300]}")
        self.consec_fail += 1
        Ledger.log("failed", model=model, why=last)
        if self.consec_fail >= 6:
            raise QuotaExit(PROVIDER, f"provider unavailable ({self.consec_fail} consecutive failures, last: {last})")
        raise LLMError(f"gave up after 5 tries: {last}")

    # ---- public -------------------------------------------------------------
    def chat(self, model, messages, max_tokens=150, temperature=0.0, json_mode=False, extra=None) -> Result:
        extra = extra or {}
        key = sha(PROVIDER, model, json.dumps(messages, sort_keys=True), max_tokens, temperature, json_mode, json.dumps(extra, sort_keys=True))
        hit = kv().get("llm", key)
        if hit:
            d = json.loads(hit); Ledger.log("cache_hit", model=model)
            return Result(**{**d, "cached": True})
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **extra}
        if json_mode: body["response_format"] = {"type": "json_object"}
        truncated = False
        with self.sem:
            for _ in range(2):
                r, lat = self._post(body, model)
                j = r.json(); ch = j["choices"][0]
                fin = ch.get("finish_reason") or "?"; ret = j.get("model", "?"); u = j.get("usage") or {}
                routed = r.headers.get("x-routed-via", "")
                Ledger.log("call", model=model, returned_model=ret, routed_via=routed, finish_reason=fin, latency=round(lat, 2),
                           prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0))
                with self.lock:
                    self.cost += u.get("prompt_tokens", 0) / 1e6 * self.p_in + u.get("completion_tokens", 0) / 1e6 * self.p_out
                seen = self.frozen.setdefault(model, [])
                if ret not in seen:
                    if seen and _base(ret) != _base(seen[0]):
                        raise ModelDrift(f"requested {model}: first saw '{seen[0]}', now '{ret}'. Not substituting; pause and report.")
                    seen.append(ret); write_json(FROZEN, self.frozen)   # alias strings of the same model (e.g. -latest vs -2512) are recorded, not fatal
                if fin == "length" and not truncated:   # retry once with a bigger budget
                    truncated = True; Ledger.log("truncation", model=model, max_tokens=body["max_tokens"])
                    body["max_tokens"] = min(body["max_tokens"] * 3, 4096); continue
                break
        res = Result((ch.get("message") or {}).get("content") or "", fin, ret, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), False, truncated)
        if fin != "length":
            kv().set("llm", key, json.dumps({k: getattr(res, k) for k in ("text", "finish_reason", "model", "prompt_tokens", "completion_tokens", "truncated_retry")}))
        return res

    def json_chat(self, model, messages, max_tokens=700, **kw):
        """chat() + robust JSON parse; one re-ask with a reminder if the first reply is not valid JSON."""
        for i in range(2):
            res = self.chat(model, messages if i == 0 else messages + [{"role": "user", "content": "Your previous reply was not valid JSON. Reply with ONLY the JSON object."}], max_tokens=max_tokens, json_mode=True, **kw)
            try:
                return _extract_json(res.text), res
            except Exception:
                Ledger.log("invalid_json", model=model)
        raise LLMError("invalid JSON after re-ask")

    def model_status(self):
        r = self.http.get("/models"); r.raise_for_status()
        return {m["id"]: m for m in r.json()["data"]}
