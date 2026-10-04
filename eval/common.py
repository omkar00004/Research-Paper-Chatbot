"""Shared paths, constants, atomic IO, sqlite KV cache, resumable state."""
import hashlib, json, os, sqlite3, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EVAL = ROOT / "eval"
CACHE = EVAL / "cache"
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
for d in (CACHE, RESULTS, RAW):
    d.mkdir(parents=True, exist_ok=True)

CORPUS_DIR = ROOT / "uploads"                      # the 23 real papers (user-provided)
LEGACY_DIRS = [ROOT / "Research Paper", ROOT / "uploads" / "Attention Is All You Need.pdf"]  # corpus of the original 21-query RAGAS run
SEED = 20261004
ANSWER_MODEL = "gpt-oss-20b"            # production GROQ_MODEL is openai/gpt-oss-20b
WRITER_MODEL = "nemotron-3-super-120b"  # question writer: NOT the answer model. Same model as the judge (disclosed): qwen3.8-27b, gemini-3.x and ministral-14b hit multi-hour free-tier caps in the dry run (before any result existed); only the NVIDIA route sustained volume
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "nemotron-3-super-120b")  # judge: third family
PROVIDER = "freellmapi"
EXIT_QUOTA, EXIT_BUDGET, EXIT_DRIFT = 3, 4, 5

def rawdir(limit=None):
    d = (RESULTS / "dry" / "raw") if limit else RAW
    d.mkdir(parents=True, exist_ok=True); return d

def sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()

def atomic_write(path: Path, text: str):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def write_json(path, obj):
    atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False, default=float))

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default

def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def write_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


class KV:
    """sqlite key-value store (namespaced). Thread-safe, atomic per write."""
    def __init__(self, path=CACHE / "cache.sqlite"):
        self.path = str(path); self.lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS kv(ns TEXT, k TEXT, v BLOB, PRIMARY KEY(ns,k))")
        self.db.commit()
    def get(self, ns, k):
        with self.lock:
            r = self.db.execute("SELECT v FROM kv WHERE ns=? AND k=?", (ns, k)).fetchone()
        return r[0] if r else None
    def get_many(self, ns, keys):
        out = {}
        with self.lock:
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                q = f"SELECT k,v FROM kv WHERE ns=? AND k IN ({','.join('?' * len(part))})"
                out.update(dict(self.db.execute(q, (ns, *part)).fetchall()))
        return out
    def set_many(self, ns, items):
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO kv VALUES(?,?,?)", [(ns, k, v) for k, v in items])
            self.db.commit()
    def set(self, ns, k, v):
        self.set_many(ns, [(k, v)])

_kv = None
def kv() -> KV:
    global _kv
    if _kv is None:
        _kv = KV()
    return _kv


class State:
    """eval/state.json: status (+ payload) of every (stage, condition, question_id). Atomic, debounced writes."""
    def __init__(self, path=EVAL / "state.json", flush_every=2.0):
        self.path = Path(path); self.lock = threading.Lock(); self.flush_every = flush_every
        self.data = read_json(self.path, {}) ; self._last = 0.0; self._dirty = False
    @staticmethod
    def key(stage, cond, qid): return f"{stage}|{cond}|{qid}"
    def get(self, stage, cond, qid):
        return self.data.get(self.key(stage, cond, qid))
    def done(self, stage, cond, qid):
        r = self.get(stage, cond, qid); return bool(r and r.get("status") == "done")
    def mark(self, stage, cond, qid, status="done", out=None):
        with self.lock:
            self.data[self.key(stage, cond, qid)] = {"status": status, "out": out, "t": round(time.time())}
            self._dirty = True
            if time.time() - self._last > self.flush_every:
                self._flush()
    def _flush(self):
        write_json(self.path, self.data); self._last = time.time(); self._dirty = False
    def flush(self):
        with self.lock:
            if self._dirty: self._flush()
    def items(self, stage, cond):
        pre = f"{stage}|{cond}|"
        return {k[len(pre):]: v for k, v in self.data.items() if k.startswith(pre)}
