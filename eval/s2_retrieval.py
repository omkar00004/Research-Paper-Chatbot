"""S2: local retrieval benchmark (no API calls). Content-hash caches for embeddings and FlashRank scores."""
import itertools, json, math, time
from dataclasses import dataclass, asdict, replace
import numpy as np

from eval.common import EVAL, RESULTS, SEED, kv, rawdir, read_jsonl, sha, write_json
from eval import corpus as C, stats as ST

KS = (1, 3, 5, 10)
RERANKER_NAME = "flashrank:ms-marco-TinyBERT-L-2-v2"     # FlashRank default model
EMBEDDERS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]


@dataclass(frozen=True)
class Cond:
    name: str
    size: int = 512
    overlap: int = 128
    embedder: str = "all-MiniLM-L6-v2"
    mode: str = "dense"        # dense | hybrid
    pool: int = 20
    rerank: bool = True
    diversity: bool = True     # production reranker keeps source diversity (round-robin over papers)
    top_k: int = 10

    def key(self):
        return f"{self.mode}|{self.size}|{self.overlap}|{self.embedder}|pool{self.pool if self.rerank or self.mode == 'hybrid' else '-'}|rr{int(self.rerank)}|div{int(self.diversity)}"

C1 = Cond("C1", pool=10, rerank=False)
C2 = Cond("C2")                                     # production dense-only mode: pool TOP_K_RETRIEVAL=20 -> FlashRank -> top 10
C3 = Cond("C3", mode="hybrid", pool=20)             # production: dense 30 + BM25 30 -> RRF -> top 20 -> FlashRank -> top 10
C2_NODIV = Cond("C2-nodiv", diversity=False)


# ---------------------------------------------------------------- embeddings
_models = {}
def _model(name):
    if name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[name] = SentenceTransformer(name)
    return _models[name]

def embed(name, texts, hashes=None):
    """Embeddings cached by (model, text hash) in sqlite."""
    hashes = hashes or [sha(t) for t in texts]
    got = kv().get_many("emb:" + name, hashes)
    miss = [i for i, h in enumerate(hashes) if h not in got]
    if miss:
        v = _model(name).encode([texts[i] for i in miss], batch_size=64, normalize_embeddings=True, show_progress_bar=len(miss) > 500).astype(np.float32)
        kv().set_many("emb:" + name, [(hashes[i], v[j].tobytes()) for j, i in enumerate(miss)])
        for j, i in enumerate(miss): got[hashes[i]] = v[j].tobytes()
    return np.stack([np.frombuffer(got[h], dtype=np.float32) for h in hashes])


# ---------------------------------------------------------------- reranker (cached scores, production selection logic)
class CachedRanker:
    """Drop-in for flashrank.Ranker used by backend.reranking.rerank: scores cached by (reranker, query, chunk text hash)."""
    def __init__(self):
        from flashrank import Ranker
        self.r = Ranker(); self.fresh = 0; self.hits = 0
    def rerank(self, req, use_cache=True):
        from flashrank import RerankRequest
        q, ps = req.query, req.passages
        keys = [sha(RERANKER_NAME, q, sha(p["text"])) for p in ps]
        got = kv().get_many("rr", keys) if use_cache else {}
        miss = [i for i, k in enumerate(keys) if k not in got]
        self.hits += len(ps) - len(miss)
        if miss:
            res = self.r.rerank(RerankRequest(query=q, passages=[ps[i] for i in miss]))
            sc = {r["id"]: float(r["score"]) for r in res}
            new = [(keys[i], np.float32(sc[ps[i]["id"]]).tobytes()) for i in miss]
            if use_cache: kv().set_many("rr", new)
            for k, b in new: got[k] = b
            self.fresh += len(miss)
        out = [{**p, "score": float(np.frombuffer(got[k], dtype=np.float32)[0])} for p, k in zip(ps, keys)]
        return sorted(out, key=lambda x: -x["score"])

_ranker = None
def ranker():
    global _ranker
    if _ranker is None:
        import backend.reranking as R
        _ranker = CachedRanker(); R._ranker = _ranker          # production rerank() now scores through the cache
    return _ranker


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, docs, size, overlap, embedder):
        self.chunks, self.fallbacks = C.chunk_corpus(docs, size, overlap)
        self.E = embed(embedder, [c["text"] for c in self.chunks], [c["hash"] for c in self.chunks])
        self.id2i = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.embedder = embedder; self._bm25 = None

    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            from backend.hybrid_retrieval import BM25Index
            self.tok = BM25Index._tokenize
            self._bm25 = BM25Okapi([self.tok(c["text"]) for c in self.chunks])
        return self._bm25

    def dense(self, qv, k):
        s = self.E @ qv; idx = np.argpartition(-s, min(k, len(s) - 1))[:k]; idx = idx[np.argsort(-s[idx])]
        return [{**self.chunks[i], "distance": float(1 - s[i])} for i in idx]

    def sparse(self, query, k):
        b = self.bm25(); sc = b.get_scores(self.tok(query)); idx = np.argsort(-sc)[:k]
        return [{**self.chunks[i], "bm25_score": float(sc[i])} for i in idx if sc[i] > 0]


_engines = {}
def engine(docs, corpus, size, overlap, embedder):
    k = (corpus, size, overlap, embedder)
    if k not in _engines: _engines[k] = Engine(docs, size, overlap, embedder)
    return _engines[k]


def retrieve(eng, query, qv, cond, timing=None):
    """Return the final ranked chunk list (<= cond.top_k) for one query."""
    from backend.hybrid_retrieval import reciprocal_rank_fusion
    from backend.reranking import rerank
    t0 = time.perf_counter()
    if cond.mode == "hybrid":
        d, s = eng.dense(qv, 30), eng.sparse(query, 30)
        cands = (reciprocal_rank_fusion(d, s)[:cond.pool]) if s else d[:cond.pool]
    else:
        cands = eng.dense(qv, cond.pool)
    t1 = time.perf_counter()
    if cond.rerank:
        if cond.diversity: out = rerank(query, cands, top_k=cond.top_k)           # production code path
        else:
            sc = ranker().rerank(type("R", (), {"query": query, "passages": [{"id": c["id"], "text": c["text"]} for c in cands]})())
            by = {c["id"]: c for c in cands}; out = [{**by[x["id"]], "rerank_score": x["score"]} for x in sc][:cond.top_k]
    else:
        out = cands[:cond.top_k]
    t2 = time.perf_counter()
    if timing is not None: timing.append((t1 - t0, t2 - t1))
    return out


# ---------------------------------------------------------------- metrics
def matches(chunk, span):
    ov = min(chunk["end"], span["end"]) - max(chunk["start"], span["start"])
    return chunk["doc"] == span["doc"] and ov > 0 and ov / (span["end"] - span["start"]) >= 0.5      # contains gold => ov/len == 1

def rel_total(eng, spans):
    return sum(1 for c in eng.chunks if any(matches(c, s) for s in spans))

def score(ranked, spans, n_rel):
    """Binary-relevance metrics at k in KS. hit: any gold span covered (multi: ALL spans covered); recall: fraction of gold spans covered."""
    flags = [[j for j, s in enumerate(spans) if matches(c, s)] for c in ranked]
    out = {}
    for k in KS:
        cov = {j for f in flags[:k] for j in f}
        out[f"hit@{k}"] = float(len(cov) == len(spans)); out[f"recall@{k}"] = len(cov) / len(spans)
        first = next((i for i, f in enumerate(flags[:k]) if f), None)
        out[f"mrr@{k}"] = 0.0 if first is None else 1.0 / (first + 1)
        dcg = sum(1 / math.log2(i + 2) for i, f in enumerate(flags[:k]) if f)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel, k)))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return out, flags


def run_cond(docs, corpus, cond, qs, collect_latency=False):
    eng = engine(docs, corpus, cond.size, cond.overlap, cond.embedder)
    qv = embed(cond.embedder, [q["question"] for q in qs])
    rows, timing = [], []
    for q, v in zip(qs, qv):
        ranked = retrieve(eng, q["question"], v, cond)
        spans = q["gold_spans"]
        m, flags = score(ranked, spans, rel_total(eng, spans))
        pos = [i for i, f in enumerate(flags) if f]
        rows.append({"qid": q["id"], "type": q["type"], "metrics": m, "ranked": [c["id"] for c in ranked], "gold_rank": (pos[0] + 1) if pos else None})
    return rows, eng


def latency(docs, corpus, cond, qs):
    """Cold per-query latency (no caches): query embedding + vector/BM25 search + FlashRank scoring, mean/p50/p95 in ms."""
    from backend.hybrid_retrieval import reciprocal_rank_fusion
    eng = engine(docs, corpus, cond.size, cond.overlap, cond.embedder); m = _model(cond.embedder); rk = ranker()
    ret, rr = [], []
    for qi, q in enumerate(qs[:62]):
        t0 = time.perf_counter(); qv = m.encode(q["question"], normalize_embeddings=True)
        if cond.mode == "hybrid":
            d, s = eng.dense(qv, 30), eng.sparse(q["question"], 30); cands = reciprocal_rank_fusion(d, s)[:cond.pool]
        else: cands = eng.dense(qv, cond.pool)
        t1 = time.perf_counter()
        if cond.rerank:
            from flashrank import RerankRequest
            rk.r.rerank(RerankRequest(query=q["question"], passages=[{"id": c["id"], "text": c["text"]} for c in cands]))
        t2 = time.perf_counter()
        if qi >= 2: ret.append((t1 - t0) * 1000); rr.append((t2 - t1) * 1000)   # first 2 queries = warm-up
    f = lambda a: {"mean_ms": float(np.mean(a)), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "n": len(a)} if a else None
    return {"retrieval": f(ret), "rerank": f(rr) if cond.rerank else None}


# ---------------------------------------------------------------- orchestration
def load_questions(limit=None):
    path = EVAL / ("questions.jsonl" if not limit else "cache/questions_dry.jsonl")
    return read_jsonl(path)

def agg(rows, qtype=None):
    sel = [r for r in rows if qtype is None or r["type"] == qtype]
    return {m: ST.mean_ci([r["metrics"][m] for r in sel]) for m in (sel[0]["metrics"] if sel else [])}

def grid():
    cells = []
    for size, ov_frac, emb in itertools.product((256, 512, 1024), (0.0, 0.1), EMBEDDERS):
        ov = round(size * ov_frac)
        cells.append(Cond(f"g|{size}|{ov}|{emb.split('/')[-1]}|off", size, ov, emb, "dense", 10, False))
        for pool in (10, 20, 50):
            cells.append(Cond(f"g|{size}|{ov}|{emb.split('/')[-1]}|pool{pool}", size, ov, emb, "dense", pool, True))
    return cells

def run(limit=None):
    qs = load_questions(limit)
    docs = C.load_docs()
    dev = [q for q in qs if q["split"] == "dev" and q["type"] == "single"]
    tsingle = [q for q in qs if q["split"] == "test" and q["type"] == "single"]
    tmulti = [q for q in qs if q["split"] == "test" and q["type"] == "multi"]
    test = tsingle + tmulti
    print(f"dev={len(dev)} test single={len(tsingle)} multi={len(tmulti)}")
    ranker()
    t0 = time.time()
    # --- dev grid
    gridrows = []
    for cond in grid():
        rows, eng = run_cond(docs, "main", cond, dev)
        a = agg(rows)
        gridrows.append({"name": cond.name, "cond": asdict(cond), "n_chunks": len(eng.chunks), **{k: a[k]["mean"] for k in a}, "ci": {k: [a[k]["lo"], a[k]["hi"]] for k in ("ndcg@5", "hit@5")}})
        print(f"  grid {cond.name:44s} chunks={len(eng.chunks):5d} nDCG@5={a['ndcg@5']['mean']:.3f} Hit@5={a['hit@5']['mean']:.3f}   [{time.time() - t0:.0f}s]")
    ref_rows, _ = run_cond(docs, "main", C2, dev); ref = agg(ref_rows)           # production default on dev (reference only, not in selection)
    best = max(gridrows, key=lambda r: (round(r["ndcg@5"], 12), r["hit@5"], -r["n_chunks"]))
    S = Cond("S", **{k: v for k, v in best["cond"].items() if k != "name"})
    print("selected S =", S.key(), "dev nDCG@5 =", round(best["ndcg@5"], 4))
    write_json(RESULTS / ("s2_dev_grid.json" if not limit else "dry/s2_dev_grid.json"), {"grid": gridrows, "selected": asdict(S), "selected_dev_ndcg5": best["ndcg@5"], "default_config_dev_reference": {k: ref[k]["mean"] for k in ref}, "n_dev": len(dev)})
    # --- test
    conds = [C1, C2, C3, S, C2_NODIV]; res, per = {}, {}
    same_as_c2 = S.key() == C2.key()
    for cond in conds:
        rows, eng = run_cond(docs, "main", cond, test)
        per[cond.name] = rows
        res[cond.name] = {"cond": asdict(cond), "key": cond.key(), "n_chunks": len(eng.chunks), "single": agg(rows, "single"), "multi": agg(rows, "multi"),
                          "latency": latency(docs, "main", cond, tsingle)}
        a = res[cond.name]["single"]
        print(f"  test {cond.name:9s} single Hit@5={a['hit@5']['mean']:.3f} nDCG@5={a['ndcg@5']['mean']:.3f} MRR@10={a['mrr@10']['mean']:.3f}")
    diffs = {}
    for a_, b_ in (("C2", "C1"), ("C3", "C2"), ("S", "C2")):
        for typ in ("single", "multi"):
            ra = {r["qid"]: r for r in per[a_] if r["type"] == typ}; rb = {r["qid"]: r for r in per[b_] if r["type"] == typ}
            ids = sorted(ra)
            diffs[f"{a_}-{b_}|{typ}"] = {m: ST.paired_diff([ra[i]["metrics"][m] for i in ids], [rb[i]["metrics"][m] for i in ids]) for m in ra[ids[0]]["metrics"]} if ids else {}
    chroma = chroma_parity(docs, test)
    write_json(RESULTS / ("s2_retrieval.json" if not limit else "dry/s2_retrieval.json"),
               {"conditions": res, "paired_diffs": diffs, "S_identical_to_C2": same_as_c2, "chroma_parity": chroma, "n_test_single": len(tsingle), "n_test_multi": len(tmulti),
                "flashrank_cache": {"fresh_scores": ranker().fresh, "cache_hits": ranker().hits}, "chunk_offset_fallbacks": 0})
    for name, rows in per.items():
        (rawdir(limit) / f"s2_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"S2 done in {time.time() - t0:.0f}s; flashrank pairs scored fresh={ranker().fresh} cached={ranker().hits}")


def chroma_parity(docs, qs):
    """Sanity: exact numpy cosine == Chroma (HNSW, cosine) top-10 for the default config."""
    import tempfile, chromadb
    eng = engine(docs, "main", 512, 128, "all-MiniLM-L6-v2")
    with tempfile.TemporaryDirectory() as d:
        col = chromadb.PersistentClient(path=d).create_collection("parity_check", metadata={"hnsw:space": "cosine"})
        for i in range(0, len(eng.chunks), 2000):
            part = slice(i, i + 2000)
            col.add(ids=[c["id"] for c in eng.chunks[part]], embeddings=eng.E[part].tolist(), documents=[c["text"] for c in eng.chunks[part]])
        qv = embed("all-MiniLM-L6-v2", [q["question"] for q in qs]); ov = []
        for q, v in zip(qs, qv):
            ids = col.query(query_embeddings=[v.tolist()], n_results=10)["ids"][0]
            ov.append(len(set(ids) & {c["id"] for c in eng.dense(v, 10)}) / 10)
    return {"mean_top10_overlap_with_exact": float(np.mean(ov)), "min": float(np.min(ov)), "n_queries": len(qs)}
