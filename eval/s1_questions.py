"""S1: build the question set (cached, resumable LLM calls by the question-writer model)."""
import random, re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from eval.common import EVAL, RESULTS, SEED, WRITER_MODEL, read_json, sha, write_json, write_jsonl
from eval.corpus import corpus_has_term, eligible_passages, load_docs, locate_quote
from eval import llm as L

BANNED = re.compile(r"(?i)\b(the passage|the excerpt|the text|this paper|this study|the study|the authors?|the given|above|following text|this work|this section)\b")
SYS = ("You write evaluation questions for a retrieval benchmark over a corpus of 23 research papers on fact-checking, claim verification, "
       "retrieval-augmented generation (RAG) and LLM self-correction. Reply with ONE JSON object and nothing else.")
SINGLE = """Passage (from the paper "{title}"):
\"\"\"{text}\"\"\"

Write ONE question that can be answered using ONLY this passage.
Rules:
- The question must be self-contained: someone searching the whole corpus must be able to tell what it asks about. Name the specific method, dataset, system or setting when needed. Never write "the passage", "the text", "the excerpt", "this paper", "the authors" or "the study", and do not quote the paper title, year or author names; use method/dataset/system names that occur inside the passage. If the question cannot be made unambiguous that way, reply {{"skip": true}}.
- Ask about a specific fact, mechanism, number, definition or reason, not about what the paper is generally about.
- "answer": at most 2 sentences, using only information in the passage.
- "evidence_quote": ONE contiguous span copied VERBATIM from the passage (40-300 characters) that contains the information needed to answer.
- If the passage is unsuitable (table fragment, references, boilerplate, garbled), reply {{"skip": true}}.
JSON: {{"question": "...", "answer": "...", "evidence_quote": "..."}}"""
MULTI = """Passage A (from the paper "{ta}"):
\"\"\"{a}\"\"\"

Passage B (from the paper "{tb}"):
\"\"\"{b}\"\"\"

Write ONE question that needs information from BOTH passages (e.g. comparing or combining what the two say about a shared topic) and cannot be answered from either alone.
Rules: self-contained question that names the specific methods/datasets/settings; never write "the passage", "the text", "this paper" or "the authors"; "answer" at most 3 sentences using only the passages; "evidence_quote_a" and "evidence_quote_b" are contiguous spans copied VERBATIM (40-300 characters) from passage A and passage B respectively. If the pair is unsuitable reply {{"skip": true}}.
JSON: {{"question": "...", "answer": "...", "evidence_quote_a": "...", "evidence_quote_b": "..."}}"""
UNANS = """The corpus consists of these 23 papers: {titles}.

Write 25 realistic, specific technical questions (this batch: {area}) that a researcher might ask but that are NOT answered by this corpus: they should concern named methods, datasets, systems or results from OTHER areas of ML/NLP/CS (e.g. computer vision, speech, robotics, game-playing RL, databases, compilers, protein modelling, recommender systems), so none of these papers discusses them. For each, give "key_terms": 1-3 distinctive proper names or technical terms from the question that would certainly appear in any text that answers it.
JSON: {{"questions": [{{"question": "...", "key_terms": ["..."]}}, ...]}}"""

STOP = set("what which does that this with from have were been their there about into used using when where whose than then also such each both more most other only over under between during after before while would could should".split())
def words(t): return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]+", t.lower()) if len(w) > 3 and w not in STOP}
def answer_in_question(q, a):
    wa = words(a)
    return bool(wa) and len(wa & words(q)) / len(wa) >= 0.8

def title(doc): return re.sub(r"^\d+[a-z]?_", "", doc[:-4]).replace("_", " ")
def flat(t): return " ".join(t.split())

def stratified(by_doc, rng):
    """Round-robin across papers; inside a paper round-robin across section types."""
    per = {}
    for d, ps in by_doc.items():
        g = defaultdict(list)
        for p in ps: g[p["section"]].append(p)
        for v in g.values(): rng.shuffle(v)
        types = sorted(g); rng.shuffle(types)
        o = []
        while any(g.values()):
            for t in types:
                if g[t]: o.append(g[t].pop())
        per[d] = o
    docs = sorted(per); rng.shuffle(docs); out = []
    while any(per.values()):
        for d in docs:
            if per[d]: out.append(per[d].pop(0))
    return out

def pools(docs):
    by = {"test": defaultdict(list), "multi": defaultdict(list), "dev": defaultdict(list)}
    for n, d in docs.items():
        for p in eligible_passages(n, d):
            h = int(sha(SEED, p["id"])[:8], 16) % 100
            by["test" if h < 55 else "multi" if h < 80 else "dev"][n].append(p)
    rng = random.Random(SEED)
    return {k: stratified(v, rng) for k, v in by.items()}

def pair_up(docs, passages):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    per, sel = Counter(), []
    for p in passages:                       # <= 8 passages per paper keeps the similarity matrix small
        if per[p["doc"]] < 8: sel.append(p); per[p["doc"]] += 1
    m = SentenceTransformer("all-MiniLM-L6-v2")
    E = m.encode([flat(docs[p["doc"]]["text"][p["start"]:p["end"]]) for p in sel], normalize_embeddings=True)
    S = E @ E.T; used, pairs, dc = set(), [], Counter()
    for i, p in enumerate(sel):
        if i in used or dc[p["doc"]] >= 3: continue
        cands = [(S[i, j], j) for j in range(len(sel)) if j not in used and j != i and sel[j]["doc"] != p["doc"] and dc[sel[j]["doc"]] < 3 and S[i, j] >= 0.35]
        if cands:
            _, j = max(cands); used |= {i, j}; dc[p["doc"]] += 1; dc[sel[j]["doc"]] += 1; pairs.append((p, sel[j]))
    return pairs

def ptext(docs, p): return flat(docs[p["doc"]]["text"][p["start"]:p["end"]])

def gen_single(llm, docs, p):
    msg = [{"role": "system", "content": SYS}, {"role": "user", "content": SINGLE.format(title=title(p["doc"]), text=ptext(docs, p))}]
    j, _ = llm.json_chat(WRITER_MODEL, msg, max_tokens=900)
    if j.get("skip"): return {"ok": False, "reason": "writer_skip"}
    q, a, ev = (str(j.get(k, "")).strip() for k in ("question", "answer", "evidence_quote"))
    if len(q) < 25 or len(a) < 3: return {"ok": False, "reason": "empty_or_short"}
    if BANNED.search(q): return {"ok": False, "reason": "banned_phrase"}
    if answer_in_question(q, a): return {"ok": False, "reason": "answer_in_question"}
    loc = locate_quote(p["doc"], docs[p["doc"]]["text"], ev, p["start"], p["end"])
    if not loc: return {"ok": False, "reason": "quote_not_verbatim"}
    if not 40 <= loc[1] - loc[0] <= 450: return {"ok": False, "reason": "quote_length"}
    return {"ok": True, "question": q, "gold_answer": a, "gold_spans": [{"doc": p["doc"], "start": loc[0], "end": loc[1], "quote": docs[p["doc"]]["text"][loc[0]:loc[1]]}],
            "section": p["section"]}

def gen_multi(llm, docs, pair):
    a, b = pair
    msg = [{"role": "system", "content": SYS}, {"role": "user", "content": MULTI.format(ta=title(a["doc"]), a=ptext(docs, a), tb=title(b["doc"]), b=ptext(docs, b))}]
    j, _ = llm.json_chat(WRITER_MODEL, msg, max_tokens=1200)
    if j.get("skip"): return {"ok": False, "reason": "writer_skip"}
    q, ans = (str(j.get(k, "")).strip() for k in ("question", "answer"))
    if len(q) < 25 or len(ans) < 3: return {"ok": False, "reason": "empty_or_short"}
    if BANNED.search(q): return {"ok": False, "reason": "banned_phrase"}
    spans = []
    for p, k in ((a, "evidence_quote_a"), (b, "evidence_quote_b")):
        loc = locate_quote(p["doc"], docs[p["doc"]]["text"], str(j.get(k, "")), p["start"], p["end"])
        if not loc or not 40 <= loc[1] - loc[0] <= 450: return {"ok": False, "reason": "quote_not_verbatim"}
        spans.append({"doc": p["doc"], "start": loc[0], "end": loc[1], "quote": docs[p["doc"]]["text"][loc[0]:loc[1]]})
    return {"ok": True, "question": q, "gold_answer": ans, "gold_spans": spans, "section": a["section"] + "+" + b["section"]}

def fill(state, stage, role, cands, quota, fn, key, llm, workers=4):
    """Process candidates in order until `quota` verified items exist. Deterministic: first `quota` ok items by candidate order."""
    outs, pos = {}, 0
    def run(c):
        k = key(c); st = state.get(stage, role, k)
        if st and st["status"] == "done": return k, st["out"]
        try:
            o = fn(c)
        except (L.QuotaExit, L.BudgetExceeded, L.ModelDrift): raise
        except Exception as e:   # transient LLM failure: counted, NOT recorded as done, so a resume retries it
            L.Ledger.log("failed", model=WRITER_MODEL, why=str(e)[:120]); return k, {"ok": False, "reason": "llm_error"}
        state.mark(stage, role, k, "done", o); return k, o
    while pos < len(cands):
        ok = sum(1 for c in cands[:pos] if outs.get(key(c), {}).get("ok"))
        if ok >= quota: break
        batch = cands[pos:pos + (quota - ok)]; pos += len(batch)
        with ThreadPoolExecutor(workers) as ex:
            for k, o in ex.map(run, batch): outs[k] = o
    state.flush()
    good = [(c, outs[key(c)]) for c in cands[:pos] if outs.get(key(c), {}).get("ok")][:quota]
    tried = [outs[key(c)] for c in cands[:pos]]
    return good, Counter(o.get("reason", "ok") for o in tried)

AREAS = ["computer vision and image generation", "speech, audio and music", "robotics and control", "game-playing and multi-agent reinforcement learning",
         "databases, systems and compilers", "bioinformatics, chemistry and medicine", "recommender systems and graph neural networks", "time-series forecasting, tabular ML and optimisation"]

def gen_unanswerable(llm, docs, state, quota):
    names = ", ".join(title(d) for d in sorted(docs))
    good, rej, seen = [], Counter(), set()
    for rnd, area in enumerate(AREAS):
        if len(good) >= quota: break
        st = state.get("s1", "unans", f"gen{rnd}")
        if st and st["status"] == "done": items = st["out"]
        else:
            try:
                j, _ = llm.json_chat(WRITER_MODEL, [{"role": "system", "content": SYS}, {"role": "user", "content": UNANS.format(titles=names, area=area)}], max_tokens=3000)
            except L.LLMError as e:    # transient: skip this round (not marked done; a resume retries it)
                L.Ledger.log("failed", model=WRITER_MODEL, why=str(e)[:120]); rej["round_failed"] += 1; continue
            items = j.get("questions", []); state.mark("s1", "unans", f"gen{rnd}", "done", items)
        for it in items:
            q, terms = str(it.get("question", "")).strip(), [str(t) for t in it.get("key_terms", [])]
            if len(q) < 25 or not terms or q in seen: rej["malformed_or_dup"] += 1; continue
            seen.add(q)
            if any(corpus_has_term(docs, t) for t in terms): rej["key_term_in_corpus"] += 1; continue
            good.append({"question": q, "key_terms": terms})
    return good[:quota], rej

def run(llm, state, limit=None, n_test_single=100, n_multi=20, n_dev=40, n_unans=20):
    if limit: n_test_single = n_dev = limit; n_multi = max(2, limit // 3); n_unans = limit
    L.set_stage("S1")
    docs = load_docs(); P = pools(docs); rows, report = [], {}
    pairs = pair_up(docs, P["multi"])
    plan = [("test", "single", P["test"], n_test_single, gen_single, lambda c: c["id"]),
            ("test", "multi", pairs, n_multi, gen_multi, lambda c: c[0]["id"] + "|" + c[1]["id"]),
            ("dev", "single", P["dev"], n_dev, gen_single, lambda c: c["id"])]
    for split, typ, cands, quota, fn, key in plan:
        good, rej = fill(state, "s1", f"{split}-{typ}", cands, quota, lambda c, fn=fn: fn(llm, docs, c), key, llm)
        for i, (c, o) in enumerate(good, 1):
            rows.append({"id": f"{split[0]}{typ[0]}{i:03d}", "split": split, "type": typ, "question": o["question"], "gold_answer": o["gold_answer"],
                         "gold_spans": o["gold_spans"], "section": o["section"], "writer_model": WRITER_MODEL})
        report[f"{split}-{typ}"] = {"requested": quota, "accepted": len(good), "candidates_tried": sum(rej.values()), "outcomes": dict(rej)}
    un, rej = gen_unanswerable(llm, docs, state, n_unans)
    for i, o in enumerate(un, 1):
        rows.append({"id": f"tu{i:03d}", "split": "test", "type": "unanswerable", "question": o["question"], "gold_answer": "UNANSWERABLE",
                     "gold_spans": [], "key_terms": o["key_terms"], "section": "-", "writer_model": WRITER_MODEL})
    report["test-unanswerable"] = {"requested": n_unans, "accepted": len(un), "rejected": dict(rej)}
    legacy = read_json(EVAL / "test_set.json", [])
    for i, r in enumerate(legacy, 1):
        rows.append({"id": f"dl{i:03d}", "split": "dev_legacy", "type": "legacy", "question": r["question"], "gold_answer": r["ground_truth"], "gold_spans": [],
                     "section": "-", "source_paper": r.get("source_paper", "")})
    state.flush()
    out = EVAL / ("questions.jsonl" if not limit else "cache/questions_dry.jsonl")
    write_jsonl(out, rows)
    cnt = Counter((r["split"], r["type"]) for r in rows)
    report["counts"] = {f"{a}/{b}": n for (a, b), n in cnt.items()}
    report["per_paper_test_single"] = dict(Counter(r["gold_spans"][0]["doc"] for r in rows if r["split"] == "test" and r["type"] == "single"))
    report["section_mix"] = dict(Counter(r["section"] for r in rows if r["type"] == "single"))
    report["eligible_passages"] = {k: len(v) for k, v in P.items()}; report["multi_pairs_available"] = len(pairs)
    write_json((RESULTS / "s1_questions_report.json") if not limit else (RESULTS / "dry" / "s1_questions_report.json"), report)
    return rows, report
