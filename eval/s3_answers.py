"""S3: answers + LLM judge on a fixed test subset. One answer call per (question, ordered context) and one judge call per answer, both content-hash cached."""
import json, os, random, time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from eval.common import ANSWER_MODEL, JUDGE_MODEL, RESULTS, SEED, rawdir, read_json, write_json
from eval import llm as L, stats as ST
from eval import s2_retrieval as R
from eval import corpus as C

ABSTAIN = "I cannot find sufficient information in the provided research papers to answer this question."
SUBSET = {"single": 45, "multi": 15, "unanswerable": 20}
JUDGE_RUBRIC = """You are a strict evaluator of a retrieval-augmented QA system. Judge ONLY against the materials below; do not use outside knowledge.

Question: {question}
Gold answer: {gold}
Gold evidence (verbatim from the source paper):
{evidence}

Retrieved context given to the system:
{context}

System answer: {answer}

Rules:
- "correct": "yes" if the system answer states the key facts of the gold answer and does not contradict it; "partial" if it captures only part of the key facts or is too vague; "no" if wrong, contradictory, off-topic, or if it abstains. If the gold answer is UNANSWERABLE (the corpus does not contain the answer): "yes" only if the system clearly declines / says the information is not available, otherwise "no".
- "faithful": "yes" if every factual claim in the system answer is supported by the retrieved context above; "no" if any claim is not supported by that context (even if it is true in the real world). An answer that only declines to answer is "yes".
- "abstained": "yes" if the system answer declines or says the context lacks the information (even partially), else "no".
- "rationale": ONE short line.
Reply with ONLY this JSON: {{"correct": "yes|partial|no", "faithful": "yes|no", "abstained": "yes|no", "rationale": "..."}}"""


def answer_messages(question, chunks):
    from backend.generation import SYSTEM_PROMPT, _build_context, _extract_sources
    if not chunks:        # C0: no retrieval
        return [{"role": "system", "content": "You are a research-paper question-answering assistant."}, {"role": "user", "content": f"Question: {question}\n\nAnswer in 1-3 sentences from your own knowledge."}]
    ctx = _build_context(chunks, _extract_sources(chunks))
    user = (f"Context from research papers:\n\n{ctx}\n\n---\n\nQuestion: {question}\n\n"
            f"Answer in 1-3 sentences using ONLY the context above. If the context does not contain the answer, reply exactly: \"{ABSTAIN}\"")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def ctx_text(chunks):
    return "\n---\n".join(f"[{i}] ({c['doc']}) {c['text']}" for i, c in enumerate(chunks, 1)) or "(none: no retrieval)"


def judge(llm, q, answer, chunks):
    ev = "\n".join(f"- ({s['doc']}) {s['quote']}" for s in q["gold_spans"]) or "(none: question is unanswerable from the corpus)"
    prompt = JUDGE_RUBRIC.format(question=q["question"], gold=q["gold_answer"], evidence=ev, context=ctx_text(chunks), answer=answer)
    j, _ = llm.json_chat(JUDGE_MODEL, [{"role": "user", "content": prompt}], max_tokens=900)
    out = {"correct": str(j.get("correct", "")).lower(), "faithful": str(j.get("faithful", "")).lower(), "abstained": str(j.get("abstained", "")).lower(), "rationale": str(j.get("rationale", ""))[:300]}
    if out["correct"] not in ("yes", "partial", "no") or out["faithful"] not in ("yes", "no") or out["abstained"] not in ("yes", "no"):
        raise L.LLMError(f"judge returned invalid labels: {out}")
    return out


class Tracer:
    """Langfuse tracing for S3 only. Never lets a tracing failure affect results."""
    def __init__(self):
        self.ok = False
        try:
            from langfuse import get_client
            if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
                self.c = get_client(); self.ok = True
        except Exception: pass
    def trace(self, cond, q, chunks, answer, res, judged):
        if not self.ok: return
        try:
            from langfuse import propagate_attributes
            with propagate_attributes(tags=["eval", "S3", f"cond:{cond}", f"type:{q['type']}"] + (["dry-run"] if L.RUN == "dry" else []), trace_name="s3_answer", metadata={"qid": q["id"], "cond": cond}):
                with self.c.start_as_current_observation(name="rag_answer", as_type="span", input={"question": q["question"]}, metadata={"chunk_ids": [c["id"] for c in chunks]}) as sp:
                    with self.c.start_as_current_observation(name="llm_answer", as_type="generation", model=ANSWER_MODEL, input=q["question"], output=answer, usage_details={"input": res.prompt_tokens, "output": res.completion_tokens}):
                        pass
                    sp.update(output={"answer": answer, "judge": judged})
                    self.c.score_current_trace(name="correct", value={"yes": 1.0, "partial": 0.5, "no": 0.0}[judged["correct"]])
                    self.c.score_current_trace(name="faithful", value=1.0 if judged["faithful"] == "yes" else 0.0)
        except Exception as e:
            self.ok = False; L.Ledger.log("langfuse_error", why=str(e)[:160])
    def flush(self):
        if self.ok:
            try: self.c.flush()
            except Exception: pass


def subset(qs, limit):
    rng = random.Random(SEED)
    out = []
    for typ, n in SUBSET.items():
        pool = [q for q in qs if q["split"] == "test" and q["type"] == typ]
        rng.shuffle(pool)
        n = {"single": 3, "multi": 1, "unanswerable": 1}[typ] if limit else n
        out += pool[:n]
    return out


def contexts(docs, cond, qs):
    """Final ranked chunks for each question (local; works for unanswerable questions too). C0 -> no chunks."""
    if cond.name == "C0": return {q["id"]: [] for q in qs}
    eng = R.engine(docs, "main", cond.size, cond.overlap, cond.embedder)
    qv = R.embed(cond.embedder, [q["question"] for q in qs])
    return {q["id"]: R.retrieve(eng, q["question"], v, cond) for q, v in zip(qs, qv)}


def conditions(extras):
    sel = read_json(RESULTS / "s2_dev_grid.json") or read_json(RESULTS / "dry" / "s2_dev_grid.json")
    if not sel: raise SystemExit("Run s2 first (needs the dev-selected config S).")
    S = R.Cond("S", **{k: v for k, v in sel["selected"].items() if k != "name"})
    conds = [R.C1, R.C2] + ([] if S.key() == R.C2.key() else [S])
    if extras: conds += [R.C3, R.Cond("C0", rerank=False, pool=0, top_k=0)]
    return conds, S.key() == R.C2.key()


def run(llm, state, limit=None, extras=False):
    L.set_stage("S3")
    qs = R.load_questions(limit); docs = C.load_docs()
    sub = subset(qs, limit); conds, s_is_c2 = conditions(extras)
    ctxs = {c.name: contexts(docs, c, sub) for c in conds}
    tr = Tracer(); print(f"S3: {len(sub)} questions x {len(conds)} conditions ({[c.name for c in conds]}); S identical to C2: {s_is_c2}; langfuse={'on' if tr.ok else 'off'}")
    work = [(c, q) for c in conds for q in sub if not state.done("s3", c.name, q["id"])]
    def one(cq):
        c, q = cq; chunks = ctxs[c.name][q["id"]]
        try:
            res = llm.chat(ANSWER_MODEL, answer_messages(q["question"], chunks), max_tokens=150, extra={"reasoning_effort": "low"})
            if not res.text.strip(): raise L.LLMError("empty answer")
            j = judge(llm, q, res.text.strip(), chunks)
        except (L.QuotaExit, L.BudgetExceeded, L.ModelDrift): raise
        except Exception as e:
            L.Ledger.log("failed", model=ANSWER_MODEL, why=str(e)[:150]); return
        out = {"qid": q["id"], "type": q["type"], "answer": res.text.strip(), "finish_reason": res.finish_reason, "length_retry": res.truncated_retry, "chunk_ids": [x["id"] for x in chunks], **j}
        state.mark("s3", c.name, q["id"], "done", out); tr.trace(c.name, q, chunks, out["answer"], res, j)
    with ThreadPoolExecutor(4) as ex: list(ex.map(one, work))
    state.flush(); tr.flush()
    summarize(state, sub, conds, s_is_c2, limit)


def summarize(state, sub, conds, s_is_c2, limit):
    rows = {c.name: {q["id"]: state.get("s3", c.name, q["id"])["out"] for q in sub if state.done("s3", c.name, q["id"])} for c in conds}
    types = {q["id"]: q["type"] for q in sub}
    def metrics(name):
        r = rows[name]; ans = [v for k, v in r.items() if types[k] != "unanswerable"]; un = [v for k, v in r.items() if types[k] == "unanswerable"]
        na = [v for v in ans if v["abstained"] == "no"]
        f = ST.mean_ci
        return {"n_answerable": len(ans), "n_unanswerable": len(un), "failed_or_missing": len(sub) - len(r),
                "correct_yes": f([v["correct"] == "yes" for v in ans]), "correct_yes_or_partial": f([v["correct"] in ("yes", "partial") for v in ans]),
                "correct_score_partial_half": f([{"yes": 1, "partial": .5, "no": 0}[v["correct"]] for v in ans]),
                "faithful_all_answers": f([v["faithful"] == "yes" for v in ans]), "faithful_among_non_abstained": f([v["faithful"] == "yes" for v in na]),
                "abstain_rate_answerable": f([v["abstained"] == "yes" for v in ans]),
                "abstain_rate_unanswerable": f([v["abstained"] == "yes" for v in un]), "hallucinated_answer_rate_unanswerable": f([v["abstained"] == "no" for v in un]),
                "length_truncated_answers": sum(v["finish_reason"] == "length" for v in r.values())}
    res = {c.name: metrics(c.name) for c in conds}
    def paired(a, b, fn, types_=None):
        ids = [i for i in rows[a] if i in rows[b] and (types_ is None or types[i] in types_)]
        return ST.paired_diff([fn(rows[a][i]) for i in ids], [fn(rows[b][i]) for i in ids])
    ans_t = ("single", "multi")
    pairs = [("C2", "C1")] + ([("S", "C2")] if not s_is_c2 else []) + ([("C3", "C2")] if "C3" in rows else [])
    diffs = {f"{a}-{b}": {"correct_yes(answerable)": paired(a, b, lambda v: v["correct"] == "yes", ans_t),
                          "faithful(answerable)": paired(a, b, lambda v: v["faithful"] == "yes", ans_t),
                          "hallucinated_answer(unanswerable)": paired(a, b, lambda v: v["abstained"] == "no", ("unanswerable",))} for a, b in pairs}
    out = {"models": {"answer": ANSWER_MODEL, "judge": JUDGE_MODEL}, "subset": {t: sum(1 for q in sub if q["type"] == t) for t in SUBSET}, "S_identical_to_C2": s_is_c2,
           "conditions": res, "paired_diffs": diffs, "definitions": {
               "correct_yes": "judge says 'yes' vs gold answer (partial counts as not correct); answerable questions only",
               "faithful": "judge says every claim is supported by the retrieved context",
               "abstain_rate_unanswerable": "share of unanswerable questions where the system declined",
               "hallucinated_answer_rate_unanswerable": "1 - abstain rate: the system gave a substantive answer to an unanswerable question"}}
    write_json(RESULTS / ("s3_answers.json" if not limit else "dry/s3_answers.json"), out)
    for name, r in rows.items():
        (rawdir(limit) / f"s3_{name}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in r.values()))
    for n, m in res.items():
        print(f"  {n:3s} answerable n={m['n_answerable']}: correct={m['correct_yes']['mean']} faithful={m['faithful_all_answers']['mean']} | unanswerable n={m['n_unanswerable']}: abstain={m['abstain_rate_unanswerable']['mean']}")
