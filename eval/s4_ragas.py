"""S4: RAGAS continuity check on the ORIGINAL 21 queries (legacy corpus), C1 vs C2 only; Context Precision + Faithfulness only.

The 21 queries were generated from the old index (20 templated stub PDFs + Attention Is All You Need), so they are run against that
corpus, not the 23 new papers. All RAGAS LLM calls go through the shared cached/budgeted client (judge model, temperature 0).
"""
import asyncio, faulthandler, json, time
import numpy as np
from langchain_core.outputs import Generation, LLMResult
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference
from ragas.run_config import RunConfig

from eval.common import ANSWER_MODEL, JUDGE_MODEL, RESULTS, rawdir, write_json
from eval import llm as L, stats as ST, corpus as C, s2_retrieval as R
from eval.s3_answers import answer_messages

README_CLAIM = {"dense_only_context_precision": 0.8690, "hybrid_context_precision": 0.9167, "abs_delta": 0.0477, "rel_delta_pct": 5.5, "source": "README.md 'Before/After Comparison' (added in commit 115e8b5, 2026-08-25)"}


class SharedLLM(BaseRagasLLM):
    """RAGAS LLM backed by eval.llm.LLM (cache, ledger, budget, quota exit). Always temperature 0, JSON mode."""
    def __init__(self, llm):
        super().__init__(); self.llm = llm; self.fatal = None
    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        try:
            r = self.llm.chat(JUDGE_MODEL, [{"role": "user", "content": prompt.to_string()}], max_tokens=1500, json_mode=True)
        except (L.QuotaExit, L.BudgetExceeded, L.ModelDrift) as e:   # ragas swallows job exceptions: remember it and re-raise after evaluate()
            self.fatal = e; raise
        return LLMResult(generations=[[Generation(text=r.text)]])
    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)
    def is_finished(self, response): return True


def run(llm, state, limit=None, extras=False):
    L.set_stage("S4")
    qs = [q for q in R.load_questions(limit) if q["split"] == "dev_legacy"]
    if limit: qs = qs[:limit]
    docs = C.load_docs("legacy")
    conds = [R.C1, R.C2] + ([R.C3] if extras else [])
    print(f"S4: {len(qs)} legacy queries on legacy corpus ({len(docs)} docs, {sum(len(d['text']) for d in docs.values())} chars), conditions {[c.name for c in conds]}")
    shared = SharedLLM(llm); out = {}
    for cond in conds:
        eng = R.engine(docs, "legacy", cond.size, cond.overlap, cond.embedder)
        qv = R.embed(cond.embedder, [q["question"] for q in qs])
        samples, meta = [], []
        for q, v in zip(qs, qv):
            chunks = R.retrieve(eng, q["question"], v, cond)
            res = llm.chat(ANSWER_MODEL, answer_messages(q["question"], chunks), max_tokens=150, extra={"reasoning_effort": "low"})
            samples.append(SingleTurnSample(user_input=q["question"], retrieved_contexts=[c["text"] for c in chunks], response=res.text.strip(), reference=q["gold_answer"]))
            meta.append({"qid": q["id"], "answer": res.text.strip(), "chunk_ids": [c["id"] for c in chunks]})
        cp, fa = LLMContextPrecisionWithReference(llm=shared), Faithfulness(llm=shared)
        for attempt in range(4):      # failed jobs (gateway 429/502) are retried; finished calls come from the cache, so a pass only repeats the missing ones
            faulthandler.dump_traceback_later(1200, exit=False)       # diagnose a hang instead of sitting silent
            result = evaluate(EvaluationDataset(samples), metrics=[cp, fa], llm=shared, run_config=RunConfig(max_workers=4, timeout=240, max_retries=1, max_wait=30),
                              raise_exceptions=False, show_progress=False)
            faulthandler.cancel_dump_traceback_later()
            if shared.fatal: raise shared.fatal
            df = result.to_pandas()
            if not df[["llm_context_precision_with_reference", "faithfulness"]].isna().any().any(): break
            print(f"  {cond.name}: pass {attempt + 1} left {int(df[['llm_context_precision_with_reference', 'faithfulness']].isna().any(axis=1).sum())} NaN rows; retrying from cache")
            time.sleep(60)
        for m, row in zip(meta, df.to_dict("records")):
            m["context_precision"] = None if row.get("llm_context_precision_with_reference") is None or np.isnan(row.get("llm_context_precision_with_reference")) else float(row["llm_context_precision_with_reference"])
            m["faithfulness"] = None if row.get("faithfulness") is None or np.isnan(row.get("faithfulness")) else float(row["faithfulness"])
        out[cond.name] = meta
        (rawdir(limit) / f"s4_{cond.name}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
        print(f"  {cond.name}: CP mean={np.nanmean([m['context_precision'] if m['context_precision'] is not None else np.nan for m in meta]):.4f} "
              f"faithfulness mean={np.nanmean([m['faithfulness'] if m['faithfulness'] is not None else np.nan for m in meta]):.4f} "
              f"(NaN/failed: CP={sum(m['context_precision'] is None for m in meta)}, F={sum(m['faithfulness'] is None for m in meta)})")
    summ = {"n_queries": len(qs), "corpus": "legacy (20 stub PDFs + Attention Is All You Need), production default config 512/128/MiniLM", "ragas_version": "0.4.3",
            "ragas_judge": JUDGE_MODEL, "answer_model": ANSWER_MODEL, "metrics": {}, "paired": {}, "readme_claim": README_CLAIM}
    for n, meta in out.items():
        summ["metrics"][n] = {k: ST.mean_ci([m[k] for m in meta if m[k] is not None]) for k in ("context_precision", "faithfulness")}
        summ["metrics"][n]["failed"] = {k: sum(m[k] is None for m in meta) for k in ("context_precision", "faithfulness")}
    for a, b in (("C2", "C1"),) + ((("C3", "C2"),) if extras else ()):
        ia = {m["qid"]: m for m in out[a]}; ib = {m["qid"]: m for m in out[b]}
        for k in ("context_precision", "faithfulness"):
            ids = [i for i in ia if ia[i][k] is not None and ib[i][k] is not None]
            summ["paired"][f"{a}-{b}|{k}"] = ST.paired_diff([ia[i][k] for i in ids], [ib[i][k] for i in ids])
    write_json(RESULTS / ("s4_ragas.json" if not limit else "dry/s4_ragas.json"), summ)
