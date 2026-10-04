"""S4b (optional, off by default): RAGAS Context Precision + Faithfulness on a fixed 30-question subset of the S3 test subset
(single-passage only), C1 vs C2 only, frozen JUDGE_MODEL, through the shared cached client.

Runs only with --ragas-test, only after S0-S6 are complete and saved, and only if the cumulative call ledger is <= 80% of --max-calls.
RAGAS parse failures give NaN: they are never scored as 0, they are excluded from the means and counted per metric and condition;
a metric with more than 20% NaN in a condition is reported as unreliable.
"""
import math, random
import numpy as np

from eval.common import EVAL, JUDGE_MODEL, RESULTS, SEED, atomic_write, rawdir, read_json, write_json
from eval import llm as L, stats as ST, corpus as C, s2_retrieval as R, s3_answers as A
from eval.s4_ragas import SharedLLM

N_SUBSET = 30
NAN_LIMIT = 0.20
REQUIRED = ["s1_questions_report.json", "s2_dev_grid.json", "s2_retrieval.json", "s3_answers.json", "s4_ragas.json", "s5_errors.json", "summary.json"]
METRICS = {"context_precision": "llm_context_precision_with_reference", "faithfulness": "faithfulness"}


def attempts_so_far():
    """Cumulative HTTP attempts in the full-run ledger (successful calls + 429s + 5xx): what --max-calls counts."""
    return sum(1 for r in L.read_ledger() if r["run"] == "full" and r["kind"] in ("call", "429", "5xx"))


def check_ready(max_calls):
    """Return (ok, reason)."""
    if max_calls is None: return False, "S4b needs --max-calls (the 80% guard is defined against it)."
    miss = [f for f in REQUIRED if not (RESULTS / f).exists()] + [f for f in ("human_check_questions.csv", "human_check_answers.csv") if not (EVAL / f).exists()]
    if miss: return False, f"S0-S6 are not complete and saved; missing: {miss}"
    s3 = read_json(RESULTS / "s3_answers.json")
    bad = {c: m["failed_or_missing"] for c, m in s3["conditions"].items() if c in ("C1", "C2") and m["failed_or_missing"]}
    if bad: return False, f"S3 still has missing items {bad}; finish S3 first."
    used = attempts_so_far()
    if used > 0.8 * max_calls: return False, f"cumulative ledger = {used} HTTP attempts > 80% of --max-calls {max_calls} ({0.8 * max_calls:.0f}). Not running S4b."
    return True, f"ledger {used} attempts <= 80% of --max-calls {max_calls}"


def aggregate(per):
    """per[cond][metric] = list of floats / None (None = NaN). NaN is excluded from means and pairs, never treated as 0; >20% NaN => unreliable."""
    metrics, paired = {}, {}
    n = len(per["C1"][next(iter(METRICS))])
    for m in METRICS:
        unreliable = False
        for cond in ("C1", "C2"):
            vals = [v for v in per[cond][m] if v is not None]; nan = n - len(vals); bad = nan / n > NAN_LIMIT
            unreliable |= bad
            metrics.setdefault(cond, {})[m] = {**ST.mean_ci(vals), "nan_excluded": nan, "nan_fraction": nan / n, "unreliable": bad}
        ids = [i for i in range(n) if per["C1"][m][i] is not None and per["C2"][m][i] is not None]
        d = ST.paired_diff([per["C2"][m][i] for i in ids], [per["C1"][m][i] for i in ids]) if ids else None
        paired[m] = {"C2-C1": d, "n_pairs": len(ids), "unreliable": unreliable}
    return metrics, paired


def selftest():
    n = 30
    per = {"C1": {"context_precision": [1.0] * 24 + [None] * 6, "faithfulness": [0.5] * 30}, "C2": {"context_precision": [1.0] * 30, "faithfulness": [1.0] * 29 + [None]}}
    m, p = aggregate(per)
    assert m["C1"]["context_precision"]["mean"] == 1.0 and m["C1"]["context_precision"]["n"] == 24      # NaN excluded, not averaged in as 0
    assert m["C1"]["context_precision"]["nan_excluded"] == 6 and m["C1"]["context_precision"]["unreliable"] is False   # exactly 20% is not "more than 20%"
    per["C1"]["context_precision"][0] = None
    m, p = aggregate(per)
    assert m["C1"]["context_precision"]["unreliable"] and p["context_precision"]["unreliable"]          # 7/30 > 20%
    assert m["C2"]["faithfulness"]["nan_excluded"] == 1 and not m["C2"]["faithfulness"]["unreliable"] and p["faithfulness"]["n_pairs"] == 29
    print("s4b selftest ok")


def run(llm, state, max_calls):
    ok, why = check_ready(max_calls)
    print("S4b gate:", why)
    if not ok: raise SystemExit(2)
    llm.max_calls = max_calls - attempts_so_far()          # this stage may not push the total past --max-calls
    L.set_stage("S4b")
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference
    from ragas.run_config import RunConfig

    qs = {q["id"]: q for q in R.load_questions()}
    sub = [q for q in A.subset(list(qs.values()), None) if q["type"] == "single"]
    rng = random.Random(SEED + 4); rng.shuffle(sub)
    sub = sorted(sub[:N_SUBSET], key=lambda q: q["id"])
    assert len(sub) == N_SUBSET, f"S3 subset has only {len(sub)} single-passage questions"
    eng = R.engine(C.load_docs(), "main", 512, 128, "all-MiniLM-L6-v2"); text = {c["id"]: c["text"] for c in eng.chunks}
    shared = SharedLLM(llm); per = {}
    for cond in ("C1", "C2"):
        samples = []
        for q in sub:
            st = state.get("s3", cond, q["id"])
            if not st or st["status"] != "done": raise SystemExit(f"S3 item {cond}/{q['id']} missing")
            o = st["out"]; samples.append(SingleTurnSample(user_input=q["question"], retrieved_contexts=[text[i] for i in o["chunk_ids"]], response=o["answer"], reference=q["gold_answer"]))
        for attempt in range(2):          # second pass only repeats transient failures (finished calls are cached); deterministic parse failures stay NaN
            result = evaluate(EvaluationDataset(samples), metrics=[LLMContextPrecisionWithReference(llm=shared), Faithfulness(llm=shared)], llm=shared,
                              run_config=RunConfig(max_workers=4, timeout=240, max_retries=1, max_wait=30), raise_exceptions=False, show_progress=False)
            if shared.fatal: raise shared.fatal
            df = result.to_pandas()
            if not df[list(METRICS.values())].isna().any().any(): break
        per[cond] = {m: [None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in df[col]] for m, col in METRICS.items()}
        print(f"  {cond}: " + ", ".join(f"{m} NaN={sum(v is None for v in per[cond][m])}/{N_SUBSET}" for m in METRICS))

    out = {"n_subset": N_SUBSET, "question_ids": [q["id"] for q in sub], "judge": JUDGE_MODEL, "ragas_version": "0.4.3", "answers": "the S3 answers and retrieved contexts for C1 and C2 (no new answer calls)",
           "caveats": f"n={N_SUBSET} single-passage questions; the judge is a single LLM ({JUDGE_MODEL}, also the question writer); RAGAS metrics can return NaN on parse failures, which are excluded, never scored 0.",
           "metrics": {}, "paired": {}}
    out["metrics"], out["paired"] = aggregate(per)
    out["per_question"] = [{"qid": q["id"], **{f"{c}_{m}": per[c][m][i] for c in ("C1", "C2") for m in METRICS}} for i, q in enumerate(sub)]
    write_json(RESULTS / "s4b_ragas_test.json", out)
    md = ["# S4b: RAGAS on the test split (optional)\n", f"**n = {N_SUBSET} single-passage questions (fixed subset of the S3 test subset); the judge is a single LLM ({JUDGE_MODEL}).** Means with 95% bootstrap CIs; NaN samples are excluded (never scored as 0).\n",
          "| condition | metric | mean [95% CI] | n used | NaN excluded | status |", "|---|---|---|---|---|---|"]
    for cond, ms in out["metrics"].items():
        for m, v in ms.items():
            md.append(f"| {cond} | {m} | {ST.fmt(v, d=4) if v['mean'] is not None else 'n/a'} | {v['n']} | {v['nan_excluded']} | {'UNRELIABLE (>20% NaN)' if v['unreliable'] else 'ok'} |")
    md.append("\n## Paired difference C2 - C1 (same questions with both scores present)\n")
    for m, p in out["paired"].items():
        d = p["C2-C1"]
        if d is None: md.append(f"- {m}: n/a"); continue
        rel = "n/a" if d["rel_diff"] is None else f"{d['rel_diff'] * 100:+.1f}% [{d['rel_lo'] * 100:+.1f}%, {d['rel_hi'] * 100:+.1f}%]"
        md.append(f"- {m}{' (UNRELIABLE: a condition has >20% NaN)' if p['unreliable'] else ''}: C2 {d['mean_a']:.4f} vs C1 {d['mean_b']:.4f}; absolute {d['abs_diff']:+.4f} [{d['abs_lo']:+.4f}, {d['abs_hi']:+.4f}]; relative {rel}; n pairs = {p['n_pairs']}")
    atomic_write(RESULTS / "s4b_ragas_test.md", "\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv: selftest()
