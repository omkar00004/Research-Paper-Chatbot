"""Assemble summary.json, results.md, per_question.csv, dev_grid.md, error_analysis.md and print the PASTE-BACK SUMMARY."""
import csv, json, subprocess
from collections import Counter

from eval.common import EVAL, RESULTS, ANSWER_MODEL, JUDGE_MODEL, WRITER_MODEL, atomic_write, read_json, rawdir, write_json
from eval import llm as L
from eval.s2_retrieval import KS

HEAD = [f"{m}@{k}" for m in ("hit", "recall", "mrr", "ndcg") for k in KS]

def f(ci, d=3):
    return "n/a" if not ci or ci.get("mean") is None else f"{ci['mean']:.{d}f} [{ci['lo']:.{d}f}, {ci['hi']:.{d}f}]"

def pct(x, d=1): return "n/a" if x is None else f"{x * 100:+.{d}f}%"

def git(*a):
    try: return subprocess.check_output(["git", *a], cwd=EVAL.parent, text=True).strip()
    except Exception: return "?"

def families(m):
    return {"gpt-oss-20b": "OpenAI (gpt-oss)", "nemotron-3-super-120b": "NVIDIA (Nemotron)", "ministral-14b": "Mistral", "gemini-3.5-flash-lite": "Google (Gemini)"}.get(m, m)

def run(limit=None):
    d = RESULTS / ("dry" if limit else "")
    s1, grid, s2, s3, s4, s5 = (read_json(d / n) for n in ("s1_questions_report.json", "s2_dev_grid.json", "s2_retrieval.json", "s3_answers.json", "s4_ragas.json", "s5_errors.json"))
    frozen = read_json(L.FROZEN, {}); times = read_json(RESULTS / "raw" / "stage_times.json", [])
    led = L.ledger_table("dry" if limit else "full"); tot = Counter()
    for a in led.values():
        for k, v in a.items(): tot[k] += v
    runtime = sum(t["seconds"] for t in times if t["run"] == ("dry" if limit else "full"))
    S = {"s1": s1, "grid": grid, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "models": {"writer": WRITER_MODEL, "answer": ANSWER_MODEL, "judge": JUDGE_MODEL, "returned_strings": frozen},
         "ledger_totals": dict(tot), "runtime_s": runtime, "git": {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "eval", "backend", "results"))}}
    write_json(d / "summary.json", S)
    md = ["# Results\n", f"_git {S['git']['head'][:10]} on {S['git']['branch']}, dirty={S['git']['dirty']}_\n"]
    pq = []
    if s2:
        md.append("## S2 Retrieval (test split; mean [95% bootstrap CI over questions])\n")
        md.append(f"n single = {s2['n_test_single']}, n multi = {s2['n_test_multi']}. Multi-passage Hit@k = ALL gold spans in top-k; Recall@k = fraction of gold spans found. For single-passage questions Recall@k == Hit@k.\n")
        for typ in ("single", "multi"):
            md.append(f"\n### {typ}-passage\n"); md.append("| cond | " + " | ".join(HEAD) + " |"); md.append("|---|" + "---|" * len(HEAD))
            for c, v in s2["conditions"].items(): md.append(f"| {c} | " + " | ".join(f(v[typ].get(h)) for h in HEAD) + " |")
        md.append("\n### Config and latency (cold, ms/query, single-passage questions)\n"); md.append("| cond | config | chunks | retrieval ms (mean/p50/p95) | rerank ms (mean/p50/p95) |"); md.append("|---|---|---|---|---|")
        for c, v in s2["conditions"].items():
            lat = v["latency"]; fm = lambda x: "-" if not x else f"{x['mean_ms']:.0f}/{x['p50_ms']:.0f}/{x['p95_ms']:.0f}"
            md.append(f"| {c} | `{v['key']}` | {v['n_chunks']} | {fm(lat['retrieval'])} | {fm(lat['rerank'])} |")
        md.append("\n### Paired differences (A - B), same questions. ABSOLUTE = difference in the metric; RELATIVE = (mean_A - mean_B) / mean_B\n")
        md.append("| A-B | type | metric | mean A | mean B | absolute diff [95% CI] | relative diff [95% CI] |"); md.append("|---|---|---|---|---|---|---|")
        for key, ms in s2["paired_diffs"].items():
            pair, typ = key.split("|")
            for m in ("hit@1", "hit@5", "recall@10", "mrr@10", "ndcg@5", "ndcg@10"):
                x = ms.get(m)
                if x: md.append(f"| {pair} | {typ} | {m} | {x['mean_a']:.3f} | {x['mean_b']:.3f} | {x['abs_diff']:+.3f} [{x['abs_lo']:+.3f}, {x['abs_hi']:+.3f}] | {pct(x['rel_diff'])} [{pct(x['rel_lo'])}, {pct(x['rel_hi'])}] |")
        md.append(f"\nS identical to C2: {s2['S_identical_to_C2']}. Chroma parity (exact cosine vs Chroma HNSW, top-10 overlap): {s2['chroma_parity']}.\n")
    if grid:
        gm = ["# Dev grid (40 single-passage dev questions; selection metric nDCG@5)\n", "| config | chunks | " + " | ".join(HEAD) + " |", "|---|---|" + "---|" * len(HEAD)]
        for r in sorted(grid["grid"], key=lambda r: -r["ndcg@5"]):
            gm.append(f"| {r['name']} | {r['n_chunks']} | " + " | ".join(f"{r[h]:.3f}" for h in HEAD) + " |")
        gm.append(f"\nSelected S: `{grid['selected']}` with dev nDCG@5 = {grid['selected_dev_ndcg5']:.4f}. Production default (512/128/MiniLM/pool20/FlashRank) on dev, reference only: nDCG@5 = {grid['default_config_dev_reference']['ndcg@5']:.4f}.\n"
                   "Pool size has no effect when the re-ranker is off, so the 72-cell grid has 48 distinct cells.\n")
        atomic_write(d / "dev_grid.md", "\n".join(gm)); md.append(f"## Dev grid: see dev_grid.md. Selected S = {grid['selected']} (dev nDCG@5 {grid['selected_dev_ndcg5']:.4f}); n_dev = {grid['n_dev']}\n")
    if s3:
        md.append("## S3 Answers + judge\n"); md.append(f"answer={s3['models']['answer']}, judge={s3['models']['judge']}, subset={s3['subset']}. Definitions: {s3['definitions']}\n")
        md.append("| cond | n ans | correct (yes) | correct (yes+partial) | faithful (all answers) | faithful (non-abstained) | abstain on answerable | n unans | abstain on unanswerable | hallucinated answer on unanswerable | failed |"); md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for c, m in s3["conditions"].items():
            md.append(f"| {c} | {m['n_answerable']} | {f(m['correct_yes'])} | {f(m['correct_yes_or_partial'])} | {f(m['faithful_all_answers'])} | {f(m['faithful_among_non_abstained'])} | {f(m['abstain_rate_answerable'])} | {m['n_unanswerable']} | {f(m['abstain_rate_unanswerable'])} | {f(m['hallucinated_answer_rate_unanswerable'])} | {m['failed_or_missing']} |")
        md.append("\n| A-B | metric | mean A | mean B | absolute diff [95% CI] | relative diff [95% CI] | n |"); md.append("|---|---|---|---|---|---|---|")
        for pair, ms in s3["paired_diffs"].items():
            for m, x in ms.items():
                if x: md.append(f"| {pair} | {m} | {x['mean_a']:.3f} | {x['mean_b']:.3f} | {x['abs_diff']:+.3f} [{x['abs_lo']:+.3f}, {x['abs_hi']:+.3f}] | {pct(x['rel_diff'])} [{pct(x['rel_lo'])}, {pct(x['rel_hi'])}] | {x['n']} |")
    if s4:
        md.append("\n## S4 RAGAS continuity (original 21 queries, legacy corpus)\n"); md.append(f"judge={s4['ragas_judge']}, answer={s4['answer_model']}, ragas {s4['ragas_version']}, corpus: {s4['corpus']}\n")
        md.append("| cond | context precision | faithfulness | failed (CP, F) |"); md.append("|---|---|---|---|")
        for c, m in s4["metrics"].items(): md.append(f"| {c} | {f(m['context_precision'], 4)} | {f(m['faithfulness'], 4)} | {m['failed']} |")
        for k, x in s4["paired"].items():
            if x: md.append(f"\n{k}: mean A {x['mean_a']:.4f}, mean B {x['mean_b']:.4f}, absolute {x['abs_diff']:+.4f} [{x['abs_lo']:+.4f}, {x['abs_hi']:+.4f}], relative {pct(x['rel_diff'])} [{pct(x['rel_lo'])}, {pct(x['rel_hi'])}] (n={x['n']})")
        md.append(f"\nOriginal README claim: {s4['readme_claim']}\n")
    if s5:
        em = ["# Error analysis (C2: no gold span in top 5)\n", f"{s5['n_failures_c2_top5']} of {s5['n_answerable_test']} answerable test questions failed; {s5['n_analysed']} analysed (cap {s5['cap']}); judge {s5['judge']}.\n", "| category | count |", "|---|---|"]
        em += [f"| {k} | {v} |" for k, v in sorted(s5["counts"].items(), key=lambda kv: -kv[1])]
        em.append(f"\nProgrammatic pool facts: {s5['programmatic_pool_facts']}\n\n| qid | type | category | rationale | question |\n|---|---|---|---|---|")
        em += [f"| {r['qid']} | {r['type']} | {r['category']} | {r['rationale']} | {r['question']} |" for r in s5["items"]]
        atomic_write(d / "error_analysis.md", "\n".join(em)); md.append("\n## S5 Error analysis: see error_analysis.md\n")
    atomic_write(d / "results.md", "\n".join(md))
    # per-question CSV (S2 + S3)
    rows = []
    for c in (s2 or {}).get("conditions", {}):
        p = rawdir(limit) / f"s2_{c}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                r = json.loads(l); rows.append({"stage": "S2", "cond": c, "qid": r["qid"], "type": r["type"], "gold_rank": r["gold_rank"], **r["metrics"]})
    for c in (s3 or {}).get("conditions", {}):
        p = rawdir(limit) / f"s3_{c}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                r = json.loads(l); rows.append({"stage": "S3", "cond": c, "qid": r["qid"], "type": r["type"], "correct": r["correct"], "faithful": r["faithful"], "abstained": r["abstained"], "finish_reason": r["finish_reason"], "answer": r["answer"], "rationale": r["rationale"]})
    if rows:
        cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("stage", "cond", "qid", "type"), k))
        with open(d / "per_question.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    L.render_ledger_md(d / "call_ledger.md")
    paste(S, d)

def paste(S, d):
    s1, grid, s2, s3, s4, s5 = S["s1"], S["grid"], S["s2"], S["s3"], S["s4"], S["s5"]; fz = S["models"]["returned_strings"]
    o = ["", "=" * 25 + " PASTE-BACK SUMMARY " + "=" * 25]
    o.append(f"git: {S['git']['head']} (branch {S['git']['branch']}; uncommitted changes in eval/backend/results at report time: {S['git']['dirty']})")
    o.append("Models (requested -> returned string(s) seen): " + "; ".join(f"{role}={m} -> {fz.get(m, '?')}" for role, m in (("question-writer", WRITER_MODEL), ("answer", ANSWER_MODEL), ("judge", JUDGE_MODEL))))
    o.append(f"Families: writer={families(WRITER_MODEL)}, answer={families(ANSWER_MODEL)}, judge={families(JUDGE_MODEL)}. Judge and generator share a family: {families(JUDGE_MODEL) == families(ANSWER_MODEL)}. Writer and judge are the same model: {WRITER_MODEL == JUDGE_MODEL}.")
    if s2:
        cc = {(v['cond']['size'], v['cond']['overlap']): v['n_chunks'] for v in s2["conditions"].values()}
        o.append(f"Corpus: 23 PDFs (uploads/), ~2.09M characters of cleaned text. Chunks per config on the test run: {{{', '.join(f'{k[0]}/{k[1]}: {n}' for k, n in sorted(cc.items()))}}}; dev grid chunk counts in dev_grid.md.")
    if s1: o.append("Questions: " + ", ".join(f"{k}={n}" for k, n in sorted(s1["counts"].items())) + f". Writer outcomes: " + "; ".join(f"{k}: {v.get('outcomes', v.get('rejected'))}" for k, v in s1.items() if isinstance(v, dict) and k.startswith(('test-', 'dev-'))))
    if s2:
        o.append(f"\nRETRIEVAL (test; mean [95% CI]; n single={s2['n_test_single']}, n multi={s2['n_test_multi']}):")
        for typ in ("single", "multi"):
            for c, v in s2["conditions"].items():
                o.append(f"  {typ:6s} {c:8s} " + " ".join(f"{h}={f(v[typ][h])}" for h in HEAD))
        o.append("PAIRED DIFFERENCES (A-B; abs = metric points, rel = % of B):")
        for key, ms in s2["paired_diffs"].items():
            for m in ("hit@5", "recall@10", "mrr@10", "ndcg@5"):
                x = ms.get(m)
                if x: o.append(f"  {key:16s} {m:10s} {x['mean_a']:.3f} vs {x['mean_b']:.3f}  abs {x['abs_diff']:+.3f} [{x['abs_lo']:+.3f},{x['abs_hi']:+.3f}]  rel {pct(x['rel_diff'])} [{pct(x['rel_lo'])},{pct(x['rel_hi'])}]")
        o.append("Latency (ms/query mean; retrieval + rerank): " + "; ".join(f"{c}: {v['latency']['retrieval']['mean_ms']:.0f}+{(v['latency']['rerank'] or {'mean_ms': 0})['mean_ms']:.0f}" for c, v in s2["conditions"].items()))
    if grid: o.append(f"Dev-selected config S: {grid['selected']} dev nDCG@5={grid['selected_dev_ndcg5']:.4f} (n_dev={grid['n_dev']}); production default on dev nDCG@5={grid['default_config_dev_reference']['ndcg@5']:.4f}. S identical to C2: {s2['S_identical_to_C2'] if s2 else '?'}")
    if s3:
        o.append(f"\nANSWERS (subset {s3['subset']}):")
        for c, m in s3["conditions"].items():
            o.append(f"  {c}: correct(yes)={f(m['correct_yes'])} correct(yes|partial)={f(m['correct_yes_or_partial'])} faithful={f(m['faithful_all_answers'])} abstain@unanswerable={f(m['abstain_rate_unanswerable'])} hallucinated@unanswerable={f(m['hallucinated_answer_rate_unanswerable'])} failed={m['failed_or_missing']}")
        for pair, ms in s3["paired_diffs"].items():
            for m, x in ms.items():
                if x: o.append(f"  {pair} {m}: abs {x['abs_diff']:+.3f} [{x['abs_lo']:+.3f},{x['abs_hi']:+.3f}] rel {pct(x['rel_diff'])} (n={x['n']})")
    if s4:
        o.append(f"\nRAGAS (21 original queries, legacy corpus, judge {s4['ragas_judge']}):")
        for c, m in s4["metrics"].items(): o.append(f"  {c}: context_precision={f(m['context_precision'], 4)} faithfulness={f(m['faithfulness'], 4)} failed={m['failed']}")
        for k, x in s4["paired"].items():
            if x: o.append(f"  {k}: abs {x['abs_diff']:+.4f} [{x['abs_lo']:+.4f},{x['abs_hi']:+.4f}] rel {pct(x['rel_diff'])} [{pct(x['rel_lo'])},{pct(x['rel_hi'])}]")
        o.append("  Provenance of the old '+25%': not found in repo/git/logs. README table: CP 0.8690 -> 0.9167 (+0.0477 absolute, +5.5% relative), 'Dense-Only'=dense+FlashRank vs 'Hybrid'=hybrid+FlashRank, no saved run; the only saved run (2026-08-01) has all scores None; judge then = llama-3.3-70b-versatile (README era: qwen/qwen3-27b, inferred).")
    if s5: o.append(f"\nERROR ANALYSIS (C2 misses in top 5: {s5['n_failures_c2_top5']}/{s5['n_answerable_test']}; analysed {s5['n_analysed']}): {s5['counts']}; programmatic pool facts {s5['programmatic_pool_facts']}")
    t = S["ledger_totals"]
    o.append(f"\nLEDGER: HTTP calls={int(t.get('calls', 0))}, cache hits={int(t.get('cache_hits', 0))}, retries={int(t.get('retries', 0))}, 429s={int(t.get('http_429', 0))}, 5xx={int(t.get('http_5xx', 0))}, truncation retries={int(t.get('truncations', 0))}, failed={int(t.get('failed', 0))}, invalid JSON={int(t.get('invalid_json', 0))}, tokens in/out={int(t.get('prompt_tokens', 0))}/{int(t.get('completion_tokens', 0))}, cost=$0.00 (free tier), measured stage runtime={S['runtime_s']:.0f}s (excludes time waiting between resumed sessions)")
    ch = read_json(RESULTS / "s2_chroma_hit_check.json")
    if ch: o.append(f"CHROMA CHECK (C1, 512/128/MiniLM, n={ch['n']} single test questions): exact cosine vs Chroma default HNSW: Hit@1 {ch['exact']['hit@1']:.2f} vs {ch['chroma']['hit@1']:.2f}; Hit@5 {ch['exact']['hit@5']:.2f} vs {ch['chroma']['hit@5']:.2f}; Hit@10 {ch['exact']['hit@10']:.2f} vs {ch['chroma']['hit@10']:.2f}; nDCG@5 {ch['exact']['ndcg@5']:.3f} vs {ch['chroma']['ndcg@5']:.3f}. All S2 numbers use EXACT search, so they overstate the deployed dense retrieval by roughly this much; 25 chunk texts are duplicated in the index.")
    o.append("LIMITATIONS: see eval/README.md (LLM-written questions share vocabulary with gold passages; writer==judge model; small dev set (n=40) makes config selection noisy; chunk-size comparisons are confounded by context length; 28%-of-corpus thesis; PDF ligatures untouched; S2 uses exact search vs Chroma HNSW; judge/writer served partly via an ':free' host alias; unanswerable test is saturated at 100% abstention; human check pending unless filled).")
    o.append("=" * 70)
    txt = "\n".join(o); atomic_write(d / "paste_back.txt", txt); print(txt)
