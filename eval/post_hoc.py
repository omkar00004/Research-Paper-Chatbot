"""Post-hoc analyses on CACHED results only (no API calls).

1. EXPLORATORY: paired bootstrap, C2 with vs without the production source-diversity step. This comparison was added AFTER seeing the S2 results,
   so it is not part of the pre-planned analysis and carries no confirmatory weight.
2. eval/human_check_errors.csv: 10 random C2 top-5 misses for manual categorisation (the LLM's category is deliberately NOT shown).
"""
import csv, json, random
from eval.common import EVAL, RESULTS, RAW, SEED, read_jsonl, write_json
from eval import agreement as AG, corpus as C, s2_retrieval as R, stats as ST

def load(name): return {r["qid"]: r for r in map(json.loads, (RAW / f"s2_{name}.jsonl").read_text().splitlines())}

def diversity():
    a, b = load("C2-nodiv"), load("C2")
    out = {"status": "EXPLORATORY (added after seeing results)", "difference": "no-diversity minus production C2 (positive = removing the diversity step helps)", "types": {}}
    md = ["# EXPLORATORY: C2 without vs with the source-diversity step\n",
          "Added after seeing the S2 results; not a pre-planned comparison. Difference = (C2 without diversity) - (production C2), same questions, paired 95% bootstrap CI (10,000 resamples). Absolute = metric points; relative = difference / production C2.\n",
          "| type | n | metric | no-diversity | production C2 | absolute diff [95% CI] | relative diff [95% CI] |", "|---|---|---|---|---|---|---|"]
    for typ in ("single", "multi"):
        ids = sorted(i for i in a if a[i]["type"] == typ); out["types"][typ] = {}
        for m in ("hit@5", "ndcg@5"):
            x = ST.paired_diff([a[i]["metrics"][m] for i in ids], [b[i]["metrics"][m] for i in ids]); out["types"][typ][m] = x
            rel = "n/a" if x["rel_diff"] is None else f"{x['rel_diff']*100:+.1f}% [{x['rel_lo']*100:+.1f}%, {x['rel_hi']*100:+.1f}%]"
            md.append(f"| {typ} | {x['n']} | {m} | {x['mean_a']:.3f} | {x['mean_b']:.3f} | {x['abs_diff']:+.3f} [{x['abs_lo']:+.3f}, {x['abs_hi']:+.3f}] | {rel} |")
    write_json(RESULTS / "s2_exploratory_diversity.json", out); (RESULTS / "s2_exploratory_diversity.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

def errors_sheet():
    qs = {q["id"]: q for q in read_jsonl(EVAL / "questions.jsonl")}
    dest = EVAL / "human_check_errors.csv"
    if dest.exists() and any(r[-1].strip() for r in list(csv.reader(open(dest, newline="")))[1:]):
        print("eval/human_check_errors.csv already has your labels: not overwriting."); return
    fails = [r for r in load("C2").values() if r["metrics"]["recall@5"] == 0]
    random.Random(SEED + 55).shuffle(fails); pick = sorted(fails[:10], key=lambda r: r["qid"])
    eng = R.engine(C.load_docs(), "main", 512, 128, "all-MiniLM-L6-v2"); chunk = {c["id"]: c for c in eng.chunks}
    with open(EVAL / "human_check_errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "gold_span", "top5_chunks", "your_category (chunk_boundary_split / vocabulary_mismatch / reranker_demoted / pdf_extraction_problem / multi_hop / other)"])
        for r in pick:
            q = qs[r["qid"]]
            w.writerow([q["id"], q["type"], q["question"], "\n".join(f"[{s['doc']}] {s['quote']}" for s in q["gold_spans"]),
                        "\n---\n".join(f"[{i}] ({chunk[c]['doc']}) {chunk[c]['text']}" for i, c in enumerate(r["ranked"][:5], 1)), ""])
    print(f"wrote eval/human_check_errors.csv: {len(pick)} of {len(fails)} C2 top-5 misses")

def validity_split():
    """EXPLORATORY (added after the human-check labels were revised): retrieval metrics on the 20 audited test questions, split by your validity label.
    The frozen test set is NOT edited; this only shows how much the audited questions that were not clearly valid move the numbers. n is tiny."""
    labels = AG.compute()["questions"]["labels"]; conds = ["C1", "C2", "C3", "S"]; data = {c: load(c) for c in conds}
    groups = {"valid": [i for i, l in labels.items() if l == "valid"], "ambiguous or invalid": [i for i, l in labels.items() if l != "valid"]}
    groups["not audited (reference)"] = sorted(set(data["C2"]) - set(labels))
    out = {"status": "EXPLORATORY, post hoc, tiny n; the frozen test question set was not edited", "groups": {}}
    md = ["# EXPLORATORY: retrieval metrics by human validity label\n",
          "Added after the human-check labels were revised. The frozen test set is unchanged and every headline number still uses all 120 answerable test questions. "
          "Groups are tiny and mix single- and multi-passage questions (multi-passage Hit@k needs ALL gold spans, so groups with more multi-passage questions score lower for that reason alone); treat this as a diagnostic, not evidence. 'not audited' = the other test questions whose validity nobody checked.\n",
          "| group | n (multi-passage) | cond | Hit@5 | nDCG@5 |", "|---|---|---|---|---|"]
    for g, ids in groups.items():
        nm = sum(data["C2"][i]["type"] == "multi" for i in ids); out["groups"][g] = {"n": len(ids), "n_multi": nm, "ids": ids, "metrics": {}}
        for c in conds:
            ci = {m: ST.mean_ci([data[c][i]["metrics"][m] for i in ids]) for m in ("hit@5", "ndcg@5")}; out["groups"][g]["metrics"][c] = ci
            md.append(f"| {g} | {len(ids)} ({nm}) | {c} | {ST.fmt(ci['hit@5'])} | {ST.fmt(ci['ndcg@5'])} |")
        out["groups"][g]["paired_C2_minus_C1"] = {m: ST.paired_diff([data["C2"][i]["metrics"][m] for i in ids], [data["C1"][i]["metrics"][m] for i in ids]) for m in ("hit@5", "ndcg@5")}
    md.append("\n## Paired C2 - C1 within each group (absolute metric points, 95% bootstrap CI)\n"); md += ["| group | n (multi-passage) | Hit@5 | nDCG@5 |", "|---|---|---|---|"]
    for g, v in out["groups"].items():
        p_ = v["paired_C2_minus_C1"]; f_ = lambda x: f"{x['abs_diff']:+.3f} [{x['abs_lo']:+.3f}, {x['abs_hi']:+.3f}]"
        md.append(f"| {g} | {v['n']} ({v['n_multi']}) | {f_(p_['hit@5'])} | {f_(p_['ndcg@5'])} |")
    write_json(RESULTS / "s2_exploratory_validity_split.json", out); (RESULTS / "s2_exploratory_validity_split.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("diversity", "all"): diversity()
    if what in ("errors", "all"): errors_sheet()
    if what in ("validity", "all"): validity_split()
