"""Post-hoc analyses on CACHED results only (no API calls).

1. EXPLORATORY: paired bootstrap, C2 with vs without the production source-diversity step. This comparison was added AFTER seeing the S2 results,
   so it is not part of the pre-planned analysis and carries no confirmatory weight.
2. eval/human_check_errors.csv: 10 random C2 top-5 misses for manual categorisation (the LLM's category is deliberately NOT shown).
"""
import csv, json, random
from eval.common import EVAL, RESULTS, RAW, SEED, read_jsonl, write_json
from eval import corpus as C, s2_retrieval as R, stats as ST

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

if __name__ == "__main__":
    diversity(); errors_sheet()
