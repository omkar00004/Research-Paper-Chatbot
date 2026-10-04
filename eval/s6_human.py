"""S6: blank human-check sheets (20 questions, 20 C2 answers) + hidden judge key for agreement.py."""
import csv, json, random
from eval.common import EVAL, SEED, rawdir
from eval import corpus as C, s2_retrieval as R

def run(limit=None):
    qs = R.load_questions(limit); rng = random.Random(SEED + 6)
    ans = [q for q in qs if q["split"] == "test" and q["type"] in ("single", "multi")]
    n = 3 if limit else 20
    pick = rng.sample(ans, min(n, len(ans)))
    out = EVAL / ("human_check_questions.csv" if not limit else "cache/human_check_questions_dry.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "type", "question", "gold_answer", "gold_evidence", "human_label (valid / ambiguous / invalid)"])
        for q in pick: w.writerow([q["id"], q["type"], q["question"], q["gold_answer"], " || ".join(f"[{s['doc']}] {s['quote']}" for s in q["gold_spans"]), ""])
    raw = rawdir(limit) / "s3_C2.jsonl"
    if not raw.exists(): print("S6: no S3 C2 answers yet; questions sheet written, answers sheet skipped"); return
    rows = [json.loads(l) for l in raw.read_text().splitlines()]
    rng.shuffle(rows); rows = rows[:3 if limit else 20]
    eng = R.engine(C.load_docs(), "main", 512, 128, "all-MiniLM-L6-v2"); by = {c["id"]: c["text"] for c in eng.chunks}
    qmap = {q["id"]: q for q in qs}
    out = EVAL / ("human_check_answers.csv" if not limit else "cache/human_check_answers_dry.csv")
    with open(out, "w", newline="") as f, open(rawdir(limit) / "human_check_answers_key.csv", "w", newline="") as k:
        w, kw = csv.writer(f), csv.writer(k); w.writerow(["id", "question", "retrieved_context", "answer", "human_faithful (faithful / not faithful)"]); kw.writerow(["id", "judge_faithful"])
        for r in rows:
            w.writerow([r["qid"], qmap[r["qid"]]["question"], "\n---\n".join(by[c] for c in r["chunk_ids"]), r["answer"], ""]); kw.writerow([r["qid"], r["faithful"]])
    print(f"S6: wrote {out.parent.name}/human_check_*.csv")
