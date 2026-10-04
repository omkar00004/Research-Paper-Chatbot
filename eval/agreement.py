"""Percent agreement and Cohen's kappa between your labels and (a) the question filter, (b) the LLM judge.

Usage: python -m eval.agreement   (after filling the last column of eval/human_check_questions.csv and eval/human_check_answers.csv)
"""
import csv, sys
from collections import Counter
from eval.common import EVAL, rawdir

def kappa(a, b):
    n = len(a)
    if n == 0: return None, None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / n ** 2
    return po, (None if pe == 1 else (po - pe) / (1 - pe))

def rows(path):
    with open(path, newline="") as f: return list(csv.reader(f))[1:]

def main():
    qf = [r for r in rows(EVAL / "human_check_questions.csv") if r[-1].strip()]
    if qf:
        human = [r[-1].strip().lower() == "valid" for r in qf]; filt = [True] * len(qf)       # every released question passed the filter
        po, k = kappa(human, filt)
        print(f"Question filter vs human (n={len(qf)}): human marked valid {sum(human)}/{len(qf)}; percent agreement={po:.1%}; "
              f"kappa={'undefined (the filter label is constant)' if k is None else f'{k:.3f}'}; counts={dict(Counter(r[-1].strip().lower() for r in qf))}")
    af = [r for r in rows(EVAL / "human_check_answers.csv") if r[-1].strip()]
    if af:
        key = dict(rows(rawdir() / "human_check_answers_key.csv"))
        h = ["faithful" if r[-1].strip().lower().startswith("faithful") else "not" for r in af]; j = ["faithful" if key[r[0]] == "yes" else "not" for r in af]
        po, k = kappa(h, j)
        print(f"Judge vs human faithfulness (n={len(af)}): percent agreement={po:.1%}; kappa={'undefined (a rater is constant)' if k is None else f'{k:.3f}'}; human={dict(Counter(h))} judge={dict(Counter(j))}")
    if not qf and not af: print("No filled labels found yet.")

if __name__ == "__main__": main()
