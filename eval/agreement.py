"""Human-check statistics: your labels vs (a) the question filter and (b) the LLM judge. No API calls.

Usage: python -m eval.agreement [--write]     (--write also saves results/human_check_agreement.txt)
Reads eval/human_check_questions.csv (valid / ambiguous / invalid) and eval/human_check_answers.csv (faithful / not faithful),
plus the hidden judge labels in results/raw/human_check_answers_key.csv. Labels are matched strictly: an unrecognised label stops the run.
"""
import csv, sys
from collections import Counter
from math import sqrt

from eval.common import EVAL, RESULTS, atomic_write, rawdir

Q_LABELS = {"valid", "ambiguous", "invalid"}
A_LABELS = {"faithful": True, "not faithful": False, "unfaithful": False}     # 'unfaithful' accepted as a synonym of 'not faithful'


def kappa(a, b):
    n = len(a)
    if n == 0: return None, None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / n ** 2
    return po, (None if pe == 1 else (po - pe) / (1 - pe))


def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def rows(path):
    with open(path, newline="") as f: return list(csv.reader(f))[1:]


def compute():
    """Return a dict with every human-check statistic (used by the report and by main())."""
    out = {}
    qr = rows(EVAL / "human_check_questions.csv"); qlab = {r[0]: r[-1].strip().lower() for r in qr if r[-1].strip()}
    bad = {i: l for i, l in qlab.items() if l not in Q_LABELS}
    if bad: raise ValueError(f"unrecognised question labels (allowed: {sorted(Q_LABELS)}): {bad}")
    if qlab:
        cnt = Counter(qlab.values()); n = len(qlab); v = cnt["valid"]; typ = {r[0]: r[1] for r in qr}
        human_valid = [qlab[i] == "valid" for i in qlab]
        po, k = kappa(human_valid, [True] * n)       # every released question passed the programmatic filter
        out["questions"] = {"n_rows": len(qr), "n_labelled": n, "counts": {k_: cnt.get(k_, 0) for k_ in ("valid", "ambiguous", "invalid")},
                            "valid_rate": v / n, "valid_ci95": wilson(v, n), "not_invalid_rate": (n - cnt["invalid"]) / n, "not_invalid_ci95": wilson(n - cnt["invalid"], n),
                            "invalid_rate": cnt["invalid"] / n, "invalid_ci95": wilson(cnt["invalid"], n), "filter_agreement": po, "filter_kappa": k,
                            "filter_kappa_note": "uninformative: the filter label is constant (it never rejects), so kappa is 0 by construction when labels vary",
                            "non_valid": {i: {"label": qlab[i], "type": typ[i]} for i in qlab if qlab[i] != "valid"}, "labels": qlab,
                            "by_type": {t: {"n": sum(typ[i] == t for i in qlab), **{l: sum(typ[i] == t and qlab[i] == l for i in qlab) for l in ("valid", "ambiguous", "invalid")}} for t in sorted(set(typ[i] for i in qlab))}}
    ar = rows(EVAL / "human_check_answers.csv"); alab = {}
    for r in ar:
        if r[-1].strip():
            l = r[-1].strip().lower()
            if l not in A_LABELS: raise ValueError(f"unrecognised answer label for {r[0]}: {r[-1]!r} (allowed: {sorted(A_LABELS)})")
            alab[r[0]] = A_LABELS[l]
    if alab:
        key = dict(rows(rawdir() / "human_check_answers_key.csv")); ids = list(alab)
        h = [alab[i] for i in ids]; j = [key[i] == "yes" for i in ids]; po, k = kappa(h, j); agree = sum(x == y for x, y in zip(h, j))
        lenient = [i for i in ids if key[i] == "yes" and not alab[i]]            # judge says faithful, human says not
        strict = [i for i in ids if key[i] != "yes" and alab[i]]                  # judge says not faithful, human says faithful
        out["answers"] = {"n_rows": len(ar), "n_labelled": len(ids), "human_faithful": sum(h), "human_not_faithful": len(h) - sum(h), "judge_faithful": sum(j), "judge_not_faithful": len(j) - sum(j),
                          "agreement": po, "agreement_count": agree, "agreement_ci95": wilson(agree, len(ids)), "kappa": k,
                          "judge_more_lenient_ids": lenient, "judge_stricter_ids": strict, "both_faithful": sum(x and y for x, y in zip(h, j)), "both_not_faithful": sum((not x) and (not y) for x, y in zip(h, j)),
                          "labels": {i: ("faithful" if alab[i] else "not faithful") for i in ids}, "judge_labels": {i: ("faithful" if key[i] == "yes" else "not faithful") for i in ids}}
    return out


def render(c):
    L = []
    if "questions" in c:
        q = c["questions"]; ci = lambda t: f"{t[0]:.0%}-{t[1]:.0%}"
        L.append(f"QUESTIONS (n={q['n_labelled']} of {q['n_rows']} labelled): valid {q['counts']['valid']}, ambiguous {q['counts']['ambiguous']}, invalid {q['counts']['invalid']}.")
        L.append(f"  valid {q['valid_rate']:.0%} (95% CI {ci(q['valid_ci95'])}); not invalid {q['not_invalid_rate']:.0%} (95% CI {ci(q['not_invalid_ci95'])}); invalid {q['invalid_rate']:.0%} (95% CI {ci(q['invalid_ci95'])}).")
        L.append(f"  Question filter vs human: percent agreement {q['filter_agreement']:.1%}; kappa {q['filter_kappa']:.3f} ({q['filter_kappa_note']}).")
        L.append("  By type: " + "; ".join(f"{t}: {v['valid']}/{v['n']} valid ({v['ambiguous']} ambiguous, {v['invalid']} invalid)" for t, v in q["by_type"].items()) + ".")
        L.append("  Not valid: " + ", ".join(f"{i} ({v['type']}: {v['label']})" for i, v in q["non_valid"].items()))
    if "answers" in c:
        a = c["answers"]; kap = "undefined" if a["kappa"] is None else f"{a['kappa']:.3f}"
        L.append(f"JUDGE FAITHFULNESS vs HUMAN (n={a['n_labelled']} of {a['n_rows']} labelled): percent agreement {a['agreement']:.1%} ({a['agreement_count']}/{a['n_labelled']}, 95% CI {a['agreement_ci95'][0]:.0%}-{a['agreement_ci95'][1]:.0%}); "
                 f"Cohen's kappa {kap}.")
        L.append(f"  Human: {a['human_faithful']} faithful / {a['human_not_faithful']} not faithful. Judge: {a['judge_faithful']} faithful / {a['judge_not_faithful']} not faithful. "
                 f"Both faithful {a['both_faithful']}, both not faithful {a['both_not_faithful']}.")
        L.append(f"  Judge more lenient than the human (judge faithful, human not) on {len(a['judge_more_lenient_ids'])}: {', '.join(a['judge_more_lenient_ids']) or '-'}; "
                 f"judge stricter (judge not faithful, human faithful) on {len(a['judge_stricter_ids'])}: {', '.join(a['judge_stricter_ids']) or '-'}.")
    if not c: L.append("No filled labels found yet.")
    return "\n".join(L)


def main():
    c = compute(); txt = render(c); print(txt)
    if "--write" in sys.argv: atomic_write(RESULTS / "human_check_agreement.txt", txt + "\n")


if __name__ == "__main__": main()
