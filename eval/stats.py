"""Percentile bootstrap CIs over questions (seeded), plain and paired."""
import numpy as np
from eval.common import SEED

B = 10000

def _idx(n, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))

def mean_ci(x):
    x = np.asarray(x, float)
    if len(x) == 0: return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = x[_idx(len(x))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n": len(x)}

def paired_diff(a, b):
    """a, b: per-question scores of two conditions on the SAME questions. Absolute diff (a-b) and relative diff ((mean_a-mean_b)/mean_b), each with 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float); n = len(a)
    if n == 0: return None
    ix = _idx(n); ma, mb = a[ix].mean(1), b[ix].mean(1); d = ma - mb
    out = {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "abs_diff": float(a.mean() - b.mean()),
           "abs_lo": float(np.percentile(d, 2.5)), "abs_hi": float(np.percentile(d, 97.5))}
    if b.mean() > 0:
        r = (d[mb > 0] / mb[mb > 0])
        out.update(rel_diff=float((a.mean() - b.mean()) / b.mean()), rel_lo=float(np.percentile(r, 2.5)), rel_hi=float(np.percentile(r, 97.5)))
    else:
        out.update(rel_diff=None, rel_lo=None, rel_hi=None)
    return out

def fmt(ci, pct=False, d=3):
    if ci is None or ci["mean"] is None: return "n/a"
    s = 100 if pct else 1
    return f"{ci['mean']*s:.{d if not pct else 1}f} [{ci['lo']*s:.{d if not pct else 1}f}, {ci['hi']*s:.{d if not pct else 1}f}]"
