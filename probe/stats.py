"""Small-sample statistics: Wilson intervals, paired bootstrap, exact McNemar."""
from __future__ import annotations

import math
import random
from typing import Sequence


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A right, B wrong) and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], iters: int = 10000, seed: int = 0,
                          clusters: Sequence | None = None) -> tuple[float, float, float]:
    """Mean(a-b) with a percentile CI. If clusters is given, resample clusters (e.g. source
    question across languages) instead of rows, which respects cross-language pairing."""
    assert len(a) == len(b)
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    if clusters is None:
        groups = [[v] for v in d]
    else:
        g: dict = {}
        for v, cl in zip(d, clusters):
            g.setdefault(cl, []).append(v)
        groups = list(g.values())
    stats = []
    for _ in range(iters):
        s = n = 0
        for _ in range(len(groups)):
            grp = groups[rng.randrange(len(groups))]
            s += sum(grp)
            n += len(grp)
        stats.append(s / n)
    stats.sort()
    # Plain percentile interval: it is NOT widened to contain the point estimate.
    return (sum(d) / len(d), stats[int(0.025 * iters)], stats[int(0.975 * iters) - 1])


def bootstrap_mean(x: Sequence[float], iters: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not x:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    st = sorted(sum(x[rng.randrange(len(x))] for _ in x) / len(x) for _ in range(iters))
    # Plain percentile interval: it is NOT widened to contain the point estimate.
    return (sum(x) / len(x), st[int(0.025 * iters)], st[int(0.975 * iters) - 1])


def signflip_cluster_p(a: Sequence[float], b: Sequence[float], clusters: Sequence, mc_iters: int = 200000,
                       seed: int = 0) -> float:
    """Two-sided sign-flip (randomization) test of mean(a-b)=0 that respects clustering:
    per-cluster sums of (a-b) are the exchangeable units (e.g. one MGSM question asked in
    sw, yo and zu is one cluster). Exact enumeration over the non-zero clusters when there
    are <= 22 of them, else Monte Carlo. Returns P(|T*| >= |T_obs|)."""
    sums: dict = {}
    for x, y, c in zip(a, b, clusters):
        sums[c] = sums.get(c, 0) + (x - y)
    v = [s for s in sums.values() if s != 0]
    if not v:
        return 1.0
    obs = abs(sum(v))
    eps = 1e-9
    if len(v) <= 22:
        hit = 0
        for mask in range(1 << len(v)):
            t = 0
            for i, s in enumerate(v):
                t += -s if (mask >> i) & 1 else s
            hit += abs(t) >= obs - eps
        return hit / (1 << len(v))
    rng = random.Random(seed)
    hit = sum(abs(sum(s if rng.random() < 0.5 else -s for s in v)) >= obs - eps for _ in range(mc_iters))
    return (hit + 1) / (mc_iters + 1)


def holm(p: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (FWER)."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def bh(p: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (FDR)."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i], reverse=True)
    adj = [0.0] * m
    run = 1.0
    for k, i in enumerate(order):
        rank = m - k
        run = min(run, p[i] * m / rank)
        adj[i] = min(1.0, run)
    return adj


def sig3(p: float) -> str:
    """p-value to three significant figures (e.g. 0.0490, 0.0938, 1.00)."""
    if p != p:
        return "nan"
    if p >= 0.9995:
        return "1.00"
    return f"{p:#.3g}" if p >= 0.001 else f"{p:.2e}"
