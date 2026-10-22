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
    m = sum(d) / len(d)
    return (m, min(m, stats[int(0.025 * iters)]), max(m, stats[int(0.975 * iters) - 1]))


def bootstrap_mean(x: Sequence[float], iters: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not x:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    st = sorted(sum(x[rng.randrange(len(x))] for _ in x) / len(x) for _ in range(iters))
    m = sum(x) / len(x)
    return (m, min(m, st[int(0.025 * iters)]), max(m, st[int(0.975 * iters) - 1]))
