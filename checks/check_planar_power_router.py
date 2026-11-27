"""Exact (Fraction) check of the planar power-router claim for tiny s.
Mirrors IndexRecall/PlanarFamily.lean. Run: python3 checks/check_planar_power_router.py"""
from fractions import Fraction as F

def run(s):
    n = 2 * s
    pitch = 10 * n
    height = (4 * pitch * n) ** 2
    delta = lambda t: F(t, n * n)
    drift = sum(delta(t) for t in range(n)) / n
    origin = lambda i: 3 * i * pitch
    labels = [(i, d) for i in range(s) for d in (False, True)]
    site = lambda l, t: (origin(l[0]) + (pitch if l[1] else 0) + delta(t),
                         1 if l[1] else (2 if t == 0 else 0))
    mean = {l: (sum(site(l, t)[0] for t in range(n)) / n, sum(site(l, t)[1] for t in range(n)) / n)
            for l in labels}
    q = lambda i: (origin(i) + drift + F(pitch, 3), height)
    d2 = lambda p, r: (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2
    # bias: M on lower rows, 0 on upper, M = max gap + 1
    M = max(d2(q(i), mean[(i, False)]) - d2(q(i), mean[(l, True)])
            for i in range(s) for l in range(s)) + 1
    w = {l: (0 if l[1] else M) for l in labels}
    for i in range(s):
        tip = (i, False)
        # all points: tip unique nearest
        pts = [(l, t) for l in labels for t in range(n)]
        best = min(pts, key=lambda a: d2(q(i), site(*a)))
        assert best == (tip, 0)
        assert sum(d2(q(i), site(*a)) == d2(q(i), site(tip, 0)) for a in pts) == 1
        rank0 = 1 + sum(d2(q(i), mean[l]) < d2(q(i), mean[tip]) for l in labels)
        sc = {l: d2(q(i), mean[l]) - w[l] for l in labels}
        rank1 = 1 + sum(sc[l] < sc[tip] for l in labels)
        assert rank0 == s + 1 and rank1 == 1, (s, i, rank0, rank1)
    print(f"s={s}: plain mean rank = {s+1}, power-router rank = 1 for all {s} queries; "
          f"work {n + (s+1)*n} -> {n + n}")

for s in (2, 3, 4):
    run(s)
