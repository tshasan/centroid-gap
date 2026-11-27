"""Tiny exact checks for the one-bias-per-list results.
(1) cyclic-centroid affine router on the OneVector family: balanced loads, tie-free.
(2) discrete duality: for random integer scores and every integer load vector, brute-force
    the best assignment, then find a bias b (exact, via Bellman-Ford feasibility on the
    complementary-slackness difference constraints) that makes it argmax(a+b)."""
from fractions import Fraction as F
from itertools import product
import random

# (1)
for k in (3, 4, 5, 6):
    dim = k + 1
    e = lambda i: [1 if t == i else 0 for t in range(dim)]
    r = [[a - b for a, b in zip(e(j), e((j - 1) % k))] for j in range(k)]
    qs = {}
    for i in range(k):
        qs[(i, '+')] = e(i); qs[(i, '-')] = [-x for x in e(i)]
    load = [0] * k; hits = 0
    for (i, sg), q in qs.items():
        sc = [sum(x * y for x, y in zip(q, r[j])) for j in range(k)]
        top = [j for j in range(k) if sc[j] == max(sc)]
        assert len(top) == 1
        load[top[0]] += 1; hits += (top[0] == i)
    assert load == [2] * k and hits == k, (k, load, hits)
print("(1) balanced load (2,..,2) tie-free at b=0, recall@1 = 1/2, k=3..6")

# (2)
random.seed(1)
def dual_bias(a, assign, k):
    # need b with a[i][assign[i]] + b[assign[i]] >= a[i][j] + b[j] for all i,j.
    # b[j] - b[l] <= a[i][l] - a[i][j] where l=assign[i]; shortest-path potentials.
    INF = None
    d = [F(0)] * k
    for _ in range(k + 1):
        ch = False
        for i, l in enumerate(assign):
            for j in range(k):
                w = a[i][l] - a[i][j]   # b[j] <= b[l] + w
                if d[j] > d[l] + w: d[j] = d[l] + w; ch = True
        if not ch: return d
    return None  # negative cycle => assignment not optimal
for trial in range(30):
    n, k = 5, 3
    a = [[F(random.randint(-9, 9)) for _ in range(k)] for _ in range(n)]
    for load in product(range(n + 1), repeat=k):
        if sum(load) != n: continue
        best = None
        for assign in product(range(k), repeat=n):
            if [assign.count(j) for j in range(k)] != list(load): continue
            v = sum(a[i][assign[i]] for i in range(n))
            if best is None or v > best[0]: best = (v, assign)
        b = dual_bias(a, best[1], k)
        assert b is not None
        # every query's assigned list is a (possibly tied) argmax of a+b
        for i in range(n):
            s = [a[i][j] + b[j] for j in range(k)]
            assert s[best[1][i]] == max(s)
print("(2) every integer load vector realised as argmax(a+b) up to ties (30 random instances)")
