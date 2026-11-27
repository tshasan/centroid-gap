#!/usr/bin/env python3
"""Reproduce the IVF examples with exact coordinates, scans and SSE searches."""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import chain
from math import lcm
from struct import Struct
from time import perf_counter


def mean(points):
    return tuple(Fraction(sum(axis), len(points)) for axis in zip(*points))


def distance_squared(first, second):
    return sum((a - b) ** 2 for a, b in zip(first, second))


def integer_distances(query, points):
    """Return squared distances multiplied by a common positive denominator."""
    scale = lcm(*(Fraction(value).denominator for value in query))
    scaled_query = tuple(int(value * scale) for value in query)
    return {
        point: distance_squared(scaled_query, tuple(value * scale for value in point))
        for point in points
    }


def exchange(original, swaps):
    lists = [row.copy() for row in original]
    for (a, first), (b, second) in swaps:
        lists[a][first], lists[b][second] = lists[b][second], lists[a][first]
    return lists


def check_layouts(original, swapped, rerouted, queries, ranks):
    """Check exact scans, all probe thresholds, routing ties, and storage."""
    k, width = len(original), len(original[0])
    points = set(chain.from_iterable(original))
    assert len(points) == k * width
    assert all(len(row) == width for row in chain(original, swapped))
    assert set(chain.from_iterable(swapped)) == points
    layouts = (
        (original, [mean(row) for row in original]),
        (swapped, [mean(row) for row in swapped]),
        (original, rerouted),
    )
    # The points were scaled by B². An extra factor of 4B recovers the
    # common denominator 4B³ for the original-unit representative coordinates.
    for _, representatives in layouts:
        assert len(representatives) == k
        assert all(
            (coordinate * 4 * width).denominator == 1
            for coordinate in chain.from_iterable(representatives)
        )

    work = []
    for query, expected_answer in queries:
        distances = integer_distances(query, points)
        answer = min(distances, key=distances.__getitem__)
        assert answer == expected_answer
        assert sum(d == distances[answer] for d in distances.values()) == 1
        for (lists, representatives), rank in zip(layouts, ranks):
            scores = [distance_squared(query, rep) for rep in representatives]
            assert len(set(scores)) == k
            order = sorted(range(k), key=scores.__getitem__)
            assert answer in lists[order[rank - 1]]

            best, evaluated = None, 0
            for probe, index in enumerate(order, 1):
                row = lists[index]
                row_best = min(row, key=distances.__getitem__)
                if best is None or distances[row_best] < distances[best]:
                    best = row_best
                evaluated += len(row)
                assert (best == answer) == (probe >= rank)
                if probe == rank:
                    assert evaluated == rank * width
                    work.append(k + evaluated)
    expected_work = tuple(k + rank * width for rank in ranks)
    assert tuple(work) == expected_work * len(queries)
    return expected_work


def scalar_case(k):
    width = (32 * k) ** 2 + 1
    scale = width**2
    original = [
        [(anchor * scale + slot,) for slot in range(width)]
        for anchor in [0, *range(6 * k + 1, 7 * k)]
    ]
    tip = (2 * k * scale,)
    original[0][-1] = tip
    swapped = exchange(original, [((0, -1), (1, -1))])
    rerouted = [mean(row) for row in original]
    rerouted[0] = (Fraction((8 * k - 1) * scale, 4),)
    queries = [
        ((Fraction((120 + step) * k * scale, 32),), tip)
        for step in range(5)
    ]
    work = check_layouts(original, swapped, rerouted, queries, (k, 1, 1))
    ratio = Fraction(work[0], work[1])
    assert k - Fraction(1, 1024) < ratio < k
    return original, scale, len(queries), work


def planar_layouts(k):
    scale = k**2
    pitch = 10 * k
    height = (4 * pitch * k) ** 2
    original = []
    queries = []
    for i in range(k // 2):
        origin = 3 * pitch * i
        low = [(origin * scale + t, 2 * scale if t == 0 else 0) for t in range(k)]
        high = [((origin + pitch) * scale + t, scale) for t in range(k)]
        original.extend((low, high))
        drift = mean(low)[0] - origin * scale
        query = (origin * scale + drift + Fraction(pitch * scale, 3), height * scale)
        queries.append((query, low[0]))

    swapped = exchange(original, [((i, 0), (i + 1, 0)) for i in range(0, k, 2)])
    rerouted = [mean(row) for row in original]
    for i in range(0, k, 2):
        rerouted[i] = (original[i][0][0], Fraction(3 * scale, 2))
    assert len({answer for _, answer in queries}) == len(queries)
    return original, swapped, rerouted, queries, scale


def planar_case(k):
    original, swapped, rerouted, queries, scale = planar_layouts(k)
    work = check_layouts(original, swapped, rerouted, queries, (k // 2 + 1, 1, 1))
    assert Fraction(work[0], work[1]) == 1 + Fraction(k, 4)
    return original, scale, len(queries), work


def check_query_perturbations(k):
    original, swapped, rerouted, queries, scale = planar_layouts(k)
    pitch = 10 * k
    horizontal = Fraction(pitch * scale, 24)
    vertical = Fraction((4 * pitch * k) ** 2 * scale, 4)
    corners = [
        ((x + dx, y + dy), answer)
        for (x, y), answer in queries
        for dx in (-horizontal, horizontal)
        for dy in (-vertical, vertical)
    ]
    check_layouts(original, swapped, rerouted, corners, (k // 2 + 1, 1, 1))
    return len(corners)


def planar_precision(k, rnd, subtract_common_term):
    """Count unique correct answers and strict, correctly ranked routing results.

    Round stored coordinates, query coordinates and every operation. Rounding
    the exact means directly isolates scoring from centroid training error.
    """
    original, swapped, rerouted, queries, scale = planar_layouts(k)

    def converted(point):
        return tuple(rnd(c / scale) for c in point)

    def score(q, p):
        dx = rnd(q[0] - p[0])
        if subtract_common_term:
            vertical = rnd(rnd(p[1] * p[1]) - rnd(rnd(2 * q[1]) * p[1]))
        else:
            dy = rnd(q[1] - p[1])
            vertical = rnd(dy * dy)
        return rnd(rnd(dx * dx) + vertical)

    points = [converted(p) for row in original for p in row]
    routers = [
        [converted(mean(row)) for row in original],
        [converted(mean(row)) for row in swapped],
        [converted(p) for p in rerouted],
    ]
    correct = strict = 0
    for i, (q, answer) in enumerate(queries):
        q = converted(q)
        distances = [score(q, p) for p in points]
        best = min(distances)
        correct += score(q, converted(answer)) == best and distances.count(best) == 1
        for reps, owner, rank in zip(routers, (2 * i, 2 * i + 1, 2 * i), (k // 2 + 1, 1, 1)):
            scores = [score(q, p) for p in reps]
            strict += len(set(scores)) == k and sum(x < scores[owner] for x in scores) + 1 == rank
    return correct, strict


def scalar_interval_sse(points):
    prefix_sum, prefix_squared = [0], [0]
    for point in points:
        prefix_sum.append(prefix_sum[-1] + point)
        prefix_squared.append(prefix_squared[-1] + point**2)

    def sse(start, stop):
        total = prefix_sum[stop] - prefix_sum[start]
        return prefix_squared[stop] - prefix_squared[start] - Fraction(total**2, stop - start)

    return sse


def scalar_all_cuts(original, scale):
    """Optimize 1/2 means by evaluating every contiguous split on the line."""
    points = sorted(point[0] for point in chain.from_iterable(original))
    n = len(points)
    sse = scalar_interval_sse(points)
    best, cuts = sse(0, n), []
    for cut in range(1, n):
        loss = sse(0, cut) + sse(cut, n)
        if loss < best:
            best, cuts = loss, [cut]
        elif loss == best:
            cuts.append(cut)

    # Optimal centers are distinct since one center costs strictly more.
    # Any optimum must assign to the nearer center, and a midpoint tie can
    # produce only one of two adjacent sorted splits for distinct points.
    assert best < sse(0, n)
    assert cuts == [len(original[0])]
    assert points[:cuts[0]] == [point[0] for point in original[0]]
    return best / scale**2, n


def scalar_split_costs(points, k):
    """Compute optimal SSE for sorted prefixes at each cluster count."""
    n = len(points)
    sse = scalar_interval_sse(points)
    layers = [None, [None, *(sse(0, stop) for stop in range(1, n + 1))]]
    transitions = n
    for clusters in range(2, k + 1):
        previous = layers[-1]
        current = [None] * (n + 1)

        def solve(first, last, first_cut, last_cut):
            nonlocal transitions
            if first > last:
                return
            stop = (first + last) // 2
            choices = range(first_cut, min(stop - 1, last_cut) + 1)
            transitions += len(choices)
            cost, cut = min((previous[cut] + sse(cut, stop), cut) for cut in choices)
            current[stop] = cost
            # The Monge inequality for SSE of sorted intervals makes the
            # leftmost minimizing cut nondecreasing as the prefix grows.
            solve(first, stop - 1, first_cut, cut)
            solve(stop + 1, last, cut, last_cut)

        solve(clusters, n, clusters - 1, n - 1)
        layers.append(current)
    return layers, transitions


def scalar_dp(original, scale):
    points = [point[0] for point in chain.from_iterable(original)]
    assert points == sorted(points)
    width = len(original[0])
    k = len(original)
    layers, transitions = scalar_split_costs(points, k)
    stop = len(points)
    best = layers[k][stop]
    # A strictly lower loss with k clusters rules out coincident or unused
    # optimal centers. With distinct centers, every optimal split is sorted.
    assert best < layers[k - 1][stop]
    sse = scalar_interval_sse(points)
    for clusters in range(k, 1, -1):
        # Check for ties over all cuts, including cuts outside the DP's
        # monotone search bounds for finding the leftmost minimizer.
        cuts = [
            cut for cut in range(clusters - 1, stop)
            if layers[clusters - 1][cut] + sse(cut, stop) == layers[clusters][stop]
        ]
        assert cuts == [width * (clusters - 1)]
        stop = cuts[0]
    return best / scale**2, transitions


@dataclass
class PartitionSearchState:
    """Explicit state machine for branch-and-bound exact partition search."""

    subset_cost: list
    max_clusters: int
    best: int
    winners: list = field(default_factory=list)
    considered: int = 0
    pruned: int = 0

    def search(self, remaining: int, chosen: tuple = (), loss: int = 0) -> None:
        if remaining == 0:
            if loss < self.best:
                self.best, self.winners = loss, []
            self.winners.append(chosen)
            return
        first = remaining & -remaining
        others = remaining ^ first
        submask = others
        while True:
            subset = submask | first
            next_loss = loss + self.subset_cost[subset]
            self.considered += 1
            if next_loss > self.best:
                self.pruned += 1
            else:
                self.search(remaining ^ subset, (*chosen, subset), next_loss)
            # With only one cluster left, it must take all remaining points.
            if submask == 0 or len(chosen) == self.max_clusters - 1:
                break
            submask = (submask - 1) & others


def planar_all_partitions(original, scale):
    """Search all unlabeled partitions into at most four nonempty clusters."""
    points = list(chain.from_iterable(original))
    n = len(points)
    assert n == 16
    denominator = lcm(*range(1, n + 1))
    subset_cost = [0] * (1 << n)
    sum_x = [0] * (1 << n)
    sum_y = [0] * (1 << n)
    sum_squared = [0] * (1 << n)
    for subset in range(1, 1 << n):
        first = subset & -subset
        rest = subset ^ first
        x, y = points[first.bit_length() - 1]
        sum_x[subset] = sum_x[rest] + x
        sum_y[subset] = sum_y[rest] + y
        sum_squared[subset] = sum_squared[rest] + x * x + y * y
        subset_cost[subset] = (
            sum_squared[subset] * denominator
            - (sum_x[subset] ** 2 + sum_y[subset] ** 2)
            * (denominator // subset.bit_count())
        )

    expected = tuple(0b1111 << (4 * i) for i in range(4))
    baseline = sum(subset_cost[subset] for subset in expected)
    state = PartitionSearchState(subset_cost=subset_cost, max_clusters=4, best=baseline)
    state.search((1 << n) - 1)
    assert state.best == baseline
    assert state.winners == [expected]
    return Fraction(state.best, denominator * scale**2), state.considered, state.pruned


def main():
    if not __debug__:
        raise SystemExit("Run without Python's -O option so the checks execute.")
    started = perf_counter()
    for label, cases, check in (
        ("scalar", (2, 3, 4, 8), scalar_case),
        ("planar", (4, 6, 16, 64, 128), planar_case),
    ):
        for k in cases:
            start = perf_counter()
            rows, scale, queries, work = check(k)
            print(
                f"{label} k={k} N={len(rows) * len(rows[0]):,} queries={queries} "
                f"work={work} ({perf_counter() - start:.2f}s)",
                flush=True,
            )
            if label == "scalar" and k == 2:
                loss, candidates = scalar_all_cuts(rows, scale)
                print(f"  unique optimum among {candidates:,} sorted partitions; SSE={loss}")
            if label == "scalar" and k == 3:
                start = perf_counter()
                loss, transitions = scalar_dp(rows, scale)
                print(
                    f"  unique optimum by sorted-prefix DP; SSE={loss}; "
                    f"{transitions:,} transitions ({perf_counter() - start:.2f}s)",
                    flush=True,
                )
            if label == "planar" and k == 4:
                loss, considered, pruned = planar_all_partitions(rows, scale)
                print(
                    f"  unique optimum; SSE={loss}; {considered:,} candidate clusters "
                    f"considered, {pruned:,} pruned"
                )
    print(f"planar k=16: {check_query_perturbations(16)} perturbed queries passed.")
    print(f"Exact checks passed in {perf_counter() - started:.2f}s.")
    binary32 = Struct("f")
    formats = (("binary32", lambda x: binary32.unpack(binary32.pack(x))[0]), ("binary64", float))
    for name, rnd in formats:
        for centered in (False, True):
            correct, strict = planar_precision(64, rnd, centered)
            expected = (0, 0) if name == "binary32" and not centered else (32, 96)
            assert (correct, strict) == expected
            method = "subtract common height squared" if centered else "direct squared distance"
            print(f"planar k=64 {name}, {method}: unique exact answers {correct}/32, "
                  f"tie-free routers with expected rank {strict}/96")


if __name__ == "__main__":
    main()
