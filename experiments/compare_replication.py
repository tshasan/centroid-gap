#!/usr/bin/env python3
"""Compare two result JSON files leaf by leaf and report every differing value.

python3 experiments/compare_replication.py PUBLISHED.json REPLICATED.json
Keys that hold wall-clock measurements (names containing "time", "seconds",
"latency", "qps", "_ms" or "_us") are listed separately as timings.
"""

import json
import re
import sys

TIMING = re.compile(r"time|second|latency|qps|_ms\b|_us\b|build|elapsed", re.I)


def leaves(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{path}/{key}")
    elif isinstance(node, list) and any(isinstance(x, (dict, list)) for x in node):
        for i, value in enumerate(node):
            yield from leaves(value, f"{path}[{i}]")
    else:
        yield path, node


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip())
    a = dict(leaves(json.load(open(sys.argv[1]))))
    b = dict(leaves(json.load(open(sys.argv[2]))))
    counts = {"same": 0, "diff": 0, "timing_same": 0, "timing_diff": 0}
    diffs = []
    for path in sorted(set(a) | set(b)):
        timing = bool(TIMING.search(path))
        same = path in a and path in b and a[path] == b[path]
        key = ("timing_" if timing else "") + ("same" if same else "diff")
        counts[key] += 1
        if not same and not timing:
            diffs.append((path, a.get(path, "<missing>"), b.get(path, "<missing>")))
    print(json.dumps(counts))
    for path, x, y in diffs:
        print(f"DIFF {path}: published={x!r} replicated={y!r}")


if __name__ == "__main__":
    main()
