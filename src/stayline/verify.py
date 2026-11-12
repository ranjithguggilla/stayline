"""Compare two metrics.json files and fail if headline numbers drift."""

from __future__ import annotations

import json
import sys

KEYS = [
    ("test", "auc"),
    ("test", "log_loss"),
    ("test", "ece"),
    ("test", "lift_top20"),
    ("economics", "break_even_save_rate"),
    ("reasons", "lime_overlap_jaccard"),
]


def _get(m, path):
    for k in path:
        m = m[k]
    return m


def compare(a: dict, b: dict, tol: float = 0.01) -> list[str]:
    bad = []
    for path in KEYS:
        x, y = _get(a, path), _get(b, path)
        if abs(x - y) > tol * max(abs(x), 1e-9):
            bad.append(f"{'.'.join(path)}: {x} vs {y}")
    if a["selection"]["shipped"] != b["selection"]["shipped"]:
        bad.append("shipped model changed")
    return bad


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m stayline.verify committed.json fresh.json", file=sys.stderr)
        return 2
    a, b = (json.load(open(p)) for p in argv)
    bad = compare(a, b)
    for line in bad:
        print("drift:", line)
    print("metrics match" if not bad else "metrics drifted")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
