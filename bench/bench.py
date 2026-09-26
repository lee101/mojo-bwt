"""Correctness-gated benchmark for mojo-bwt.

Every case checks its result against an independent reference before timing, so
a regression in the kernels shows up as a correctness failure rather than a
suspiciously good number.  Baselines are the fastest reasonable formulation
available in the interpreter, not a Python loop NumPy would never use.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_bwt  # noqa: E402


def _time(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _corpus(n: int, seed: int = 0) -> bytes:
    """Text with a skewed byte distribution, as prose or source code has, so
    the sort is not a degenerate uniform case."""
    rng = np.random.default_rng(seed)
    weights = rng.random(256) ** 4 + 0.001
    weights /= weights.sum()
    return rng.choice(256, size=n, p=weights).astype(np.uint8).tobytes()


def numpy_suffix_array(data: bytes) -> np.ndarray:
    """Prefix doubling in NumPy: the same algorithm, vectorised.

    `argsort(kind="stable")` on an `int32` key is a radix sort inside NumPy, so
    each round is linear, exactly as the counting-sort round in the kernel is.
    This is the fair baseline for the sort: the same O(n log n) algorithm, not a
    Python loop that NumPy would never use."""
    src = np.frombuffer(data, dtype=np.uint8)
    n = src.size
    text = np.empty(2 * n + 1, dtype=np.int32)
    text[: 2 * n] = np.concatenate([src, src])
    text[: 2 * n] += 1
    text[2 * n] = 0
    m = text.size

    sa = np.argsort(text, kind="stable").astype(np.int32)
    fresh = np.empty(m, dtype=bool)
    fresh[0] = True
    fresh[1:] = text[sa[1:]] != text[sa[:-1]]
    rank = np.empty(m, dtype=np.int32)
    rank[sa] = np.cumsum(fresh, dtype=np.int32) - 1
    nclasses = int(rank.max()) + 1

    width = 1
    while nclasses < m and width < m:
        shifted = np.empty(m, dtype=np.int32)
        shifted[:width] = m - width + np.arange(width, dtype=np.int32)
        shifted[width:] = sa[sa >= width] - width
        order = np.argsort(rank[shifted], kind="stable")
        sa = shifted[order]
        first = rank[sa]
        second = np.where(sa + width < m, rank[np.minimum(sa + width, m - 1)], -1)
        fresh = np.empty(m, dtype=bool)
        fresh[0] = True
        fresh[1:] = (first[1:] != first[:-1]) | (second[1:] != second[:-1])
        rank = np.empty(m, dtype=np.int32)
        rank[sa] = np.cumsum(fresh, dtype=np.int32) - 1
        nclasses = int(rank.max()) + 1
        width *= 2
    return sa


def python_inverse(last: bytes, primary: int) -> bytes:
    """The bzip2-style inverse in plain Python: materialise the permutation,
    then walk it.  A permutation walk has no vectorised form, so this is the
    fair comparison for the inverse transform."""
    n = len(last)
    if n == 0:
        return b""
    counts = [0] * 256
    for c in last:
        counts[c] += 1
    start = []
    acc = 0
    for c in range(256):
        start.append(acc)
        acc += counts[c]
    jump = [0] * n
    seen = [0] * 256
    for j, c in enumerate(last):
        jump[j] = start[c] + seen[c]
        seen[c] += 1
    out = bytearray(n)
    j = primary
    for k in range(n):
        out[n - 1 - k] = last[j]
        j = jump[j]
    return bytes(out)


def python_rle(data: bytes):
    """Run-length split in plain Python; NumPy has no run-length primitive."""
    values = bytearray()
    lengths = []
    prev = None
    run = 0
    for c in data:
        if c == prev:
            run += 1
        else:
            if run:
                values.append(prev)
                lengths.append(run)
            prev = c
            run = 1
    if run:
        values.append(prev)
        lengths.append(run)
    return bytes(values), lengths


def bench_histogram(size=1 << 20):
    data = np.frombuffer(_corpus(size, 5), dtype=np.uint8)
    assert np.array_equal(mojo_bwt.histogram(data), np.bincount(data, minlength=256))
    return (
        f"histogram n={size}",
        _time(lambda: np.bincount(data, minlength=256)),
        _time(lambda: mojo_bwt.histogram(data)),
    )


def bench_rle(size=1 << 20):
    data = _corpus(size, 4)
    values, lengths = mojo_bwt.rle_encode(data)
    assert int(lengths.sum()) == len(data), "run lengths do not cover the input"
    assert mojo_bwt.rle_decode(values, lengths) == data, "rle round trip mismatch"
    ref_values, ref_lengths = python_rle(data)
    assert bytes(values) == ref_values and list(lengths) == ref_lengths
    return (
        f"rle encode n={size}",
        _time(lambda: python_rle(data)),
        _time(lambda: mojo_bwt.rle_encode(data)),
    )


def bench_suffix_sort(size=1 << 20):
    data = _corpus(size, 1)
    want = numpy_suffix_array(data)
    assert np.array_equal(mojo_bwt.suffix_array(data), want), "suffix array mismatch"
    return (
        f"suffix sort n={size}",
        _time(lambda: numpy_suffix_array(data)),
        _time(lambda: mojo_bwt.suffix_array(data)),
    )


def bench_encode(size=1 << 20):
    data = _corpus(size, 2)
    last, primary = mojo_bwt.encode(data)
    assert mojo_bwt.decode(last, primary) == data, "round trip mismatch"
    return (
        f"bwt encode n={size}",
        _time(lambda: numpy_suffix_array(data)),
        _time(lambda: mojo_bwt.encode(data)),
    )


def bench_inverse(size=1 << 20):
    data = _corpus(size, 3)
    last, primary = mojo_bwt.encode(data)
    raw = last.tobytes()
    assert mojo_bwt.decode(last, primary) == data, "inverse mismatch"
    return (
        f"bwt inverse n={size}",
        _time(lambda: python_inverse(raw, primary)),
        _time(lambda: mojo_bwt.decode(last, primary)),
    )


def main():
    print(f"{'case':<24}{'reference':>12}{'mojo-bwt':>12}{'ratio':>10}")
    print("-" * 58)
    for fn in (bench_histogram, bench_rle, bench_suffix_sort, bench_encode,
               bench_inverse):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<24}{ref * 1e3:>10.2f}ms{got * 1e3:>10.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
