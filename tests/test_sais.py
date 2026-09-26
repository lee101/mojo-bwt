"""SA-IS suffix sorting, checked against a brute-force suffix sort.

The reference is deliberately naive: it materialises every suffix and sorts
them.  That is hopeless for large inputs and exactly right for the small and
medium sizes where a subtle bug in the type classification, the LMS naming, the
bucket arithmetic or the recursion base case would show up.
"""

import random

import numpy as np
import pytest

import mojo_bwt


def reference_suffix_array(data: bytes) -> np.ndarray:
    """Suffix array of `T + T + [sentinel]`, built by sorting suffixes."""
    n = len(data)
    seq = [b + 1 for b in data] * 2 + [0]
    order = sorted(range(len(seq)), key=lambda i: tuple(seq[i:]))
    return np.array(order, dtype=np.int32)


def _random_bytes(rng, n, alphabet):
    return bytes(rng.randrange(alphabet) for _ in range(n))


@pytest.mark.parametrize("alphabet", [2, 3, 256])
@pytest.mark.parametrize("n", list(range(1, 34)))
def test_suffix_array_exhaustive_small(n, alphabet):
    rng = random.Random(1000 * n + alphabet)
    for _ in range(3):
        data = _random_bytes(rng, n, alphabet)
        got = mojo_bwt.suffix_array(data)
        np.testing.assert_array_equal(got, reference_suffix_array(data))


@pytest.mark.parametrize("alphabet", [2, 4, 256])
@pytest.mark.parametrize("n", [64, 129, 257, 400, 1000])
def test_suffix_array_medium(n, alphabet):
    rng = random.Random(7919 * n + alphabet)
    data = _random_bytes(rng, n, alphabet)
    np.testing.assert_array_equal(
        mojo_bwt.suffix_array(data), reference_suffix_array(data)
    )


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"a",
        b"aa",
        b"aaa",
        b"aaaa",
        b"aaaaaaaaaaaaaaaaaaaaaaaa",
        b"ababababababababababab",
        bytes(range(256)),
        bytes(range(255, -1, -1)),
        b"\x00" * 300 + b"\x01" * 300,
    ],
)
def test_suffix_array_degenerate(data):
    """Runs, periodic strings and every byte value, including NUL."""
    np.testing.assert_array_equal(
        mojo_bwt.suffix_array(data), reference_suffix_array(data)
    )


def test_suffix_array_is_a_permutation():
    rng = random.Random(5)
    data = _random_bytes(rng, 777, 6)
    sa = mojo_bwt.suffix_array(data)
    assert sorted(sa.tolist()) == list(range(len(sa)))


def test_suffix_array_rows_below_n_are_the_rotations():
    """The rows the transform reads off are the sorted rotations of T."""
    rng = random.Random(11)
    data = _random_bytes(rng, 64, 3)
    n = len(data)
    sa = mojo_bwt.suffix_array(data)
    rotations = [data[r:] + data[:r] for r in range(n)]
    want = sorted(range(n), key=lambda r: rotations[r])
    assert [r for r in sa.tolist() if r < n] == want
