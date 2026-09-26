"""Burrows-Wheeler transform, LF-mapping, run-length pass and bzip2 parity.

Every check here is against an independently derived truth: a brute-force
rotation sort for the transform, the definition of the LF-mapping for the
inverse permutation, and real bzip2 output for the cross-implementation test.
The transform is exact integer work, so `assert_array_equal` is the right
assertion here -- no floating point, hence no FMA tolerance question.
"""

import bz2
import random

import numpy as np
import pytest

import bz2ref
import mojo_bwt


def reference_transform(data: bytes) -> tuple[bytes, int]:
    """Last column and primary row, by sorting the rotations directly."""
    n = len(data)
    if n == 0:
        return b"", 0
    order = sorted(range(n), key=lambda r: data[r:] + data[:r])
    last = bytes(data[(r - 1) % n] for r in order)
    return last, order.index(0)


def reference_lf_map(last: bytes) -> list[int]:
    counts = [0] * 256
    for c in last:
        counts[c] += 1
    starts = []
    acc = 0
    for c in range(256):
        starts.append(acc)
        acc += counts[c]
    seen = [0] * 256
    out = []
    for c in last:
        out.append(starts[c] + seen[c])
        seen[c] += 1
    return out


def random_bytes(rng, n, alphabet=256):
    return bytes(rng.randrange(alphabet) for _ in range(n))


# ---------------------------------------------------------------- transform


@pytest.mark.parametrize("n", list(range(1, 40)))
def test_last_column_matches_rotation_sort(n):
    rng = random.Random(31 * n)
    for _ in range(2):
        data = random_bytes(rng, n, rng.choice([2, 3, 7]))
        last, _ = reference_transform(data)
        got, primary = mojo_bwt.encode(data)
        assert got.tobytes() == last
        assert 0 <= primary < n


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"a",
        b"ab",
        b"aa",
        b"abab",
        b"banana",
        b"mississippi",
        b"abababababababab",
        b"\x00\x00\x00",
        bytes(range(256)),
        b"\xff" * 65,
    ],
)
def test_round_trip_edge_cases(data):
    last, primary = mojo_bwt.encode(data)
    assert mojo_bwt.decode(last, primary) == data


def test_round_trip_random():
    rng = random.Random(2024)
    for _ in range(40):
        n = rng.randrange(1, 300)
        data = random_bytes(rng, n, rng.choice([2, 4, 256]))
        last, primary = mojo_bwt.encode(data)
        assert mojo_bwt.decode(last, primary) == data


def test_last_column_entry_for_the_primary_row_is_the_last_byte():
    """The row that starts at byte 0 has no predecessor inside T.

    Its cyclic predecessor is the final byte of the input, and a last column
    that put anything else there would still round-trip, so only a direct
    comparison against the rotation sort catches it."""
    data = b"abcdefg"
    last, primary = mojo_bwt.encode(data)
    assert last[primary] == data[-1]


def test_encode_rejects_a_short_suffix_array():
    with pytest.raises(ValueError):
        mojo_bwt.last_column(b"abcdef", np.zeros(3, dtype=np.int32))


# ------------------------------------------------------------ LF mapping


def test_lf_map_matches_its_definition():
    rng = random.Random(77)
    for _ in range(20):
        data = random_bytes(rng, rng.randrange(1, 200), 5)
        last, _ = mojo_bwt.encode(data)
        raw = last.tobytes()
        np.testing.assert_array_equal(
            mojo_bwt.lf_map(last), np.array(reference_lf_map(raw), dtype=np.int32)
        )


def test_lf_map_is_a_permutation_that_steps_back_one_rotation():
    rng = random.Random(78)
    data = random_bytes(rng, 120, 4)
    n = len(data)
    last, _ = mojo_bwt.encode(data)
    lf = mojo_bwt.lf_map(last)
    assert sorted(lf.tolist()) == list(range(n))
    order = sorted(range(n), key=lambda r: data[r:] + data[:r])
    for row, r in enumerate(order):
        assert data[(r - 1) % n] == last[row]
        assert lf[row] == order.index((r - 1) % n)


def test_lf_map_covers_every_row_exactly_once():
    """Every byte string has a well-defined LF-mapping and it is always a
    permutation, so the kernel has nothing to reject.  This pins that down."""
    rng = random.Random(123)
    for _ in range(10):
        column = random_bytes(rng, rng.randrange(1, 200), rng.choice([2, 256]))
        lf = mojo_bwt.lf_map(column)
        assert sorted(lf.tolist()) == list(range(len(column)))


# ---------------------------------------------------------- run length pass


def test_rle_encode_matches_a_python_run_split():
    rng = random.Random(4)
    for _ in range(30):
        data = random_bytes(rng, rng.randrange(1, 400), rng.choice([2, 3, 256]))
        values, lengths = mojo_bwt.rle_encode(data)
        runs = list(_groupby(data))
        assert list(values) == [r[0] for r in runs]
        assert list(lengths) == [r[1] for r in runs]
        assert int(lengths.sum()) == len(data)
        assert mojo_bwt.rle_decode(values, lengths) == data


def _groupby(data):
    """Run-length split: list of `(value, length)` pairs."""
    out = []
    for c in data:
        if out and out[-1][0] == c:
            out[-1][1] += 1
        else:
            out.append([c, 1])
    return [(value, count) for value, count in out]


def test_rle_handles_a_single_long_run():
    data = b"z" * 5000
    values, lengths = mojo_bwt.rle_encode(data)
    assert values.tolist() == [ord("z")]
    assert lengths.tolist() == [5000]
    assert mojo_bwt.rle_decode(values, lengths) == data


def test_rle_of_empty_input():
    values, lengths = mojo_bwt.rle_encode(b"")
    assert values.size == 0 and lengths.size == 0
    assert mojo_bwt.rle_decode(values, lengths) == b""


def test_histogram_matches_bincount():
    rng = random.Random(6)
    data = random_bytes(rng, 5000, 256)
    np.testing.assert_array_equal(mojo_bwt.histogram(data), np.bincount(
        np.frombuffer(data, np.uint8), minlength=256
    ).astype(np.int32))


# ------------------------------------------------------------ bzip2 parity


def bzip2_column(data: bytes) -> tuple[bytes, int]:
    """bzip2's own last column and primary row for one block."""
    return bz2ref.last_column_of(bz2.compress(data))


@pytest.mark.parametrize(
    "data",
    [
        b"a",
        b"banana",
        b"mississippi",
        bytes(range(256)),
        b"abababababab" * 9,
        (b"the quick brown fox jumps over the lazy dog. " * 13)[:517],
    ],
)
def test_inverse_transform_recovers_the_bzip2_block(data):
    """bzip2's last column and origPtr, inverted by the Mojo kernel."""
    column, orig_ptr = bzip2_column(data)
    block = mojo_bwt.decode(column, orig_ptr)
    # The block is the input to bzip2's transform, so it must survive a
    # forward transform too: that closes the loop against bzip2's own bytes.
    last, _primary = mojo_bwt.encode(block)
    assert last.tobytes() == column
    assert bz2ref.rle1_decode(block) == data


def test_forward_transform_reproduces_the_bzip2_last_column():
    """The other direction: the block recovered from bzip2's own last column
    must transform back to that column byte for byte."""
    rng = random.Random(99)
    for _ in range(6):
        data = random_bytes(rng, rng.randrange(1, 400), rng.choice([4, 256]))
        column, orig_ptr = bzip2_column(data)
        last, _primary = mojo_bwt.encode(mojo_bwt.decode(column, orig_ptr))
        assert last.tobytes() == column


def test_whole_bzip2_pipeline_through_rle1():
    """Recover the original file bytes: bzip2 last column -> inverse BWT ->
    undo of bzip2's first run-length stage."""
    rng = random.Random(1234)
    for _ in range(5):
        data = random_bytes(rng, rng.randrange(1, 500), rng.choice([3, 256]))
        data += bytes([data[0]]) * rng.randrange(0, 40)
        column, orig_ptr = bzip2_column(data)
        assert bz2ref.rle1_decode(mojo_bwt.decode(column, orig_ptr)) == data
