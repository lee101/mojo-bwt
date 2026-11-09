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


# --------------------------------------------------- exact primary parity
#
# `reference_transform` in this file is the rotation-sort definition, and
# `bz2ref.last_column_of` is real bzip2's stored (last column, origPtr).
# Both agree with the port's `primary` on every non-periodic input.  On
# periodic inputs the stable rotation sort returns `order.index(0)`, the
# *first* row among identical rotations, while the kernel (like bzip2)
# sorts the doubled text and so orders identical rotations by suffix
# length, returning the *last* such row.  For `aa` that is py-reference 0
# vs (port, bzip2) 1; for `aaaa` naive 0 vs (port alone) 3, and bzip2
# does not even store `aaaa`'s transform because its RLE1 stage fires
# (see `test_encode_of_four_equal_bytes_differs_from_bzip2_by_the_rle1_stage`).
# So `primary` is pinned against bzip2, not against the naive rotation
# sort, whenever the input is periodic.  `decode` round-trips under
# either convention, which is why a round-trip test alone cannot catch a
# wrong primary.


@pytest.mark.parametrize(
    "data",
    [
        b"a",
        b"ab",
        b"ba",
        b"banana",
        b"mississippi",
        b"\x00\x01\x02\x03",
        bytes(range(256)),
        b"the quick brown fox jumps over the lazy dog",
    ],
)
def test_encode_primary_and_column_match_bzip2_exactly(data):
    column, orig_ptr = bz2ref.last_column_of(bz2.compress(data))
    last, primary = mojo_bwt.encode(data)
    assert last.tobytes() == column
    assert primary == orig_ptr
    # exactness with an explicit tolerance, as for every integer output
    assert np.allclose(last, np.frombuffer(column, np.uint8), atol=0, rtol=0)
    assert primary == orig_ptr == pytest.approx(orig_ptr, abs=0)


@pytest.mark.parametrize(
    "data",
    [
        b"aa",
        b"abab",
        b"abababababababab",
        b"\x00\x00",
    ],
)
def test_encode_primary_follows_bzip2_not_the_naive_rotation_sort(data):
    """Periodic inputs: the kernel (and bzip2) sort the doubled text, so
    identical rotations are ordered by suffix length and the primary is the
    *last* row among the identical rotations.  The naive stable rotation
    sort gives the *first* such row, so `order.index(0)` is not the
    upstream primary here.  This pins the port to the bzip2 convention.
    The last column agrees under both conventions, and `decode` round-trips
    under either, so only the primary comparison catches the difference.
    (`aaaa` and other runs of four equal bytes are excluded here because
    bzip2's RLE1 stage fires; see the RLE1 test below.)"""
    column, orig_ptr = bz2ref.last_column_of(bz2.compress(data))
    _naive_last, naive_primary = reference_transform(data)
    last, primary = mojo_bwt.encode(data)
    assert last.tobytes() == column == _naive_last
    assert primary == orig_ptr
    # every case here is periodic, so the naive index is 0 and the port
    # (and bzip2) must be somewhere else
    assert naive_primary == 0 and primary != naive_primary
    assert mojo_bwt.decode(last, primary) == data


def test_encode_column_matches_rotation_sort_for_every_input():
    """The last column itself is unambiguous: the rotation sort and bzip2
    agree on it for every input, periodic ones included.  Check the column
    (byte-exact) against the rotation sort over the full small-input sweep
    that `test_last_column_matches_rotation_sort` already uses, plus the
    degenerate cases below."""
    rng = random.Random(31337)
    for n in list(range(1, 40)) + [64, 129]:
        for k in range(2):
            data = random_bytes(rng, n, [2, 3, 7, 256][(n + k) % 4])
            last, _ = reference_transform(data)
            got, _ = mojo_bwt.encode(data)
            assert got.tobytes() == last


def test_last_column_with_a_brute_force_suffix_array_matches_encode():
    """`last_column` reads its rotations off *any* suffix array of the
    doubled text, so a brute-force suffix array must give the same column
    and primary as `encode`, which builds its own suffix array.  A kernel
    that happened to match its own suffix array but not the doubled-text
    order would fail here against an independently written sort."""
    rng = random.Random(47)
    for _ in range(12):
        data = random_bytes(rng, rng.randrange(1, 200), rng.choice([2, 4, 256]))
        seq = [b + 1 for b in data] * 2 + [0]
        brute_sa = np.array(
            sorted(range(len(seq)), key=lambda i: tuple(seq[i:])), dtype=np.int32
        )
        want_last, _ = reference_transform(data)
        got, _ = mojo_bwt.last_column(data, brute_sa)
        assert got.tobytes() == want_last
    for data in [b"", b"a", b"aa", b"abab", b"\x00\x00\x00"]:
        seq = [b + 1 for b in data] * 2 + [0]
        brute_sa = np.array(
            sorted(range(len(seq)), key=lambda i: tuple(seq[i:])), dtype=np.int32
        )
        want_last, _ = reference_transform(data)
        got, _ = mojo_bwt.last_column(data, brute_sa)
        assert got.tobytes() == want_last


def test_last_column_matches_bzip2_column_with_bzip2_primary():
    """Feeding `last_column` the kernel's suffix array reproduces bzip2's
    column *and* primary for every input bzip2 stores verbatim (no run of 4+
    identical bytes, which is where bzip2 inserts its RLE1 stage and the
    column is of the RLE1-encoded block, not of the raw input)."""
    rng = random.Random(2025)
    for _ in range(10):
        n = rng.randrange(1, 200)
        data = bytearray(rng.randrange(256) for _ in range(n))
        for i in range(1, n):
            if data[i] == data[i - 1] == (data[i - 2] if i >= 2 else data[i - 1]):
                data[i] = (data[i] + 1) % 256
        data = bytes(data)
        if any(data.count(bytes([b])) >= 4 for b in set(data)):
            continue
        column, orig_ptr = bz2ref.last_column_of(bz2.compress(data))
        got, primary = mojo_bwt.last_column(data, mojo_bwt.suffix_array(data))
        assert got.tobytes() == column
        assert primary == orig_ptr


def test_decode_inverts_bzip2_columns_for_rle1_free_inputs():
    """For inputs with no run of four equal bytes bzip2's RLE1 stage is a
    no-op, so its stored last column *is* the Burrows-Wheeler transform of
    the raw input: the port must invert it back to the raw bytes and
    re-encode to the identical column."""
    rng = random.Random(4242)
    for _ in range(8):
        data = random_bytes(rng, rng.randrange(1, 300), rng.choice([3, 7, 256]))
        if b"aaaa" in data or b"\x00\x00\x00\x00" in data:
            continue
        column, orig_ptr = bz2ref.last_column_of(bz2.compress(data))
        assert bz2ref.rle1_decode(data) == data  # sanity: helper runs, input is RLE1-free
        block = mojo_bwt.decode(column, orig_ptr)
        assert bz2ref.rle1_decode(block) == data
        last, primary = mojo_bwt.encode(block)
        assert last.tobytes() == column
        assert primary == orig_ptr


def test_encode_empty_input_matches_rotation_sort():
    """Empty input: the rotation sort (and bzip2, which stores no block)
    has no rows; the port returns an empty column and primary 0."""
    assert reference_transform(b"") == (b"", 0)
    last, primary = mojo_bwt.encode(b"")
    assert last.tobytes() == b""
    assert primary == 0 == pytest.approx(0, abs=0)
    assert mojo_bwt.decode(last, primary) == b""


def test_last_column_of_empty_input_matches_encode():
    """`last_column` of an empty input must agree with `encode` on primary,
    not return the kernel's -1 placeholder."""
    last, primary = mojo_bwt.encode(b"")
    column, column_primary = mojo_bwt.last_column(
        b"", mojo_bwt.suffix_array(b"")
    )
    assert column.tobytes() == last.tobytes()
    assert column_primary == primary


def test_suffix_array_of_empty_input_is_the_sentinel_only():
    """Suffix array of T+T+[0] with T empty is the single sentinel row."""
    np.testing.assert_array_equal(
        mojo_bwt.suffix_array(b""), np.array([0], dtype=np.int32)
    )


def test_suffix_array_of_single_byte_matches_brute_force():
    seq = [98, 98, 0]
    want = np.array(sorted(range(3), key=lambda i: tuple(seq[i:])), dtype=np.int32)
    np.testing.assert_array_equal(mojo_bwt.suffix_array(b"a"), want)


def test_suffix_array_of_all_equal_bytes_matches_brute_force():
    """Identical suffixes are ordered by position, which is the convention
    bzip2's sort uses and the primary above depends on."""
    rng = random.Random(9)
    for data in [b"\x00" * 2, b"a" * 3, b"z" * 24, b"\x00" * 5 + b"\x01" * 5]:
        seq = [b + 1 for b in data] * 2 + [0]
        want = np.array(
            sorted(range(len(seq)), key=lambda i: tuple(seq[i:])), dtype=np.int32
        )
        np.testing.assert_array_equal(mojo_bwt.suffix_array(data), want)


def test_encode_of_all_equal_bytes_matches_bzip2():
    """Up to three identical bytes, bzip2's RLE1 stage never fires, so its
    stored column is the transform of the raw input.  At four or more
    identical bytes RLE1 does fire (see `test_whole_bzip2_pipeline_through_rle1`
    and the four-equal-bytes case below), so the direct bzip2 comparison
    is only valid below that threshold."""
    for data in [b"\x00\x00", b"aaa"]:
        column, orig_ptr = bz2ref.last_column_of(bz2.compress(data))
        last, primary = mojo_bwt.encode(data)
        assert last.tobytes() == column
        assert primary == orig_ptr
        assert mojo_bwt.decode(last, primary) == data


def test_encode_of_four_equal_bytes_differs_from_bzip2_by_the_rle1_stage():
    """bzip2 compresses runs of four equal bytes before the transform
    (the RLE1 stage undone by `bz2ref.rle1_decode`): four `z`s are stored
    as `z z z z \\x00`, so bzip2's column is the transform of that five-byte
    block, not of the raw input.  The port's `encode` is the bare
    Burrows-Wheeler transform, so it must differ from bzip2's column here.
    The bzip2 column still decodes -- through the port -- to a block whose
    own RLE1 decode is the raw input."""
    data = b"zzzz"
    column, orig_ptr = bz2ref.last_column_of(bz2.compress(data))
    last, primary = mojo_bwt.encode(data)
    assert last.tobytes() != column
    block = mojo_bwt.decode(column, orig_ptr)
    assert bz2ref.rle1_decode(block) == data
    re_column, re_primary = mojo_bwt.encode(block)
    assert re_column.tobytes() == column
    assert re_primary == orig_ptr


# --------------------------------------------------- degenerate inputs
#
# Upstream (bzip2's `bz2.compress`) accepts any bytes-like object, so its
# raw-buffer view of e.g. a NaN float64 is the 8 bytes
# `00 00 00 00 00 00 f8 7f` -- NaN and inf are ordinary bytes there, not
# errors.  The port's `_as_bytes` converts through `np.asarray(..., uint8)`
# instead: a *list* containing NaN is rejected by the conversion
# (`ValueError: cannot convert float NaN to integer`), while a float64
# array is *cast* element-wise (`1.5 -> 1`, NaN -> a platform byte with a
# RuntimeWarning) rather than viewed as its buffer.  Those are different
# input contracts from bzip2's buffer protocol, so the tests below pin
# the port's actual behaviour and the divergence itself, rather than
# expecting bzip2's TypeError for lists/ints or bzip2's bytes view for
# numeric arrays.  What both share is the *shape* contract: the port
# raises `ValueError` for a non-1-D input, like the rest of the numeric
# API in this file (`rle_decode` mismatched lengths, `decode` bad primary
# below).


def test_encode_rejects_non_1d_input():
    with pytest.raises(ValueError):
        mojo_bwt.encode(np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        mojo_bwt.suffix_array(np.zeros((2, 2), dtype=np.uint8))


def test_histogram_rejects_non_1d_input():
    with pytest.raises(ValueError):
        mojo_bwt.histogram(np.zeros((2, 2), dtype=np.uint8))


def test_lf_map_rejects_non_1d_input():
    with pytest.raises(ValueError):
        mojo_bwt.lf_map(np.zeros((2, 2), dtype=np.uint8))


def test_rle_encode_rejects_non_1d_input():
    with pytest.raises(ValueError):
        mojo_bwt.rle_encode(np.zeros((2, 2), dtype=np.uint8))


def test_decode_rejects_non_1d_input():
    with pytest.raises(ValueError):
        mojo_bwt.decode(np.zeros((2, 2), dtype=np.uint8), 0)


def test_encode_rejects_a_list_containing_nan():
    with pytest.raises(ValueError):
        mojo_bwt.encode([float("nan")])


def test_encode_of_a_nan_float64_array_matches_the_documented_cast():
    """Divergence from bzip2's buffer view, pinned here: the port casts the
    array to uint8 (NaN becomes a platform byte, with a RuntimeWarning)
    instead of viewing its 8 raw bytes.  The test asserts the cast value
    is what the transform actually ran on, so the divergence stays
    explicit rather than hidden."""
    with pytest.warns(RuntimeWarning):
        data = np.array([np.nan])
        cast = np.asarray(data, dtype=np.uint8)
        last, primary = mojo_bwt.encode(data)
    assert last.tobytes() == cast.tobytes()
    assert mojo_bwt.decode(last, primary) == cast.tobytes()


def test_encode_of_an_inf_array_matches_the_documented_cast():
    with pytest.warns(RuntimeWarning):
        data = np.array([np.inf])
        cast = np.asarray(data, dtype=np.uint8)
        last, primary = mojo_bwt.encode(data)
    assert last.tobytes() == cast.tobytes()
    assert mojo_bwt.decode(last, primary) == cast.tobytes()


def test_histogram_of_empty_input_is_all_zeros():
    np.testing.assert_array_equal(
        mojo_bwt.histogram(b""), np.zeros(256, dtype=np.int32)
    )
    assert np.allclose(mojo_bwt.histogram(b""), np.zeros(256), atol=0, rtol=0)


def test_histogram_of_single_byte_matches_bincount():
    np.testing.assert_array_equal(
        mojo_bwt.histogram(b"\x00"),
        np.bincount(np.frombuffer(b"\x00", np.uint8), minlength=256).astype(np.int32),
    )
    np.testing.assert_array_equal(
        mojo_bwt.histogram(b"\xff" * 4),
        np.bincount(np.frombuffer(b"\xff" * 4, np.uint8), minlength=256).astype(
            np.int32
        ),
    )


def test_lf_map_of_empty_input_is_empty():
    assert mojo_bwt.lf_map(b"").size == 0


def test_lf_map_of_single_byte_is_the_zero_row():
    np.testing.assert_array_equal(
        mojo_bwt.lf_map(b"a"), np.array([0], dtype=np.int32)
    )


def test_rle_encode_of_single_element_has_one_run_of_length_one():
    values, lengths = mojo_bwt.rle_encode(b"\x00")
    assert values.tolist() == [0]
    assert lengths.tolist() == [1]
    assert mojo_bwt.rle_decode(values, lengths) == b"\x00"


def test_rle_encode_of_all_equal_input_is_one_run():
    values, lengths = mojo_bwt.rle_encode(b"\x07" * 300)
    assert values.tolist() == [7]
    assert lengths.tolist() == [300]


def test_rle_encode_of_alternating_input_has_a_run_per_byte():
    values, lengths = mojo_bwt.rle_encode(b"ababab")
    assert values.tolist() == [97, 98, 97, 98, 97, 98]
    assert lengths.tolist() == [1] * 6


def test_rle_decode_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mojo_bwt.rle_decode(b"ab", np.array([1, 2, 3], dtype=np.int32))


def test_rle_decode_rejects_negative_lengths():
    with pytest.raises(ValueError):
        mojo_bwt.rle_decode(np.array([97], dtype=np.uint8), np.array([-1], dtype=np.int32))


def test_rle_decode_of_a_single_run_of_length_2_31_minus_1_expands():
    # the kernel has no upper bound on a single run's length: one run of
    # 2**31-1 bytes fits the (2**31-1)-byte output buffer it was handed,
    # so the wrapper returns that full expansion rather than an error.
    enc = mojo_bwt.rle_decode(
        np.array([97], dtype=np.uint8),
        np.array([2**31 - 1], dtype=np.int32),
    )
    assert len(enc) == 2**31 - 1 and enc[:4] == b"aaaa"


def test_rle_decode_rejects_nan_and_inf_lengths():
    """NaN/inf in the lengths array become out-of-range int32 when cast
    (with a RuntimeWarning), and the kernel rejects the resulting negative
    / overflowing length with ValueError, not a crash or a silent zero."""
    for bad in [float("nan"), float("inf")]:
        with pytest.warns(RuntimeWarning):
            with pytest.raises(ValueError):
                mojo_bwt.rle_decode(
                    np.array([97], dtype=np.uint8),
                    np.array([bad], dtype=np.float64),
                )


def test_decode_rejects_an_out_of_range_primary():
    with pytest.raises(ValueError):
        mojo_bwt.decode(b"banana", 6)
    with pytest.raises(ValueError):
        mojo_bwt.decode(b"banana", -1)


def test_decode_of_empty_column_with_nonzero_primary_is_empty():
    """The kernel's early return for n < 1 makes any primary on an empty
    column a no-op in the port; bzip2 never stores an empty block, so this
    pins the port's own convention rather than an upstream one."""
    assert mojo_bwt.decode(b"", 1) == b""
    assert mojo_bwt.decode(b"", 0) == b""


def test_histogram_is_exact_for_every_byte_value():
    data = bytes(range(256))
    hist = mojo_bwt.histogram(data)
    np.testing.assert_array_equal(hist, np.ones(256, dtype=np.int32))
    assert np.allclose(hist, np.ones(256), atol=0, rtol=0)


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
