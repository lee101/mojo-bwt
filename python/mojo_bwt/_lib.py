"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory.  Every buffer crosses the C ABI as a
64-bit address, so the argtypes below must stay `c_int64` for addresses;
`c_int` truncates them and segfaults.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-bwt.so"

_ALPHA = 256


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(f"{_LIB_PATH} not found; run `bash build/build.sh` first")
    lib = ctypes.CDLL(str(_LIB_PATH))
    i64 = ctypes.c_int64
    i32 = ctypes.c_int32

    lib.bwt_suffix_array.restype = i32
    lib.bwt_suffix_array.argtypes = [i64] * 9

    lib.bwt_last_column.restype = i32
    lib.bwt_last_column.argtypes = [i64, i64, i64, i64, i64, i64]

    lib.bwt_histogram.restype = i32
    lib.bwt_histogram.argtypes = [i64, i64, i64]

    lib.bwt_lf_map.restype = i32
    lib.bwt_lf_map.argtypes = [i64, i64, i64, i64]

    lib.bwt_inverse.restype = i32
    lib.bwt_inverse.argtypes = [i64] * 6

    lib.bwt_rle_encode.restype = i32
    lib.bwt_rle_encode.argtypes = [i64, i64, i64, i64, i64]

    lib.bwt_rle_decode.restype = i32
    lib.bwt_rle_decode.argtypes = [i64, i64, i64, i64, i64]
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def _as_bytes(data) -> np.ndarray:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8)
    arr = np.asarray(data, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D byte sequence")
    return np.ascontiguousarray(arr)


def _sentinel_text(data: np.ndarray) -> np.ndarray:
    """Build the doubled, sentinel-terminated text the sort works on.

    The rows of the rotation matrix of `T` are the suffixes of `T + T`, so the
    sort input is the doubled string.  The sort needs a unique smallest symbol
    at the end and byte 0 is a legal input byte, so the alphabet is shifted by
    one."""
    n = int(data.size)
    text = np.empty(2 * n + 1, dtype=np.int32)
    text[0:n] = data
    text[n : 2 * n] = data
    np.add(text[: 2 * n], 1, out=text[: 2 * n])
    text[2 * n] = 0
    return text


def suffix_array(data) -> np.ndarray:
    """Suffix array of `T + T + [sentinel]`.

    Returns the `2n + 1` row indices.  The rows below `n` are the rotations of
    `T` in sorted order, which is what the transform reads off."""
    src = _as_bytes(data)
    n = int(src.size)
    text = _sentinel_text(src)
    n_total = int(text.size)
    sa = np.empty(n_total, dtype=np.int32)
    rank = np.empty(n_total, dtype=np.int32)
    nrank = np.empty(n_total, dtype=np.int32)
    shifted = np.empty(n_total, dtype=np.int32)
    cnt = np.empty(max(n_total, _ALPHA + 1), dtype=np.int32)
    rc = lib.bwt_suffix_array(
        _addr(text), n_total, _ALPHA, _addr(sa), _addr(rank), _addr(nrank),
        _addr(shifted), _addr(cnt), cnt.size,
    )
    if rc != 0:
        raise RuntimeError(f"bwt_suffix_array failed with code {rc}")
    return sa


def last_column(data, sa) -> tuple[np.ndarray, int]:
    """Last column of the cyclic transform plus the primary row index."""
    src = _as_bytes(data)
    n = int(src.size)
    text = _sentinel_text(src)
    sa = np.ascontiguousarray(sa, dtype=np.int32)
    if sa.size != text.size:
        raise ValueError(f"suffix array must hold {text.size} rows")
    lf = np.empty(max(n, 1), dtype=np.uint8)
    slot = np.empty(1, dtype=np.int32)
    rc = lib.bwt_last_column(
        _addr(text), _addr(sa), text.size, n, _addr(lf), _addr(slot)
    )
    if rc != 0:
        raise RuntimeError(f"bwt_last_column failed with code {rc}")
    return lf[:n], int(slot[0])


def encode(data) -> tuple[np.ndarray, int]:
    """Burrows-Wheeler transform of a byte string.

    Returns `(last_column, primary)`: the last column holds only real bytes and
    `primary` names the row whose rotation starts at byte 0."""
    src = _as_bytes(data)
    if src.size == 0:
        return np.empty(0, dtype=np.uint8), 0
    return last_column(src, suffix_array(src))


def lf_map(last_col) -> np.ndarray:
    """LF-mapping of a last column."""
    lf = _as_bytes(last_col)
    n = int(lf.size)
    out = np.empty(max(n, 1), dtype=np.int32)
    work = np.empty(_ALPHA, dtype=np.int32)
    rc = lib.bwt_lf_map(_addr(lf), n, _addr(out), _addr(work))
    if rc != 0:
        raise RuntimeError(f"bwt_lf_map failed with code {rc}")
    return out[:n]


def decode(last_col, primary: int) -> bytes:
    """Invert the transform by following the LF-mapping."""
    lf = _as_bytes(last_col)
    n = int(lf.size)
    out = np.empty(max(n, 1), dtype=np.uint8)
    work = np.empty(_ALPHA, dtype=np.int32)
    jump = np.empty(max(n, 1), dtype=np.int32)
    rc = lib.bwt_inverse(
        _addr(lf), n, int(primary), _addr(out), _addr(work), _addr(jump)
    )
    if rc != 0:
        raise ValueError(f"primary index {primary} outside 0..{max(n - 1, 0)}")
    return out[:n].tobytes()


def histogram(data) -> np.ndarray:
    """256-bin byte histogram."""
    src = _as_bytes(data)
    hist = np.empty(_ALPHA, dtype=np.int32)
    lib.bwt_histogram(_addr(src), src.size, _addr(hist))
    return hist


def rle_encode(data) -> tuple[np.ndarray, np.ndarray]:
    """Split a byte string into runs of equal bytes.

    Returns `(values, lengths)`, one entry per run."""
    src = _as_bytes(data)
    n = int(src.size)
    values = np.empty(max(n, 1), dtype=np.uint8)
    lengths = np.empty(max(n, 1), dtype=np.int32)
    slot = np.empty(1, dtype=np.int32)
    lib.bwt_rle_encode(_addr(src), n, _addr(values), _addr(lengths), _addr(slot))
    runs = int(slot[0])
    return values[:runs].copy(), lengths[:runs].copy()


def rle_decode(values, lengths) -> bytes:
    """Expand run-length-coded bytes."""
    vals = _as_bytes(values)
    lens = np.ascontiguousarray(lengths, dtype=np.int32)
    if vals.size != lens.size:
        raise ValueError("values and lengths must have the same length")
    total = int(lens.sum()) if lens.size else 0
    out = np.empty(max(total, 1), dtype=np.uint8)
    rc = lib.bwt_rle_decode(_addr(vals), _addr(lens), lens.size, _addr(out), out.size)
    if rc < 0:
        raise ValueError("runs do not fit in the output buffer")
    return out[:rc].tobytes()
