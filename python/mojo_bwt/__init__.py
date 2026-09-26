"""Burrows-Wheeler transform with the sorting kernels in Mojo.

`mojo_bwt` exposes the compute-oriented surface of the transform: the suffix
sort that orders the rotations, the last column, the LF-mapping used to invert
it, and the run-length pass that real codecs (bzip2 among them) put in front of
it.  The package name keeps it importable next to any pure-Python fallback.
"""

from ._lib import (
    decode,
    encode,
    histogram,
    last_column,
    lf_map,
    rle_decode,
    rle_encode,
    suffix_array,
)

__all__ = [
    "decode",
    "encode",
    "histogram",
    "last_column",
    "lf_map",
    "rle_decode",
    "rle_encode",
    "suffix_array",
]
__version__ = "0.1.0"
