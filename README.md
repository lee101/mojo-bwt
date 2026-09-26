# mojo-bwt

`mojo_bwt` is the Burrows-Wheeler transform with the sorting and permutation
work in Mojo: the suffix sort that orders the rotations, the last column, the
LF-mapping that inverts it, and the run-length pass that real codecs put in
front of it.

```python
import mojo_bwt

last, primary = mojo_bwt.encode(b"banana")
last.tobytes()          # b'nnbaaa'
primary                 # 3
mojo_bwt.decode(last, primary)   # b'banana'
```

## What upstream this ports

There is no `bwt` distribution on PyPI — `https://pypi.org/pypi/bwt/json` is a
404 — so there is no upstream Python API to mirror. What the port covers is the
Burrows-Wheeler transform as it is actually specified and implemented, and the
one reference implementation available everywhere is **bzip2**, which is in the
standard library. The parity tests therefore run against real bzip2 output:
`tests/bz2ref.py` decodes a bzip2 block back to the last column and primary row
that bzip2 stored, and the Mojo kernel has to invert exactly those bytes.

The transform is the cyclic one bzip2 computes. The rows of the rotation matrix
of `T` are the suffixes of `T + T`, so `suffix_array` sorts the doubled text and
the rows below `n` are the rotations in sorted order.

## Covered subset

| area | implemented API |
| --- | --- |
| Suffix sorting | `suffix_array` — prefix doubling with radix sort over `T + T + [sentinel]` |
| Transform | `encode` → `(last_column, primary)`, `decode(last_column, primary)` |
| Transform, one stage at a time | `last_column(data, sa)` |
| LF-mapping | `lf_map(last_column)`, used by `decode` |
| Byte counting | `histogram` — 256-bin histogram, the first-column table the LF map needs |
| Run-length coding | `rle_encode` → `(values, lengths)`, `rle_decode(values, lengths)` |

Not implemented, and left to a real codec: the RLE1/RLE2 stages and the
Huffman/MTF stages of bzip2, block splitting, the randomised-block fallback
sort, and any streaming or multi-block framing. Those are compression framing
rather than the transform, and they belong with a codec, not with a transform
library. SA-IS (the linear-time suffix sort) is also not implemented; the sort
here is prefix doubling, which is `O(n log n)`.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-bwt.so`.

The `python/mojo_bwt` layer owns every array. It shifts the bytes into
`[1, 256]`, appends a unique smallest sentinel, doubles the text, and makes one
call per stage. Buffers cross the C ABI as 64-bit addresses and are rebuilt in
Mojo as `Pointer[Float64, ...]`-style handles, which keeps the exported symbols
non-parametric.

`decode` materialises the LF-mapping and then walks it. That is what bzip2's own
decoder does: the LF-mapping needs the rank of a row *among the occurrences of
its byte before that row*, which cannot be folded into a running counter during
the walk, so the permutation has to exist.

Everything here is exact integer work — byte values and index arithmetic — so
the parity tests use `assert_array_equal` and byte-exact comparisons rather
than tolerances. There is no floating point in these kernels, so the FMA
question does not arise.

## Install

The repository pins its own Mojo toolchain, `mojo ==1.2.0.dev2026092605`:

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

`build/build.sh` produces `dist/libmojo-bwt.so`. Set `PYTHONPATH=python` when
using the package outside a Pixi task. Do not run `pixi install`: the shared
environment at `/nvme0n1-disk/mojo-toolchain` is the only environment.

## Tests

`tests/test_sais.py` checks the suffix sort against a brute-force suffix sort
that materialises every suffix, for every length from 1 to 33 at three alphabet
sizes, plus medium sizes and the degenerate cases (all-equal, periodic, every
byte value including NUL). `tests/test_bwt.py` checks the transform against a
direct rotation sort, the LF-mapping against its definition, the run-length pass
against a Python run split, and the whole thing against real bzip2.

## Performance

Best of three wall clock, same process, over a 1 MiB input with a skewed byte
distribution. Every case verifies correctness against its reference before
timing. The suffix-sort baseline is the *same* prefix-doubling algorithm
written in NumPy, where the stable `int32` argsort is a radix sort, so the
comparison is like for like; the inverse baseline is the same algorithm in plain
Python, because a permutation walk has no vectorised form.

| case | reference | mojo-bwt | result |
| --- | ---: | ---: | ---: |
| histogram, 1 MiB | 6.63 ms | 1.15 ms | 5.75x faster |
| rle encode, 1 MiB | 260.57 ms | 17.98 ms | 14.49x faster |
| suffix sort, 1 MiB text (2 MiB doubled) | 42350.87 ms | 13070.72 ms | 3.24x faster |
| bwt encode, 1 MiB | 31214.77 ms | 10476.19 ms | 2.98x faster |
| bwt inverse, 1 MiB | 1677.16 ms | 312.14 ms | 5.37x faster |

Read those absolute numbers honestly: ten seconds to transform a megabyte is
slow in absolute terms, and it is slow because prefix doubling needs `log2(2n)`
rounds, each a full pass over the doubled text. The wins come from the compiled
inner loops beating Python and NumPy bookkeeping on the same algorithm, not from
a better algorithm. A real codec would use SA-IS and get the sort down to a
single linear pass; that is the obvious next step and it is not here.

Reproduce with:

```bash
python bench/bench.py
```

## License

MIT
