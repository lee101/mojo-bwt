# VERIFY phase — what was reproduced, what was not, what changed

Date: 2026-11-09 session. Host: Linux 6.8.0 x86_64, this workspace.
Method: `pixi run build && pixi run test`, then `pixi run bench` sequentially;
every README row rerun in fresh processes; `bench/analysis.md` treated as
prior-phase output, not as ground truth.

## 1. Build / test / bench, reproduced

- `pixi run build && pixi run test`: green, **239 passed in 5.40s** (first
  run, before any edit). Matches the test count; no skips.
- `pixi run bench` (flock-wrapped `python bench/bench.py`): exit 0, five rows.
- README example checked by hand: `encode(b"banana")` → `b'nnbaaa'`, primary
  3, `decode(...) == b'banana'` — correct.

## 2. Benchmark table: reproduced numbers vs README (the catch)

The README table did **not** reproduce. Every figure was off by far more than
the 20% correction threshold, in both directions (the README both
understates and overstates the port). Five independent bench runs this
session (best-of-3 each, same process, 1 MiB skewed corpus):

| case | run-to-run spread (max−min over 5 runs) | README | measured (canonical run) | deviation of README |
| --- | ---: | ---: | ---: | ---: |
| histogram ref / mojo | 2.12–2.43 / 0.65–0.72 ms | 6.63 / 1.15 ms | 2.43 / 0.67 ms | ref +173%, mojo +72% |
| rle encode ref / mojo | 88.8–93.9 / 2.35–2.43 ms | 260.57 / 17.98 ms | 93.13 / 2.41 ms | ref +180%, mojo +646% |
| suffix sort ref / mojo | 8025–8395 / 1446–1668 ms | 42350.87 / 13070.72 ms | 8394.61 / 1560.59 ms | ref +405%, mojo +738% |
| bwt encode ref / mojo | 8070–8372 / 1562–2001 ms | 31214.77 / 10476.19 ms | 8372.46 / 1657.76 ms | ref +273%, mojo +532% |
| bwt inverse ref / mojo | 601–655 / 23.4–26.6 ms | 1677.16 / 312.14 ms | 607.77 / 26.36 ms | ref +176%, mojo +1084% |

Ratios: README claimed 5.75x / 14.49x / 3.24x / 2.98x / 5.37x; measured
3.63x / 38.64x / 5.38x / 5.05x / 23.06x. Run-to-run spread within the
session is ≤15% per cell (≤25% for one encode cell), so the README figures
cannot be explained as noise, a slower moment, or best-of-3 vs mean — they
are simply not what this code on this host produces. No code path in the
bench (`bench/bench.py` unchanged since the initial commit) or the kernels
explains a 5–10x gap; the README numbers were written from a different
(hypothetical or long-gone) run and never re-checked.

**Action:** all five absolute figures and ratios in the README table replaced
with the canonical measured run above, and the "ten seconds to transform a
megabyte" sentence corrected to "over a second and a half". No slower-than-
upstream row existed in the bench table and none was removed; the
slower-than-upstream fact (below) is a different comparison, documented
honestly in `bench/analysis.md` §5 and re-measured here.

## 3. Fairness: same input, no skipped work — checked

- **Same input within each row: yes.** Each `bench_*` builds one corpus
  (`seed` 5/4/1/2/3 per row) and times both sides on it. Between rows the
  corpus differs — that is fine, each row is self-contained.
- **Correctness-gated before timing: yes.** Every case asserts against its
  reference before `_time`. `_time` is best-of-3 wall clock (`perf_counter`),
  same process, no warmup — the pre-timing correctness call serves as warmup.
- **No skipped work on the Mojo side.** `python/mojo_bwt/_lib.py` allocates
  fresh buffers on every call (no cache/memo). Probes confirmed full work:
  histogram equals `bincount` on five seeds; RLE covers all 1 048 576 bytes
  (1 038 715 runs, round-trips); suffix array is a true permutation of
  2 097 153 rows and equals the NumPy prefix-doubling result; encode
  decomposes as suffix (1861 ms) + last-column (7.6 ms) ≈ encode (1853 ms);
  inverse output equals the input on the same `(last, primary)` both sides
  were given.
- **Suffix-sort baseline is like-for-like: yes.** NumPy side is the same
  prefix-doubling algorithm (stable `int32` argsort = radix per round); the
  Mojo kernel's one-counting-pass trick is a valid constant-factor
  optimisation (shifted SA is already ordered by the second key), not fewer
  rounds — both loop while `nclasses < m`. One asymmetry noted but immaterial:
  the "bwt encode" reference times `numpy_suffix_array` alone, omitting a
  NumPy last-column step. Measured here, a vectorised NumPy last column
  costs 23.8 ms at 1 MiB (0.29% of the sort), so the omission changes the
  encode ratio from 5.04x (sort-only) to 5.06x (full) — within noise. The
  ratio is not flattered. An alternative `python_rle`/inverse formulation
  exists (vectorised NumPy RLE: 20.5 ms vs python 93 ms; hybrid NumPy-LF +
  Python walk: 320 ms vs python 606 ms), which would lower the README ratios
  to ~8.7x and ~12.7x respectively — still "faster". The bench's choice of
  plain-Python baselines is disclosed in the README, so it is graded as a
  disclosed comparison, not a hidden one. No bench code was changed in this
  phase.

**Slower than upstream — kept, not hidden.** Against real bzip2 (the upstream
implementation; there is no `bwt` on PyPI), the port's BWT-only `encode`
takes 1672 ms vs bzip2's full-codec `compress` 148 ms at 1 MiB — **11.3x
slower than upstream**, even though bzip2 includes MTF/Huffman/RLE1. At
64 KiB: 53 ms vs 8.6 ms, 6.2x slower. (Mojo `decode` alone beats bzip2
`decompress`, 24 vs 82 ms, because decompress pays for the whole codec.)
This is the prefix-doubling `O(n log n)` sort vs bzip2's C `O(n log n)` sort
at ~10x per byte; the README already says the algorithm is not better, and
`bench/analysis.md` §5 keeps this row. Nothing was reworded to soften it.

## 4. Parity with upstream: checked `git log -p -- tests/`

- History is 3 code commits: initial port, `phase test` (ebe8003), `phase
  analyze` (analysis.md only). The `phase test` commit **removed zero
  assertions** and **loosened zero tolerances**: `git diff 30dc556..HEAD --
  tests/` shows no deleted lines. It added 410 lines of exact parity tests
  (byte-exact `==`, `assert_array_equal`, `atol=0/rtol=0`, `approx(abs=0)` —
  correct for exact integer work; there is no floating point in the kernels).
- What it changed in `src/kernels.mojo` is a genuine bug fix, verified here:
  `bwt_last_column` on empty input returned primary −1; it now returns 0,
  matching `encode(b"")`. Reproduced: `last_column(b"", suffix_array(b""))`
  → `(empty, 0)`.
- One weakened assertion found (added, not removed, in that commit):
  `assert bz2ref.rle1_decode(data) == data or True` in
  `test_decode_inverts_bzip2_columns_for_rle1_free_inputs` — the `or True`
  makes it vacuous. The equality is in fact true here (the input is RLE1-free
  by construction, and all 8 generated cases satisfy it), so the assertion
  was restored to a real check rather than deleted. Fail-first not applicable:
  the assertion already passed; the change only removes the vacuity.
- Independently re-verified this session: 50 random round-trips; suffix array
  vs brute-force at n=500/1000; periodic inputs (`aa`, `aaaa`, `abab`,
  `xyzxyz`) keep last column exact with primary following the bzip2
  doubled-text convention (naive rotation sort returns 0, port returns
  1/3/1/1, decode round-trips under the port's primary); empty input works
  end to end. The `decode(b"", 1) == b""` pin is a port convention (bzip2
  stores no empty block), correctly documented as such in the test.

## 5. What could not be reproduced / out of scope

- The original README numbers (see §2) — not reproducible in any of 5 runs.
- `bench/analysis.md`'s supplementary per-function numbers were spot-checked
  (suffix 1.64 s, encode 1.80 s, decode 23 ms, last_column 8.1 ms, RLE 2.3 ms,
  histogram 0.71 ms, bz2 151 ms) and agree with this session's probes within
  noise. Its full 64 B–1 MiB scaling table was not re-run in full; the
  64 KiB/1 MiB anchors were.
- No code bug was found requiring a fix (the one real bug, the empty-primary
  leak, was already fixed in the `phase test` commit). No kernel, wrapper,
  or bench code was changed.

## 6. Changes made

1. `README.md`: performance table — all 5 rows corrected to measured values
   (see §2); "ten seconds" sentence corrected to "over a second and a half".
2. `tests/test_bwt.py`: removed vacuous `or True` from one assertion.
3. `bench/verify.md`: this file.

Final gate after edits: `pixi run build && pixi run test` — both green,
239 passed in 4.25s, exit 0.
