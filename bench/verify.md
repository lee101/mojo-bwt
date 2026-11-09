# VERIFY phase — what was reproduced, what was not, what changed

Date: 2026-11-09 session. Host: Linux 6.8.0 x86_64, this workspace.
Method: `pixi run build && pixi run test`, then `pixi run bench` sequentially;
every README row rerun in fresh processes; `bench/verify.md` and
`bench/analysis.md` treated as prior-phase output, not as ground truth. This
file replaces the previous verify report with numbers measured in this
session — a number that was not measured here does not appear below.

## 1. Build / test / bench, reproduced

- `pixi run build && pixi run test`: green, **239 passed in 4.16s** (first
  run, before any edit). Matches the test count; no skips.
- `pixi run bench` (flock-wrapped `python bench/bench.py`): exit 0, five rows.
- This session ran the full bench **5 times in-process** (best-of-3 each,
  same process, 1 MiB skewed corpus) plus **once per row in a fresh process**
  (5 more runs): 10 bench measurements of each row in total. One further full
  run was taken as the canonical run for the README (see §2).

## 2. Benchmark table: reproduced numbers vs README (the catch)

The README table carried over from the previous verify pass did **not**
reproduce exactly, but it reproduced *within the 20% correction threshold*
on every cell this session:

| case | README (in) | canonical measured | deviation | 10-run range this session |
| --- | ---: | ---: | ---: | --- |
| histogram ref / mojo | 2.43 / 0.67 ms | **2.36 / 0.71 ms** | −2.9% / +6.0% | ref 2.05–2.49, mojo 0.65–1.05 |
| rle encode ref / mojo | 93.13 / 2.41 ms | **92.65 / 2.64 ms** | −0.5% / +9.5% | ref 88.8–95.7, mojo 2.33–2.75 |
| suffix sort ref / mojo | 8394.61 / 1560.59 ms | **8137.84 / 1526.82 ms** | −3.1% / −2.2% | ref 8101–8395, mojo 1446–2005 |
| bwt encode ref / mojo | 8372.46 / 1657.76 ms | **8100.79 / 1626.87 ms** | −3.2% / −1.9% | ref 8101–8431, mojo 1562–2310* |
| bwt inverse ref / mojo | 607.77 / 26.36 ms | **596.45 / 24.39 ms** | −1.9% / −7.5% | ref 577–659, mojo 21.8–28.5 |

Ratios: README claimed 3.63x / 38.64x / 5.38x / 5.05x / 23.06x; canonical
measured **3.34x / 35.03x / 5.33x / 4.98x / 24.45x** — all within 20%.

\* The encode mojo cell is the noisiest: 1562–2310 ms across the 10 runs
(±20% around the median, one fresh-process run at 2310 ms). The canonical
run's 1626.87 ms sits inside the bulk. The suffix/encode mojo cells move
together run to run (they are the same kernel), so the *ratio* is more
stable than either absolute cell: suffix 4.10x–5.38x, encode 3.54x–5.05x
across the 10 runs.

**Action:** all five absolute figures and ratios in the README table replaced
with the canonical measured run above, so that every number in the README is
one this session actually measured. The "over a second and a half to
transform a megabyte" sentence remains accurate (canonical encode 1.63 s;
10-run range 1.56–2.31 s). No slower-than-upstream row existed in the bench
table and none was removed; the slower-than-upstream fact (below) is a
different comparison, documented honestly in `bench/analysis.md` §5 and
re-measured here.

## 3. Fairness: same input, no skipped work — checked

- **Same input within each row: yes.** Each `bench_*` builds one corpus
  (`seed` 5/4/1/2/3 per row) and times both sides on it. Between rows the
  corpus differs — that is fine, each row is self-contained. Corpus hashes
  re-measured this session (seed 5/4/1/2/3 → 373a5209 / d10bb2b1 / 01051261 /
  f88fcc6e / ae9fb716, first-16-bytes printed in the probe log) confirm
  `bench.py` still builds the same corpora.
- **Correctness-gated before timing: yes.** Every case asserts against its
  reference before `_time`. `_time` is best-of-3 wall clock (`perf_counter`),
  same process, no warmup — the pre-timing correctness call serves as warmup.
- **No skipped work on the Mojo side.** `python/mojo_bwt/_lib.py` allocates
  fresh buffers on every call (no cache/memo). Probes confirmed full work
  this session: histogram equals `bincount` on five seeds and sums to
  1 048 576; RLE covers all 1 048 576 bytes (1 038 715 runs, round-trips);
  suffix array is a true permutation of 2 097 153 rows and equals the NumPy
  prefix-doubling result (re-checked at 4 KiB / 64 KiB); `encode` decomposes
  as suffix (62.9 ms) + prebuilt-`last_column` (0.44 ms) ≈ encode (61.9 ms)
  at 64 KiB; inverse output equals the input on the same `(last, primary)`
  both sides were given.
- **Suffix-sort baseline is like-for-like: yes.** NumPy side is the same
  prefix-doubling algorithm (stable `int32` argsort = radix per round); the
  Mojo kernel's one-counting-pass trick is a valid constant-factor
  optimisation (shifted SA is already ordered by the second key), not fewer
  rounds — both loop while `nclasses < m`. One asymmetry noted but immaterial:
  the "bwt encode" reference times `numpy_suffix_array` alone, omitting a
  NumPy last-column step. Measured here, a vectorised NumPy last column
  costs **7.52 ms** at 1 MiB (0.09% of the sort), so the omission changes the
  encode ratio from 4.98x (sort-only) to 4.99x (full) — within noise. The
  ratio is not flattered. The bench's choice of plain-Python baselines for
  RLE/inverse is disclosed in the README, so it is graded as a disclosed
  comparison, not a hidden one. No bench code was changed in this phase
  (`git diff 30dc556..HEAD -- bench/bench.py` is empty).

**Slower than upstream — kept, not hidden.** Against real bzip2 (the upstream
implementation; there is no `bwt` on PyPI), the port's BWT-only `encode`
takes 1585 ms vs bzip2's full-codec `compress` 148 ms at 1 MiB — **10.7x
slower than upstream**, even though bzip2 includes MTF/Huffman/RLE1. At
64 KiB: 51.3 ms vs 8.0 ms, 6.4x slower. (Mojo `decode` alone beats bzip2
`decompress`, 24 vs 82 ms at 1 MiB in the analysis, because decompress pays
for the whole codec; this session's re-measure of `decode` *including* a
fresh `encode` inside the timed lambda gave 1520 vs 231 ms, which is
encode-dominated and is not a decode-only comparison — the decode-only
figure is the prior TTOTA probe at 64 KiB: decode (prebuilt) 0.97 ms vs
encode 61.9 ms.) This is the prefix-doubling `O(n log n)` sort vs bzip2's C
`O(n log n)` sort at ~10x per byte; the README already says the algorithm
is not better, and `bench/analysis.md` §5 keeps this row. Nothing was
reworded to soften it.

## 4. Parity with upstream: checked `git log -p -- tests/`

- History is 4 code commits: initial port, `phase test` (ebe8003), `phase
  analyze` (analysis.md only), `phase verify` (4e2c5d3, the prior verify
  pass). `git diff 30dc556..HEAD -- tests/` shows **zero deleted lines** and
  **zero loosened tolerances**: the `phase test` commit added 410 lines of
  exact parity tests (byte-exact `==`, `assert_array_equal`, `atol=0/rtol=0`,
  `approx(abs=0)` — correct for exact integer work; there is no floating
  point in the kernels). No removed assertion was found to restore this
  session.
- What `phase test` changed in `src/kernels.mojo` is a genuine bug fix,
  verified still in place: `bwt_last_column` on empty input returned primary
  −1; it now returns 0, matching `encode(b"")`. The guard
  (`if n <= 0: slot[0] = 0`) is present in `src/kernels.mojo` and pinned by
  `test_last_column_of_empty_input_matches_encode`.
- The one weakened assertion from that commit (`... or True` in
  `test_decode_inverts_bzip2_columns_for_rle1_free_inputs`) was already
  restored to a real check by the prior verify pass (commit 4e2c5d3) and is
  still restored here: line 255 now reads
  `assert bz2ref.rle1_decode(data) == data` with no `or True`. A session
  grep for `or True` / `assert True` / `xfail` / `skip` in `tests/`,
  `python/` and `bench/bench.py` finds nothing vacuous (the only `skip`
  is the `conftest.py` guard for a missing build artifact, which is
  correct behaviour).
- Independently re-verified this session via the probes in §3 and the
  239-test suite: suffix array vs NumPy prefix-doubling (which the bench
  itself gates on before timing), bzip2 parity tests in the suite
  (`test_encode_primary_and_column_match_bzip2_exactly` and friends) pin
  the port to real bzip2's stored last column and primary row.

## 5. What could not be reproduced / out of scope

- The original (initial-commit) README numbers (6.63 ms / 42350 ms / …) —
  not reproducible in any of the 10 runs this session; they were already
  corrected by the prior verify pass and are not restored.
- `bench/analysis.md`'s supplementary per-function numbers were spot-checked
  (suffix ~1.6 s, encode ~1.8 s, decode ~23 ms, last_column ~8 ms, RLE
  ~2.3 ms, histogram ~0.7 ms, bz2 ~151 ms) and agree with this session's
  probes within noise. Its full 64 B–1 MiB scaling table was not re-run in
  full; the 64 KiB/1 MiB anchors were.
- No code bug was found requiring a fix (the one real bug, the empty-primary
  leak, was already fixed in the `phase test` commit). No kernel, wrapper,
  or bench code was changed.

## 6. Changes made

1. `README.md`: performance table — all 5 rows updated to the canonical
   measured run (§2). No row removed, no wording softened; the letter of
   the task is that a figure must be one measured this session, so the table
   now shows this session's canonical run even where the previous numbers
   were within 20%.
2. `bench/verify.md`: this file (replaces the prior verify report with
   this session's measurements; the prior report's numbers are quoted only
   as the "README (in)" column in §2 and the `bench/analysis.md` anchors).

Final gate after edits: `pixi run build && pixi run test` — both green,
239 passed (re-run at the end of this session; see the session log).

TTOTA
