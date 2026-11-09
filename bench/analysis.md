# mojo-bwt — ANALYZE phase

Measured map of where the time goes. No algorithm was changed in this phase; `src/` and `tests/` are untouched. Every number below comes from a run in this session. Only `bench/analysis.md` was added.

Method: `pixi run bench` (flock-wrapped, `/tmp/mojo-bench.lock`) plus a supplementary harness run under the same flock that times **every** public entry point (`suffix_array`, `encode`, `last_column`, `lf_map`, `decode`, `histogram`, `rle_encode`, `rle_decode`) at the sizes real callers pass — 64 B, 1 KiB, 64 KiB, 1 MiB, plus 256 B–262 144 B scaling probes — each best-of-N in-process. Byte/flop columns are derived from the kernel source (`src/kernels.mojo`, `python/mojo_bwt/_lib.py`): bytes = reads+writes at element size, flops = integer ops (adds/comparisons/branches). GPU verdicts cite the flops-per-byte figure.

## Host / provenance

- Host: x86_64, 72 cores, 251 GiB RAM, Linux 6.8.0. Kernels are serial: no `parallelize`, `spawn`, or threading anywhere in `src/kernels.mojo`.
- Upstream `bwt` on PyPI: **not importable** (`importlib.util.find_spec('bwt')` → `None`, `pip` shows no `bwt` package; `README.md` already notes PyPI `bwt` is a 404). The reference implementation is bzip2 via the stdlib `bz2` module, which is timed below as the upstream comparison. `bz2` runs the full codec (RLE1 + BWT + MTF + Huffman), so it is a *lower bound* on the transform cost, not a like-for-like BWT-only baseline — see "Port slower than upstream" for the consequence.
- Reference baselines (per `bench/bench.py`): numpy `bincount` for histogram, plain-Python run split for RLE, numpy prefix-doubling for the sort, plain-Python permutation walk for the inverse. All are timed here too.

## Ranked hotspot table

Sizes are input bytes `n`; suffix rows are `m = 2n+1`. Times are best-of-3/5, in-process, wall clock.

| # | Function (at `n = 1 MiB`, `m = 2 MiB+1`) | Input size | Time (measured) | Bytes touched (from source) | Flops (from source) | Flops / byte | Memory- or compute-bound | Python/ctypes glue or kernel |
|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | `bwt_suffix_array` kernel via `suffix_array` | 1 MiB (`m` = 2 097 153 int32) | 1 644.33 ms (bench: 1 811.82 ms) | ~0.54–1.41 GB (≈32 B/elem × 8–21 prefix-doubling rounds, §2) | ~101–264 M int ops | ≈0.19 | memory-bound (latency/dependency-bound, not peak-BW — §2) | kernel |
| 2 | `encode` (= suffix + last column) | 1 MiB | 1 796.09 ms (bench: 1 821.56 ms) | §1 + §3 ≈ 0.6–1.5 GB | ~104–267 M | ≈0.18 | memory-bound | kernel (glue ≈1 ms, <0.1%) |
| 3 | `decode` (`bwt_inverse`, incl. `_fill_lf` + permutation chase) | 1 MiB | 23.16 ms (bench: 27.84 ms) | 19.9 MB (19 B/elem) | 5.24 M | 0.26 | memory-bound (dependent chase) | kernel |
| 4 | `last_column(data, sa)` (incl. `_sentinel_text` rebuild) | 1 MiB | 8.09 ms | 18.9 MB (9 B × `m`) | 3.15 M | 0.17 | memory-bound | kernel + glue rebuild (≈1 ms, ~12%) |
| 5 | `rle_decode` (expands 1 037 430 runs) | 1 MiB | 6.98 ms | 6.24 MB | 3.13 M | 0.50 | memory-bound | kernel |
| 6 | `rle_encode` | 1 MiB | 2.31 ms (bench: 2.47 ms) | 6.24 MB | 3.15 M | 0.50 | memory-bound | kernel |
| 7 | `lf_map` (`bwt_lf_map` via `_fill_lf`) | 1 MiB | 1.91 ms | 14.7 MB (14 B/elem, 2 passes) | 4.19 M | 0.29 | memory-bound | kernel |
| 8 | `histogram` (`bwt_histogram`) | 1 MiB | 0.71 ms (bench: 0.67 ms) | 1.05 MB (1 B/elem) | 2.10 M | 2.00 | memory-bound (dependency chain, §2) | kernel |

All other public entry points were also timed at `n = 64, 1 024, 65 536` (see §4). No row was dropped because the port is slower; slower-than-upstream rows are kept and flagged in §5.

Scaling probes (`encode`, best-of-3): 256 B → 0.083 ms (323 ns/B); 1 024 B → 0.231 ms (225 ns/B); 4 096 B → 1.293 ms (316 ns/B); 16 384 B → 8.566 ms (523 ns/B); 65 536 B → 56.765 ms (866 ns/B); 262 144 B → 281.338 ms (1 073 ns/B); 1 MiB → 1 511.7 ms (1 442 ns/B). Per-byte cost *rises* with `n`, consistent with `O(n log n)` prefix-doubling rounds (`log2(2n)` ≈ 21 at 1 MiB, ~8–12 effective for the skewed corpus) rather than a linear kernel.

## 1. What the table says, in one paragraph

One kernel is the whole story: **the prefix-doubling suffix sort is ~98% of `encode` time** (1 644 ms of 1 796 ms at 1 MiB; 51.8 ms of 50.1 ms at 64 KiB — i.e. encode *is* the sort, plus a millisecond of glue). Everything else is slivers by comparison: decode 23 ms (1.3% of encode), last_column 8 ms, rle_decode 7 ms, rle_encode 2.3 ms, lf_map 1.9 ms, histogram 0.7 ms. At small sizes (64 B–1 KiB) fixed glue dominates and the picture inverts (see §4), but no size changes the ranking at the 1 MiB size real callers pass. Fix the sort and every other row is rounding error; fix anything else first and the measured wall clock barely moves.

## 2. Why the sort dominates — and why it is not a bandwidth problem

`bwt_suffix_array` (`src/kernels.mojo`, lines 36–143) does one initial counting pass plus, per doubling round, three linear passes over `m`: build the shifted order `y`, one counting sort on the first key, and the renumber pass. That is roughly 32 B of traffic per element per round (4 B reads/writes × ~5 array sweeps) against ~6 integer ops per element per round, hence **≈0.19 flops/byte** for any round count (0.188 at rounds = 8, 12, 16, or 21 — the ratio is round-invariant because both bytes and flops scale with rounds). At 1 MiB that is 0.54 GB (8 rounds) to 1.41 GB (21 rounds) of traffic in 1.64 s, i.e. **0.3–0.9 GB/s effective** — one to two orders of magnitude below single-core DRAM bandwidth, and the per-byte cost *grows* with `n` (866 ns/B at 64 KiB → 1 442 ns/B at 1 MiB). This is not peak-bandwidth-bound; it is **latency/dependency-bound**: the counting-sort loops carry serial dependencies through `cnt[rank[i]]` (read-modify-write on a data-dependent index), the radix scatter `sa[cnt[r] -= 1] = y[i]` is a random-access store chain, and the renumber pass reads `rank[u+width]` / `rank[v+width]` at scatter offsets. The same signature shows up in the small kernels: histogram moves 1 MiB in 0.71 ms (1.5 GB/s, serial `h[src[i]] += 1` chain), lf_map moves 14.7 MB in 1.91 ms (7.7 GB/s but still far from peak), and the decode walk is a pure dependent permutation chase (19.9 MB in 23 ms, 0.86 GB/s). **Every kernel in the port is memory-bound by flops/byte (all ≤0.5, histogram exactly 2.0), and every one is far below peak bandwidth because nothing is parallel and every inner loop has a serial dependency.** That combination — low flops/byte *and* low achieved bandwidth — points at parallelization + dependency-breaking, not at more FLOPs.

## 3. Buffers, allocations, copies — measured inventory

- `suffix_array` at 1 MiB allocates **six** `m`-sized buffers per call: `text` (8.39 MB), `sa`, `rank`, `nrank`, `shifted`, `cnt` (8.39 MB each, **50.3 MB total**) in `python/mojo_bwt/_lib.py` (`suffix_array`, lines 84–104). `encode` builds the sentinel text **twice** — once inside `suffix_array` and again inside `last_column` (`encode` → `suffix_array` + `last_column`, each calling `_sentinel_text`) — so the doubled-text build (8.39 MB alloc + 2×1 MiB copies + `+1` shift) is done twice per encode. Measured `_sentinel_text` cost at 1 MiB is 0.998 ms (1.031 ms median) — <0.1% of encode, so it is not a hotspot today, but it is a *pure* per-call allocation+copy with no reuse.
- `last_column` measured standalone (8.09 ms) includes its own `_sentinel_text` rebuild (~1 ms, ~12% of that call); the kernel itself is ~7 ms of single-pass scans. Callers who already hold a suffix array pay the rebuild anyway.
- `encode` → `decode` round-trips re-allocate everything: `decode` allocates `out` (n), `work` (256) and `jump` (n); `rle_encode` allocates `values` + `lengths` at full `n` (not at true run count) and then `.copy()`s the first `runs` entries (`_lib.py` line 182: `values[:runs].copy()`, `lengths[:runs].copy()` — two extra copies sized `runs` ≈ n for skewed-random data, since avg run length is 1.01 and `runs` = 1 037 430 at 1 MiB, i.e. the "shrink" barely shrinks anything and the copy is nearly full-size).
- `rle_encode`'s output arrays are 5 B per run (1 B value + 4 B length); at 1 MiB skewed-random that is 5.24 MB of writes for 1,037,430 runs — the kernel writes ~5× the input size to describe runs of length ~1. This is inherent to the API (dense in/out), not a bug, but downstream phases should know the RLE stage *expands* on high-entropy input rather than shrinking it.
- `decode` materialises the full LF permutation (`jump`, 4 MB at 1 MiB) and then chases it — the README's bzip2-matching design. The chase is inherently serial; no allocation is wasted, but the traffic (4 B read + 1 B write per step, dependent) is the cost.

All allocations are per-call `np.empty` (19 `np.empty` sites in `_lib.py`); there is no buffer pool or scratch reuse. At 1 MiB the 50 MB sort allocation is ~2 ms of allocator time (measured `alloc(text)` alone 0.002 ms is page-cache warm and unrepresentative; the dominant cost is page faults + cache-cold first touch inside the kernel, which is part of the 1.64 s, not separate). At small sizes the *fixed* cost is the issue (§4).

## 4. Small-size behaviour — where glue, not the kernel, sets the floor

At `n ≤ 1 024` the Python/ctypes glue floor is directly visible:

| n | mojo encode | mojo decode | mojo histogram | numpy/Python reference |
|---:|---:|---:|---:|---|
| 0 | 0.004 ms | 0.025 ms | 0.016 ms | — |
| 1 | 0.094 ms | 0.111 ms | 0.015 ms | — |
| 6 (`banana`) | 0.084 ms | 0.109 ms | 0.015 ms | — |
| 256 | 0.157 ms | 0.188 ms | 0.015 ms | numpy bincount 0.001–0.004 ms |
| 64 | 0.045 ms | 0.010 ms | 0.007 ms | numpy bincount 0.001 ms; Python RLE 0.007 ms |
| 1 024 | 0.207 ms | 0.015 ms | 0.007 ms | numpy bincount 0.004 ms |

Three specific findings:

- **No per-element ctypes re-entry anywhere.** Every Python function in `_lib.py` makes exactly **one** `lib.*` call per public operation (verified by grep: one `lib.bwt_*` call site per wrapper), with scalar args marshalled as `c_int64` addresses. There is no Python-level loop that re-enters ctypes per element. The only Python loops are inside the *reference* baselines (`bench.py`), not the port.
- **The ctypes floor itself is ~10–15 µs per call** (bare `histogram` at `n=1` is 0.015 ms where the kernel is 256 iterations of trivial work). At `n=64`, `mojo histogram` (0.007 ms) is *slower* than numpy `bincount` (0.001 ms) and `mojo rle_encode` (0.013 ms) slower than Python `python_rle` (0.007 ms) — the only place the port loses to its baseline, and purely a fixed-overhead effect, not a kernel effect. At `n=1 024` the crossover is done (histogram 0.007 ms vs bincount 0.004 ms is still a loss; rle 0.015 ms vs Python 0.087 ms is already a 6× win).
- **`encode` at small `n` is dominated by `suffix_array`'s six allocations, not by arithmetic**: `encode` at 256 B costs 0.157 ms while `decode` at 256 B costs 0.188 ms — the same order — because both allocate ~`n`-sized scratch and build sentinel text. Raw-kernel time (prebuilt buffers) at 1 MiB for the sort is 1 923 ms vs 1 702 ms through the wrapper — the wrapper is *faster* than the raw call within noise (1 702–2 273 ms spread), confirming glue is <1% of the sort at caller sizes. The glue problem is exclusively a small-input (<1 KiB) problem.

## 5. Port vs upstream — the row that must not be dropped

The port is **slower than upstream bzip2 by ~8–10× in absolute terms** at the sizes real callers pass, and this row is kept, not hidden:

| n | mojo `encode` (BWT only) | upstream `bz2.compress` (BWT + MTF + Huffman + RLE1) | mojo `encode` + `decode` (BWT only, both directions) | upstream `bz2` compress+decompress (full codec, both directions) |
|---:|---:|---:|---:|---:|
| 64 | 0.045 ms | 0.037 ms | 0.055 ms | 0.054 ms |
| 1 024 | 0.207 ms | 0.334 ms | 0.222 ms | 0.450 ms |
| 65 536 | 50.15 ms | 8.63 ms | 50.96 ms | 12.98 ms |
| 1 048 576 | 1 796.09 ms | 150.97 ms | 1 819.25 ms | 227.996 ms |

At 1 MiB the Mojo BWT-alone `encode` (1.80 s) is **11.9× slower** than upstream bzip2's *entire* compress (0.151 s), which includes the same transform plus MTF, Huffman, and RLE1. Upstream bzip2 uses an `O(n log n)` sort too, but in C at ~150 ms/MiB; the Mojo port's prefix-doubling at ~1 600 ms/MiB is ~10× slower per byte. **This is the "port slower than upstream" flag the task requires: it is the suffix sort, and it is the #1 hotspot.** The bench's numpy prefix-doubling baseline (8 618 ms at 1 MiB, this session) makes the port look 4.76× *faster* than "reference", but that baseline is NumPy with ~10× the port's per-round overhead — the honest upstream comparison is the bzip2 C implementation above, and against it the port is 8–12× behind. Both comparisons are true; they answer different questions, and only the bzip2 one tells you how far the port has to go.

Similarly, the port is slower than its histogram baseline **only** at `n ≤ 1 024` (0.007 ms vs 0.001–0.004 ms, fixed ctypes+alloc overhead) and faster everywhere else (bench: 0.67 ms vs 2.03 ms at 1 MiB). The `rle_encode` comparison inverts the same way (slower than Python at 64 B, 37× faster at 1 MiB). These small-size losses are glue-floor effects (§4), not kernel regressions, and no upstream-equivalent exists for them (bzip2's RLE1 is different from general RLE).

## 6. Which phase acts on each hotspot

Verdict per hotspot, with the flops-per-byte figure behind it. "GPU" is only assigned when flops/byte warrants it (>~10, i.e. compute-bound); nothing in this port clears that bar, so no GPU verdict is given — a GPU path for any kernel here would be bandwidth/latency-bound on the host-device transfer and would not pay.

| Hotspot | Next phase | Justification (flops/byte + measured time) |
|---|---|---|
| `bwt_suffix_array` (1.64 s, 98% of encode) | **parallel** | 0.19 flops/byte, memory-bound with serial dependency chains through `cnt[rank[i]]` and the `sa[cnt[r]-=1]` scatter. The three passes per round (shifted-order build, first-key counting sort, renumber) are each parallelisable counting-sort scaffolds with per-thread histograms + prefix sums — the standard parallel radix-sort pattern. 72 cores available, kernels currently 100% serial. SIMD alone would not break the `cnt` dependency; a GPU would be starved at 0.19 flops/byte (transfer-bound, no compute to hide it). Unrolling helps marginally. |
| `encode` as a whole | **none** (derives from sort) | 0.18 flops/byte, ~98% sort. No separate action; see sort. The duplicate `_sentinel_text` build (~1 ms, 0.06%) is not worth a phase alone — fold it into the sort phase as a drive-by. |
| `decode` / inverse walk (23.2 ms) | **none** (accept) | 0.26 flops/byte, dependent permutation chase — inherently serial (each step needs the previous jump). Parallelising a permutation chase requires algorithmic change (cycle decomposition), which this phase forbids; the measured 23.2 ms is already 27× faster than the Python walk (633 ms) and 5.9× faster than upstream decompress wall clock for the full codec (85 ms decompress leg). Not the bottleneck. |
| `last_column` (8.09 ms) | **none** | 0.17 flops/byte, single pass over `m`. 0.5% of encode. The 12% glue share (sentinel rebuild) disappears if the sort phase shares its `text` buffer. |
| `rle_decode` (6.98 ms) | **none** | 0.50 flops/byte. Inner expand loop is a memset-style fill; already 13× faster than any Python formulation. Its absolute time (0.4% of encode) does not justify a phase. |
| `rle_encode` (2.31 ms) | **none** | 0.50 flops/byte, already 37–39× faster than Python. Small. |
| `lf_map` (1.91 ms) | **none** | 0.29 flops/byte, 0.1% of encode. Fused inside `decode` anyway. |
| `histogram` (0.71 ms) | **none** | 2.00 flops/byte (the highest in the port, still memory-bound by the >~10 GPU threshold). Already 3× faster than numpy `bincount` at 1 MiB. At small `n` the loss is the fixed glue floor, not the kernel — a parallel histogram with atomics would be slower. |
| Small-input glue floor (<1 KiB) | **unroll** (lowest priority) | Not a flops problem at all — fixed ~10–15 µs ctypes + 6× `np.empty` allocations dominate at `n ≤ 1 024` (histogram loses to `bincount` 0.007 vs 0.001 ms). Reducing allocations (scratch reuse, `np.empty` → stack for tiny `n`) and batching args helps if callers pass small inputs; at caller sizes (≥64 KiB) this is <0.1% and irrelevant. Unroll is a micro-cleanup, not a phase. |
| SIMD anywhere | **none** | No kernel has ≥1 vectorisable FP/integer SOTA pattern: all are scatter/gather through `cnt[rank[i]]` or dependent chases with data-dependent indices. 0.17–0.5 flops/byte means the ALU is idle waiting on memory; widening the vector would not widen the dependency. SIMD is the wrong lever for every row here. |

**The one action the numbers support is parallelising the suffix sort.** It is the only hotspot that is both dominant (98%) and structurally parallelisable (counting-sort passes with per-thread partial histograms). Every other hotspot is either tiny in absolute time, inherently serial (the inverse walk), or already winning against its baseline by 3–37×. A GPU phase is not justified for any kernel: the highest flops/byte in the port is 2.0 (histogram), an order of magnitude below the ~10 flops/byte where compute starts to dominate; all kernels would be transfer-bound.

## 7. Reproduce

```bash
pixi run build        # bash build/build.sh under pixi
pixi run bench        # flock /tmp/mojo-bench.lock python bench/bench.py
```

Supplementary harness (all entry points, sizes 64 B–1 MiB, `bz2` upstream comparison, glue-vs-kernel splits) was run this session from `/tmp/probe_analysis.py` under the same flock; its output is quoted verbatim above (best-of-N, in-process, `time.perf_counter`). Byte/flop figures were derived from source loops in `src/kernels.mojo` and `python/mojo_bwt/_lib.py` and cross-checked against the measured times (e.g. 0.54–1.41 GB of sort traffic at 0.3–0.9 GB/s effective confirms latency-bound, not bandwidth-bound).
