"""Burrows-Wheeler transform kernels: suffix sorting, last column, LF mapping,
run-length coding.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

Conventions
-----------
* Text is handed to `bwt_suffix_array` as an `Int32` array of length `n` whose
  last element is the unique smallest symbol `0` and whose other elements lie
  in `[1, 256]` (byte `b` stored as `b + 1`).  A unique smallest sentinel at
  the end is what lets the prefix-doubling ranks separate every suffix.
* The transform itself is the *cyclic* Burrows-Wheeler transform over the `n`
  rotations of the input, which is what bzip2 computes: the last column holds
  only real bytes and `primary` names the row that starts the original string.
"""

comptime I32 = Pointer[Int32, AnyOrigin[mut=True]]
comptime U8 = Pointer[UInt8, AnyOrigin[mut=True]]


def ip(addr: Int) -> I32:
    return I32(unsafe_from_address=addr)


def up(addr: Int) -> U8:
    return U8(unsafe_from_address=addr)



# ---------------------------------------------------------------------------
# Suffix sorting: prefix doubling with radix sort (Manber & Myers)
# ---------------------------------------------------------------------------


@export("bwt_suffix_array")
def bwt_suffix_array(
    text_addr: Int, n: Int, k: Int, sa_addr: Int, rank_addr: Int,
    nrank_addr: Int, y_addr: Int, cnt_addr: Int, cnt_words: Int,
) abi("C") -> Int32:
    """Sort the suffixes of the sentinel-terminated `Int32` text.

    Prefix doubling: rank every suffix by its first symbol, then repeatedly
    refine the ranks on pairs `(rank[i], rank[i + width])` by a radix sort.  Each
    round is three linear passes, so the sort is `O(n log n)` with `O(n)` space
    and no comparisons, which is what makes it worth compiling.  `width` starts
    at 1 and doubles, so after round `r` the ranks distinguish suffixes that
    agree on their first `2**r` symbols.

    The second sort key is produced without a second counting pass: positions
    whose successor runs past the end have no second key and go first, then the
    shifted suffix array is already ordered by the second key, and one stable
    counting sort on the first key finishes the pair.

    Returns 0 on success, -1 for an empty text, -2 if the counting buffer is
    too small.
    """
    if n <= 0:
        return Int32(-1)
    if n == 1:
        ip(sa_addr)[unsafe_offset=0] = 0
        return Int32(0)
    if k + 1 > cnt_words:
        return Int32(-2)

    var s = ip(text_addr)
    var sa = ip(sa_addr)
    var rank = ip(rank_addr)
    var nrank = ip(nrank_addr)
    var y = ip(y_addr)
    var cnt = ip(cnt_addr)

    # First pass: counting sort on the first symbol.
    for i in range(k + 1):
        cnt[unsafe_offset=i] = 0
    for i in range(n):
        cnt[unsafe_offset=Int(s[unsafe_offset=i])] += 1
    var acc: Int32 = 0
    for i in range(k + 1):
        acc += cnt[unsafe_offset=i]
        cnt[unsafe_offset=i] = acc
    for i in range(n - 1, -1, -1):
        var c = s[unsafe_offset=i]
        cnt[unsafe_offset=c] -= 1
        sa[unsafe_offset=cnt[unsafe_offset=c]] = Int32(i)

    var classes: Int = 1
    rank[unsafe_offset=sa[unsafe_offset=0]] = 0
    for i in range(1, n):
        if s[unsafe_offset=Int(sa[unsafe_offset=i])] != s[unsafe_offset=Int(sa[unsafe_offset=i - 1])]:
            classes += 1
        rank[unsafe_offset=sa[unsafe_offset=i]] = Int32(classes - 1)

    var width: Int = 1
    while classes < n and width < n:
        if classes > cnt_words:
            return Int32(-2)
        # Order by the second key: no successor first, then the shifted
        # suffix array, which the previous round left ordered by `rank`.
        var p: Int = 0
        for i in range(n - width, n):
            y[unsafe_offset=p] = Int32(i)
            p += 1
        for i in range(n):
            var x = Int(sa[unsafe_offset=i])
            if x >= width:
                y[unsafe_offset=p] = Int32(x - width)
                p += 1
        # Stable counting sort on the first key.
        for i in range(classes):
            cnt[unsafe_offset=i] = 0
        for i in range(n):
            cnt[unsafe_offset=Int(rank[unsafe_offset=i])] += 1
        var acc2: Int32 = 0
        for i in range(classes):
            acc2 += cnt[unsafe_offset=i]
            cnt[unsafe_offset=i] = acc2
        for i in range(n - 1, -1, -1):
            var r = rank[unsafe_offset=y[unsafe_offset=i]]
            cnt[unsafe_offset=Int(r)] -= 1
            sa[unsafe_offset=cnt[unsafe_offset=Int(r)]] = y[unsafe_offset=i]
        # Renumber: two suffixes share a rank only if both halves match.  A
        # successor past the end compares as -1, below every real rank.
        classes = 1
        nrank[unsafe_offset=sa[unsafe_offset=0]] = 0
        for i in range(1, n):
            var u = Int(sa[unsafe_offset=i - 1])
            var v = Int(sa[unsafe_offset=i])
            var ru: Int32 = -1
            var rv: Int32 = -1
            if u + width < n:
                ru = rank[unsafe_offset=u + width]
            if v + width < n:
                rv = rank[unsafe_offset=v + width]
            if rank[unsafe_offset=u] != rank[unsafe_offset=v] or ru != rv:
                classes += 1
            nrank[unsafe_offset=v] = Int32(classes - 1)
        var spare = rank
        rank = nrank
        nrank = spare
        width *= 2
    return Int32(0)

# ---------------------------------------------------------------------------
# Burrows-Wheeler transform
# ---------------------------------------------------------------------------


@export("bwt_last_column")
def bwt_last_column(
    s_addr: Int, sa_addr: Int, n_total: Int, n: Int, lf_addr: Int,
    primary_addr: Int,
) abi("C") -> Int32:
    """Last column of the cyclic transform, read off a suffix array of the
    doubled text.

    The rows of the rotation matrix are exactly the suffix positions below `n`
    of the suffix array of `T + T`; dropping the sentinel row and the wrapped
    rows leaves the `n` rotations in order.  Returns 0, or -4 if the suffix
    array did not hold exactly `n` rows in `0 .. n - 1`."""
    var s = ip(s_addr)
    var sa = ip(sa_addr)
    var lf = up(lf_addr)
    var slot = ip(primary_addr)
    var m: Int = 0
    var primary: Int = -1
    for k in range(1, n_total):
        var r = Int(sa[unsafe_offset=k])
        if r >= 0 and r < n:
            var idx = r - 1
            if idx < 0:
                # Row `r == 0` has no predecessor inside T: the cyclic
                # predecessor is the last byte.
                idx = n - 1
            lf[unsafe_offset=m] = UInt8(s[unsafe_offset=idx] - 1)
            if r == 0:
                primary = m
            m += 1
    if m != n:
        return Int32(-4)
    slot[unsafe_offset=0] = Int32(primary)
    return Int32(0)


@export("bwt_histogram")
def bwt_histogram(src_addr: Int, n: Int, hist_addr: Int) abi("C") -> Int32:
    """256-bin byte histogram.  Its running sum is the first-column table the
    LF mapping needs."""
    var src = up(src_addr)
    var h = ip(hist_addr)
    for i in range(256):
        h[unsafe_offset=i] = 0
    for i in range(n):
        h[unsafe_offset=Int(src[unsafe_offset=i])] += 1
    return Int32(0)


@always_inline
def _fill_lf(lf_addr: Int, n: Int, out_addr: Int, w_addr: Int):
    """Materialise the LF-mapping into `out`, using `w` as the 256-entry
    first-column table."""
    var lf = up(lf_addr)
    var out = ip(out_addr)
    var w = ip(w_addr)
    for i in range(256):
        w[unsafe_offset=i] = 0
    for i in range(n):
        w[unsafe_offset=Int(lf[unsafe_offset=i])] += 1
    var acc: Int32 = 0
    for c in range(256):
        var cnt = w[unsafe_offset=c]
        w[unsafe_offset=c] = acc
        acc += cnt
    for j in range(n):
        var c = Int(lf[unsafe_offset=j])
        out[unsafe_offset=j] = w[unsafe_offset=c]
        w[unsafe_offset=c] += 1


@export("bwt_lf_map")
def bwt_lf_map(
    lf_addr: Int, n: Int, out_addr: Int, work_addr: Int
) abi("C") -> Int32:
    """LF-mapping of the last column.

    `LF[j]` is the row of the rotation that starts one byte earlier than the
    rotation on row `j`, that is `C[c] + rank of j among the occurrences of c`
    where `C` is the first-column start of bucket `c`.  The rank is the count
    of the same byte *before* row `j`, which is why the mapping has to be
    materialised rather than folded into the inverse walk.  Returns 0."""
    _fill_lf(lf_addr, n, out_addr, work_addr)
    return Int32(0)


@export("bwt_inverse")
def bwt_inverse(
    lf_addr: Int, n: Int, primary: Int, out_addr: Int, work_addr: Int,
    map_addr: Int,
) abi("C") -> Int32:
    """Recover the string from the last column by following the LF-mapping.

    This is what bzip2's decoder does: materialise the permutation, then walk
    it.  Returns 0, or -1 for a `primary` outside `0 .. n - 1`."""
    if n < 1:
        return Int32(0)
    if primary < 0 or primary >= n:
        return Int32(-1)
    var lf = up(lf_addr)
    var out = up(out_addr)
    _fill_lf(lf_addr, n, map_addr, work_addr)
    var jump = ip(map_addr)
    # Row `primary` is the rotation that starts at byte 0, so the byte it
    # carries is the last byte of the string; each LF step moves one byte
    # further back, so the walk fills the output from the end.
    var j = primary
    for k in range(n):
        out[unsafe_offset=n - 1 - k] = lf[unsafe_offset=j]
        j = Int(jump[unsafe_offset=j])
    return Int32(0)


# ---------------------------------------------------------------------------
# Run-length coding over the last column
# ---------------------------------------------------------------------------


@export("bwt_rle_encode")
def bwt_rle_encode(
    src_addr: Int, n: Int, values_addr: Int, lengths_addr: Int, count_addr: Int
) abi("C") -> Int32:
    """Split `src` into runs of equal bytes.

    Writes the run value to `values` and its length to `lengths`; both arrays
    must hold `n` entries.  Returns the number of runs."""
    var src = up(src_addr)
    var v = up(values_addr)
    var lens = ip(lengths_addr)
    var slot = ip(count_addr)
    if n < 1:
        slot[unsafe_offset=0] = 0
        return Int32(0)
    var runs: Int = 0
    var cur = src[unsafe_offset=0]
    var run: Int = 1
    for i in range(1, n):
        var c = src[unsafe_offset=i]
        if c == cur:
            run += 1
        else:
            v[unsafe_offset=runs] = cur
            lens[unsafe_offset=runs] = Int32(run)
            runs += 1
            cur = c
            run = 1
    v[unsafe_offset=runs] = cur
    lens[unsafe_offset=runs] = Int32(run)
    runs += 1
    slot[unsafe_offset=0] = Int32(runs)
    return Int32(runs)


@export("bwt_rle_decode")
def bwt_rle_decode(
    values_addr: Int, lengths_addr: Int, k: Int, out_addr: Int, out_words: Int
) abi("C") -> Int32:
    """Expand `k` runs.  Returns the expanded length, or -1 if it would not fit
    in `out_words`."""
    var v = up(values_addr)
    var lens = ip(lengths_addr)
    var out = up(out_addr)
    var pos: Int = 0
    for i in range(k):
        var run = Int(lens[unsafe_offset=i])
        if run < 0 or pos + run > out_words:
            return Int32(-1)
        var c = v[unsafe_offset=i]
        for j in range(run):
            out[unsafe_offset=pos + j] = c
        pos += run
    return Int32(pos)
