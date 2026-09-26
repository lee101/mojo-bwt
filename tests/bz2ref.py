"""Reference decoder for the parts of the bzip2 container this port needs.

Written from the bzip2 1.0.8 `decompress.c` state machine, not from `mojo_bwt`,
so it is an independent implementation of the same format.  It recovers, for
one block:

* `last_column` -- the transformed bytes bzip2 actually emitted, and
* `orig_ptr`   -- bzip2's primary row index,

which is exactly the pair `mojo_bwt.decode` consumes.  RLE1, the first stage of
the pipeline, is undone here too so a test can compare the recovered block with
the file bytes that went in.
"""

from __future__ import annotations

import struct

_BZ_MAGIC = 0x314159265359
_G_SIZE = 50


class BitReader:
    __slots__ = ("data", "pos", "bit")

    def __init__(self, data: bytes, pos: int):
        self.data = data
        self.pos = pos
        self.bit = 0

    def get(self, n: int) -> int:
        v = 0
        for _ in range(n):
            byte = self.data[self.pos]
            v = (v << 1) | ((byte >> (7 - self.bit)) & 1)
            self.bit += 1
            if self.bit == 8:
                self.bit = 0
                self.pos += 1
        return v

    def align(self) -> None:
        if self.bit:
            self.bit = 0
            self.pos += 1


def _huffman(lengths: list[int]) -> dict[tuple[int, int], int]:
    """Canonical code assignment, matching `BZ2_hbCreateDecodeTables`."""
    alpha = len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    table: dict[tuple[int, int], int] = {}
    code = 0
    for length in range(min_len, max_len + 1):
        for sym in range(alpha):
            if lengths[sym] == length:
                table[(length, code)] = sym
                code += 1
        code <<= 1
    return table


def _decode_symbol(br: BitReader, table: dict[tuple[int, int], int],
                   min_len: int, max_len: int) -> int:
    """Read one canonical code, extending it while it exceeds the code space."""
    code = 0
    for _ in range(min_len):
        code = (code << 1) | br.get(1)
    for length in range(min_len, max_len + 1):
        sym = table.get((length, code))
        if sym is not None:
            return sym
        code = (code << 1) | br.get(1)
    raise ValueError("invalid bzip2 Huffman code")


def read_block_header(br: BitReader) -> tuple[int, int, int]:
    """Return `(orig_ptr, randomised, nblock_max)` after the block magic."""
    magic = 0
    for _ in range(48):
        magic = (magic << 1) | br.get(1)
    if magic != _BZ_MAGIC:
        raise ValueError("not a bzip2 block")
    br.get(32)  # block CRC, not needed to recover the bytes
    randomised = br.get(1)
    if randomised:
        raise ValueError("randomised bzip2 blocks are not supported")
    orig_ptr = br.get(24)
    level = br.data[3] - ord("0")
    return orig_ptr, randomised, 100000 * level


def read_symbol_map(br: BitReader) -> list[int]:
    """The bytes this block uses, in increasing order (`seqToUnseq`)."""
    used_groups = br.get(16)
    in_use = [0] * 256
    for group in range(16):
        if used_groups & (1 << (15 - group)):
            bits = br.get(16)
            for j in range(16):
                if bits & (1 << (15 - j)):
                    in_use[group * 16 + j] = 1
    return [i for i in range(256) if in_use[i]]


def read_last_column(br: BitReader, seq_to_unseq: list[int]) -> list[int]:
    """Undo the Huffman, move-to-front and RUNA/RUNB layers.

    What comes out is the last column of the block's rotation matrix, in the
    row order bzip2 stored it."""
    n_in_use = len(seq_to_unseq)
    eob = n_in_use + 1
    alpha = n_in_use + 2

    n_groups = br.get(3)
    n_selectors = br.get(15)
    selectors_mtf = []
    for _ in range(n_selectors):
        j = 0
        while br.get(1):
            j += 1
        selectors_mtf.append(j)
    selector = list(range(n_groups))
    selectors = []
    for j in selectors_mtf:
        sel = selector.pop(j)
        selector.insert(0, sel)
        selectors.append(sel)

    min_lens = []
    tables = []
    for _ in range(n_groups):
        curr = br.get(5)
        lengths = []
        for _ in range(alpha):
            while True:
                if curr < 1 or curr > 20:
                    raise ValueError("invalid bzip2 code length")
                if br.get(1) == 0:
                    break
                curr += 1 if br.get(1) == 0 else -1
            lengths.append(curr)
        min_lens.append(min(lengths))
        tables.append(_huffman(lengths))

    mtf = list(range(256))
    out: list[int] = []
    group_no = -1
    group_pos = 0
    state = {}

    def next_sym() -> int:
        nonlocal group_no, group_pos
        if group_pos == 0:
            group_no += 1
            group_pos = _G_SIZE
            sel = selectors[group_no]
            state["minlen"] = min_lens[sel]
            state["table"] = tables[sel]
        group_pos -= 1
        return _decode_symbol(br, state["table"], state["minlen"],
                              state["minlen"] + 23)

    sym = next_sym()
    while sym != eob:
        if sym in (0, 1):
            es = -1
            weight = 1
            while True:
                es += (sym + 1) * weight
                weight *= 2
                sym = next_sym()
                if sym not in (0, 1):
                    break
            es += 1
            out.extend([seq_to_unseq[mtf[0]]] * es)
            continue
        uc = mtf.pop(sym - 1)
        mtf.insert(0, uc)
        out.append(seq_to_unseq[uc])
        sym = next_sym()
    return out


def rle1_decode(block: bytes) -> bytes:
    """Undo bzip2's initial run-length stage: four equal bytes then a count."""
    out = bytearray()
    i = 0
    n = len(block)
    while i < n:
        run = 1
        while run < 4 and i + run < n and block[i + run] == block[i]:
            run += 1
        if run == 4:
            count = block[i + 4] if i + 4 < n else 0
            out.extend(bytes([block[i]]) * (4 + count))
            i += 5
        else:
            out.extend(block[i : i + run])
            i += run
    return bytes(out)


def last_column_of(stream: bytes) -> tuple[bytes, int]:
    """First block of a bzip2 stream: `(last_column, orig_ptr)`.

    That pair is exactly what a Burrows-Wheeler decoder consumes, and bzip2
    stores nothing else about the block, so recovering the block itself is the
    job of the kernel under test."""
    if stream[:3] != b"BZh":
        raise ValueError("not a bzip2 stream")
    br = BitReader(stream, 4)
    orig_ptr, _randomised, nblock_max = read_block_header(br)
    seq_to_unseq = read_symbol_map(br)
    column = read_last_column(br, seq_to_unseq)
    if len(column) > nblock_max:
        raise ValueError("bzip2 block overruns its declared size")
    return bytes(column), orig_ptr
