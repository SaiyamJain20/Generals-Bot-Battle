"""Stdlib forward pass for the BC student Net (BN folded), using big-integer SIMD lanes.

Every feature map is ONE Python int holding NP = 506 lanes of 32 bits (padded 22-stride layout of
pyfeat). A 3x3 conv tap is a bit shift of the whole map, so a conv output channel is
sum(map(mul, int_weights, shifted_maps)) in C. ReLU, rescale, bias and zero-padding masks are done on
all lanes at once with shifts/ands (lanes are offset by 2^30 so they stay non-negative). Activations are
fixed point with SA fractional bits, weights integers with K fractional bits.
"""
import struct
import sys
from array import array
from operator import mul

S = 21
ST = 22
NP = 23 * ST
LANE = 32
OFF = 1 << 30


def spec(ch, blocks):
    s = [("stem_w", ch, 24 * 9), ("stem_b", 1, ch)]
    for i in range(blocks):
        s += [(f"b{i}w1", ch, ch * 9), (f"b{i}b1", 1, ch), (f"b{i}w2", ch, ch * 9), (f"b{i}b2", 1, ch)]
    s += [("polw", 9, ch), ("polb", 1, 9), ("pasw", 1, ch + 8), ("pasb", 1, 1)]
    return s


def unpack(blob):
    """blob bytes -> (ch, blocks, {name: [rows of floats]})"""
    assert blob[0:1] == b"S"
    mode = chr(blob[1])
    ch, blocks = blob[2], blob[3]
    pos = 4
    out = {}
    for name, nr, rl in spec(ch, blocks):
        rows = []
        for _ in range(nr):
            if mode == "q":
                sc = struct.unpack_from("<e", blob, pos)[0]
                pos += 2
                v = struct.unpack_from(f"<{rl}b", blob, pos)
                pos += rl
                rows.append([x * sc for x in v])
            else:
                sz = 2 if mode == "e" else 4
                rows.append(list(struct.unpack_from(f"<{rl}{mode}", blob, pos)))
                pos += sz * rl
        out[name] = rows
    return ch, blocks, out


def _lanes(vals):
    a = array("I", vals)
    if sys.byteorder == "big":
        a.byteswap()
    return int.from_bytes(a.tobytes(), "little")


def _split(x):
    a = array("I")
    a.frombytes(x.to_bytes(NP * 4, "little"))
    if sys.byteorder == "big":
        a.byteswap()
    return a


class Net:
    def __init__(self, blob, SA=7, KS=14, KH=10):
        ch, blocks, a = unpack(blob)
        self.ch, self.blocks, self.SA, self.KS, self.KH = ch, blocks, SA, KS, KH
        self.FULL = (1 << (LANE * NP)) - 1
        self.ONES = _lanes([1] * NP)
        canv = [0] * NP
        for r in range(S):
            for c in range(S):
                canv[(r + 1) * ST + c] = 1
        self.CANV = _lanes(canv)
        self.OFFALL = self.ONES * OFF
        self.LOW30 = OFF - 1
        self.MK = {K: self.CANV * ((1 << (LANE - K)) - 1) for K in {KS, KH}}
        self.offs = [(k // 3 - 1) * ST + (k % 3 - 1) for k in range(9)]
        self.SHC = [self._shift(self.CANV, k) & self.FULL for k in range(9)]
        # stem
        sw = a["stem_w"]
        f = float(1 << (SA + KS))
        self.W0 = [[round(sw[o][k] * f) for k in range(9)] for o in range(ch)]
        self.WSC = [[[sw[o][(16 + s) * 9 + k] * f for s in range(8)] for k in range(9)] for o in range(ch)]
        self.QS = [[round(sw[o][(1 + c) * 9 + k] * f / 255.0) for c in range(15) for k in range(9)] for o in range(ch)]
        sb = a["stem_b"][0]
        self.BS = [self.OFFALL + self.CANV * (round(sb[o] * f) + (1 << (KS - 1))) for o in range(ch)]
        fh = float(1 << (SA + KH))
        self.blk = []
        for i in range(blocks):
            q1 = [[round(x * (1 << KH)) for x in a[f"b{i}w1"][o]] for o in range(ch)]
            q2 = [[round(x * (1 << KH)) for x in a[f"b{i}w2"][o]] for o in range(ch)]
            b1 = [self.OFFALL + self.CANV * (round(a[f"b{i}b1"][0][o] * fh) + (1 << (KH - 1))) for o in range(ch)]
            b2 = [self.OFFALL + self.CANV * (round(a[f"b{i}b2"][0][o] * fh) + (1 << (KH - 1))) for o in range(ch)]
            self.blk.append((q1, b1, q2, b2))
        inv = 1.0 / (1 << SA)
        self.polw = [[x * inv for x in row] for row in a["polw"]]
        self.polb = a["polb"][0]
        self.pasw = a["pasw"][0]
        self.pasb = a["pasb"][0][0]
        self.P0 = {}

    def _shift(self, x, k):
        off = self.offs[k]
        if off > 0:
            return x >> (LANE * off)
        if off < 0:
            return x << (-LANE * off)
        return x

    def _p0(self, H, W):
        key = (H, W)
        v = self.P0.get(key)
        if v is None:
            onb = [0] * NP
            for r in range(H):
                for c in range(W):
                    onb[(r + 1) * ST + c] = 1
            x = _lanes(onb)
            sh = [self._shift(x, k) & self.FULL for k in range(9)]
            v = [sum(map(mul, self.W0[o], sh)) for o in range(self.ch)]
            self.P0[key] = v
        return v

    def _act(self, Z, K):
        Z &= self.FULL
        M = (Z >> 30) & self.ONES
        return ((Z & (M * self.LOW30)) >> K) & self.MK[K]

    def _flat(self, xs):
        sh = self._shift
        out = []
        for x in xs:
            if x:
                out.extend([sh(x, k) for k in range(9)])
            else:
                out.extend([0] * 9)
        return out

    def features(self, P, scal, H, W):
        """-> list of ch bigints (final feature maps, SA fractional bits)."""
        ch, KS, KH = self.ch, self.KS, self.KH
        xs = [_lanes(P[c]) if any(P[c]) else 0 for c in range(1, 16)]
        flat = self._flat(xs)
        p0 = self._p0(H, W)
        f = float(1 << (self.SA + KS))
        h = []
        for o in range(ch):
            g = [round(sum(map(mul, self.WSC[o][k], scal)) ) for k in range(9)]
            acc = self.BS[o] + p0[o] + sum(map(mul, g, self.SHC)) + sum(map(mul, self.QS[o], flat))
            h.append(self._act(acc, KS))
        for q1, b1, q2, b2 in self.blk:
            flat = self._flat(h)
            y = [self._act(self.BS_add(b1[o], sum(map(mul, q1[o], flat))), KH) for o in range(ch)]
            flat = self._flat(y)
            h = [self._act(b2[o] + sum(map(mul, q2[o], flat)) + (h[o] << KH), KH) for o in range(ch)]
        return h

    @staticmethod
    def BS_add(b, s):
        return b + s

    def policy(self, P, scal, legal, H, W):
        """legal: sorted list of legal non-pass action indices. Returns (best_idx, best_logit)."""
        h = self.features(P, scal, H, W)
        HC = [_split(x) for x in h]
        inv = 1.0 / (1 << self.SA)
        pooled = [sum(a) * inv / (S * S) for a in HC]
        g = pooled + list(scal)
        pas = sum(map(mul, self.pasw, g)) + self.pasb
        best, bl = 9 * S * S, pas
        polw, polb = self.polw, self.polb
        lastcell = -1
        hv = None
        for idx in legal:
            cell, k = divmod(idx, 9)
            if cell != lastcell:
                r, c = divmod(cell, S)
                p = (r + 1) * ST + c
                hv = [a[p] for a in HC]
                lastcell = cell
            v = sum(map(mul, polw[k], hv)) + polb[k]
            if v > bl:
                bl, best = v, idx
        return best, bl

    def all_logits(self, P, scal, H, W):
        """Debug: full 3970-vector of logits (floats) for comparison with torch."""
        h = self.features(P, scal, H, W)
        HC = [_split(x) for x in h]
        inv = 1.0 / (1 << self.SA)
        out = [0.0] * (S * S * 9 + 1)
        for r in range(H):
            for c in range(W):
                p = (r + 1) * ST + c
                hv = [a[p] for a in HC]
                for k in range(9):
                    out[(r * S + c) * 9 + k] = sum(map(mul, self.polw[k], hv)) + self.polb[k]
        pooled = [sum(a) * inv / (S * S) for a in HC]
        out[-1] = sum(map(mul, self.pasw, pooled + list(scal))) + self.pasb
        return out
