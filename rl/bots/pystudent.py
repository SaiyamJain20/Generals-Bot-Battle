"""Code Bot submission: behaviour-cloned tiny residual CNN (12 channels x 1 block, 6.8k params), pure stdlib.

participant: filled in by tools/build_submission.py   bot: pystudent
AI assistance: written with Claude Code (Anthropic). Network trained by behaviour cloning from our own
heuristic bots' games; weights are embedded below (BN folded, float16, base64).
Sources: generals.bot environment rules; no third-party code. The forward pass runs on big-integer SIMD lanes
(each feature map is one Python int), features are a stdlib port of our numpy feature extractor.
"""
import base64
import gc
import math
import struct
import sys
from array import array
from operator import mul

S = 21
ST = 22
NP = 23 * ST
NPLANES = 16
NACT = S * S * 9 + 1
PASS_IDX = S * S * 9
_LGC = {}

def lg(a):
    v = _LGC.get(a)
    if v is None:
        v = 0 if a <= 0 else min(255, int(math.log1p(a) * 32.0))
        _LGC[a] = v
    return v

def _flat(g):
    if g and isinstance(g[0], (list, tuple)):
        out = []
        for row in g:
            out.extend(row)
        return out
    return list(g)

class Tracker:

    def __init__(self, H, W):
        self.H, self.W = (H, W)
        n = H * W
        self.mountain = bytearray(n)
        self.castle = bytearray(n)
        self.last_enemy = bytearray(n)
        self.last_enemy_army = [0] * n
        self.last_seen = [-1] * n
        self.egen = None
        self.t = 0
        self.pos = [(r + 1) * ST + c for r in range(H) for c in range(W)]

    def update(self, obs):
        """Returns (planes: 16 lists of NP ints (0..255), scal: list of 8 floats, (O, A, T) flat lists)."""
        H, W = (self.H, self.W)
        n = H * W
        T = _flat(obs['type'])
        O = _flat(obs['owner'])
        A = _flat(obs['army'])
        t = int(obs['turn'])
        self.t = t
        mount, castle = (self.mountain, self.castle)
        first = t == 0 or not any(mount)
        last_seen, last_enemy, lea = (self.last_seen, self.last_enemy, self.last_enemy_army)
        pos = self.pos
        P = [[0] * NP for _ in range(NPLANES)]
        P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15 = P
        structs = []
        egen = None
        for i in range(n):
            ty = T[i]
            o = O[i]
            if first:
                if ty == 2 or ty == 5:
                    mount[i] = 1
            elif ty == 2:
                mount[i] = 1
            if ty == 2:
                mount[i] = 1
            if ty == 3 or (ty == 5 and (not mount[i])):
                castle[i] = 1
            if ty != 0 and ty != 5:
                last_seen[i] = t
                if o == 2:
                    last_enemy[i] = 1
                    lea[i] = A[i]
                else:
                    last_enemy[i] = 0
                    lea[i] = 0
                if ty == 4 and o == 2 and (egen is None):
                    egen = (i // W, i % W)
                if o == 1 and (ty == 3 or ty == 4):
                    structs.append((i // W, i % W))
        if egen is not None:
            self.egen = egen
        eg = self.egen
        sur = {}
        for r, c in structs:
            for rr in range(max(0, r - 6), min(H, r + 7)):
                dr = abs(rr - r)
                for cc in range(max(0, c - 6), min(W, c + 7)):
                    v = 14 - 2 * (dr + abs(cc - c))
                    if v > 0:
                        k = rr * W + cc
                        sur[k] = sur.get(k, 0) + v
        for i in range(n):
            p = pos[i]
            ty = T[i]
            o = O[i]
            P0[p] = 255
            if mount[i]:
                P1[p] = 255
            vis = ty != 0 and ty != 5
            if o == 1:
                P2[p] = 255
                P6[p] = lg(A[i])
            elif o == 2:
                P3[p] = 255
                P7[p] = lg(A[i])
            if vis and o == 0 and (not mount[i]):
                P4[p] = 255
            if not vis:
                P5[p] = 255
                if last_enemy[i]:
                    ag = t - last_seen[i]
                    P11[p] = min(255, max(0, 255 - ag * 2))
                v = lea[i]
                if v:
                    P12[p] = lg(v)
            if castle[i]:
                P8[p] = 255
            if ty == 4 and o == 1:
                P9[p] = 255
            ls = last_seen[i]
            ag = t - ls if ls >= 0 else 999
            P13[p] = ag if ag < 255 else 255
            s = sur.get(i, 0)
            if s:
                P14[p] = min(255, s * 4)
            if o == 1 and ty == 1 and (A[i] >= 35 + s):
                P15[p] = 255
        if eg is not None:
            P10[pos[eg[0] * W + eg[1]]] = 255
        my_a, op_a = (max(1, int(obs['my_army'])), max(1, int(obs['opp_army'])))
        my_l, op_l = (max(1, int(obs['my_land'])), max(1, int(obs['opp_land'])))
        scal = [t / 1200.0, t % 50 / 50.0, float(t >= 800), math.log(my_a / op_a), math.log(my_l / op_l), math.log(my_a) / 8.0, math.log(op_a) / 8.0, float(eg is not None)]
        return (P, scal, (O, A, T))

def dense(P, H, W):
    """Padded planes -> list of 16 lists of 441 (reference layout)."""
    out = []
    for pl in P:
        d = [0] * (S * S)
        for r in range(H):
            b = (r + 1) * ST
            d[r * S:r * S + W] = pl[b:b + W]
        out.append(d)
    return out

def action_index(a, W=None):
    k, r, c, d, sp = a
    if k == 1:
        return PASS_IDX
    if k == 2:
        return (r * S + c) * 9 + 8
    return (r * S + c) * 9 + d * 2 + (1 if sp == 1 else 0)

def index_action(idx):
    if idx == PASS_IDX:
        return [1, 0, 0, 0, 0]
    cell, k = divmod(idx, 9)
    r, c = divmod(cell, S)
    if k == 8:
        return [2, r, c, 0, 0]
    d, sp = divmod(k, 2)
    return [0, r, c, d, sp]
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

def legal_moves(O, A, T, H, W, planes):
    """Sorted list of legal action indices (excluding PASS, which is always legal).
    Same set as the numpy legal_mask."""
    out = []
    P15 = planes[15]
    for r in range(H):
        for c in range(W):
            i = r * W + c
            if O[i] != 1:
                continue
            a = A[i]
            base = (r * S + c) * 9
            if a >= 2:
                for d in range(4):
                    dr, dc = _DIRS[d]
                    rr, cc = (r + dr, c + dc)
                    if 0 <= rr < H and 0 <= cc < W:
                        ty = T[rr * W + cc]
                        if ty != 2 and ty != 5:
                            out.append(base + d * 2)
                            if a >= 3:
                                out.append(base + d * 2 + 1)
            if P15[(r + 1) * ST + c]:
                out.append(base + 8)
    return out

S = 21
ST = 22
NP = 23 * ST
LANE = 32
OFF = 1 << 30

def spec(ch, blocks):
    s = [('stem_w', ch, 24 * 9), ('stem_b', 1, ch)]
    for i in range(blocks):
        s += [(f'b{i}w1', ch, ch * 9), (f'b{i}b1', 1, ch), (f'b{i}w2', ch, ch * 9), (f'b{i}b2', 1, ch)]
    s += [('polw', 9, ch), ('polb', 1, 9), ('pasw', 1, ch + 8), ('pasb', 1, 1)]
    return s

def unpack(blob):
    """blob bytes -> (ch, blocks, {name: [rows of floats]})"""
    assert blob[0:1] == b'S'
    mode = chr(blob[1])
    ch, blocks = (blob[2], blob[3])
    pos = 4
    out = {}
    for name, nr, rl in spec(ch, blocks):
        rows = []
        for _ in range(nr):
            if mode == 'q':
                sc = struct.unpack_from('<e', blob, pos)[0]
                pos += 2
                v = struct.unpack_from(f'<{rl}b', blob, pos)
                pos += rl
                rows.append([x * sc for x in v])
            else:
                sz = 2 if mode == 'e' else 4
                rows.append(list(struct.unpack_from(f'<{rl}{mode}', blob, pos)))
                pos += sz * rl
        out[name] = rows
    return (ch, blocks, out)

def _lanes(vals):
    a = array('I', vals)
    if sys.byteorder == 'big':
        a.byteswap()
    return int.from_bytes(a.tobytes(), 'little')

def _split(x):
    a = array('I')
    a.frombytes(x.to_bytes(NP * 4, 'little'))
    if sys.byteorder == 'big':
        a.byteswap()
    return a

class Net:

    def __init__(self, blob, SA=7, KS=14, KH=10):
        ch, blocks, a = unpack(blob)
        self.ch, self.blocks, self.SA, self.KS, self.KH = (ch, blocks, SA, KS, KH)
        self.FULL = (1 << LANE * NP) - 1
        self.ONES = _lanes([1] * NP)
        canv = [0] * NP
        for r in range(S):
            for c in range(S):
                canv[(r + 1) * ST + c] = 1
        self.CANV = _lanes(canv)
        self.OFFALL = self.ONES * OFF
        self.LOW30 = OFF - 1
        self.MK = {K: self.CANV * ((1 << LANE - K) - 1) for K in {KS, KH}}
        self.offs = [(k // 3 - 1) * ST + (k % 3 - 1) for k in range(9)]
        self.SHC = [self._shift(self.CANV, k) & self.FULL for k in range(9)]
        sw = a['stem_w']
        f = float(1 << SA + KS)
        self.W0 = [[round(sw[o][k] * f) for k in range(9)] for o in range(ch)]
        self.WSC = [[[sw[o][(16 + s) * 9 + k] * f for s in range(8)] for k in range(9)] for o in range(ch)]
        self.QS = [[round(sw[o][(1 + c) * 9 + k] * f / 255.0) for c in range(15) for k in range(9)] for o in range(ch)]
        sb = a['stem_b'][0]
        self.BS = [self.OFFALL + self.CANV * (round(sb[o] * f) + (1 << KS - 1)) for o in range(ch)]
        fh = float(1 << SA + KH)
        self.blk = []
        for i in range(blocks):
            q1 = [[round(x * (1 << KH)) for x in a[f'b{i}w1'][o]] for o in range(ch)]
            q2 = [[round(x * (1 << KH)) for x in a[f'b{i}w2'][o]] for o in range(ch)]
            b1 = [self.OFFALL + self.CANV * (round(a[f'b{i}b1'][0][o] * fh) + (1 << KH - 1)) for o in range(ch)]
            b2 = [self.OFFALL + self.CANV * (round(a[f'b{i}b2'][0][o] * fh) + (1 << KH - 1)) for o in range(ch)]
            self.blk.append((q1, b1, q2, b2))
        inv = 1.0 / (1 << SA)
        self.polw = [[x * inv for x in row] for row in a['polw']]
        self.polb = a['polb'][0]
        self.pasw = a['pasw'][0]
        self.pasb = a['pasb'][0][0]
        self.P0 = {}

    def _shift(self, x, k):
        off = self.offs[k]
        if off > 0:
            return x >> LANE * off
        if off < 0:
            return x << -LANE * off
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
        M = Z >> 30 & self.ONES
        return (Z & M * self.LOW30) >> K & self.MK[K]

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
        ch, KS, KH = (self.ch, self.KS, self.KH)
        xs = [_lanes(P[c]) if any(P[c]) else 0 for c in range(1, 16)]
        flat = self._flat(xs)
        p0 = self._p0(H, W)
        f = float(1 << self.SA + KS)
        h = []
        for o in range(ch):
            g = [round(sum(map(mul, self.WSC[o][k], scal))) for k in range(9)]
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
        best, bl = (9 * S * S, pas)
        polw, polb = (self.polw, self.polb)
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
                bl, best = (v, idx)
        return (best, bl)

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

_BLOB = (
    'U2UMAaIuCzWWLbUzHzLhMN0vZTAaMXEmCbA0Jd+xGLV4tIQkgrS8Kquxui6usvYshrzkLL+xCzGgtHKsuS/nqEmwWjVlqoeu8J7u'
    'HguuK7FWrNWxrDUotLmvv7Tdp/qoMq50p5mhL69mqiwsdKwCpRy2+7gxqzi7TbgPvQ6zELuCNqw2m7xCqQS27DS5uImxYLwsuVs4'
    'cTsYPGY5CUCRPOwzejnaOPe4r7d/tG23VrWct2awsLoKKjY7ADxtOHw8nD4WOgk4dTTKL98zNyi+HmquTrb0sLcw9zCfMv+m5jQJ'
    'JCwuaDK2LwYzkTRGqgIldjB4rlkfBqwKMf6obCXRKWquK7wFtI67t759ub8xlLreIcc3JbaoNF60OzY9tEw4obaVNXEv5KqlskUi'
    'FK7erKWrNKy8tZqtDC3bL6wphy5lqhIq+SbeLXkoTygVLZgy5imGqpgwNas9Lf2qfSPQqx0ofS+IIEItxC6dqWUzviUksDSxYSl1'
    'rWqdIjA/qLCtW6uErz60Nq3wsUCxpi73rMAt1KprLlkpu68GKSWvw6exrosk6jH7lFmpfqpnp3axKCeLr4UxerCxIF0rV66SsiGw'
    'irEIrKaiOTEoJ6gwCzAMo80sEhMpK+MteTEvMKU0dDqtNEMy1zHsqSi0l7Yesvu6yL1quiyro7JssnCkpDFELiMw1bL4qBQoxC3v'
    'K6yuqjJCsM0so7WyMVSiA63Xq0O95rFguAO/A75XwEO5/b58Ozu7CLxrunS9T73IvLa3HLllpqooWhp1LQo3kT0CLn41mjIosg0v'
    'LrT9LAc6tTxZL4g3QDIvsRW6v7auttazabhssKW1srjjssWyxKaZspWujbI8tf+137BNsoyxRiqesvWtzDC3rJaxjLV1sS+gNTFW'
    'M+YzBrjzpMOsCiwFNHM17TZHLl0qLDYFN0EkxrM0NLG25CWXOHGrprQ9rWyrFji1Km+vuyBCtMWsOTX2MK0vTTTzMeiwkjCcsVg0'
    'ojbkqmypGDQytFgz2BduItowXbBvoC6xRzFktLWylDParmYzLTZnsZEvoqpSNGKrMzDusmswKKEPrhiuULXOL3qwqC4bML8mszrq'
    'LqCzYjWuLkK1sjAWsNgxaziLLcq2lzG5rncukSrVslOxg6yzNAMuSajOqq4w2TObLl806S+0Mj6lR621KY0toauFJ7Wv7ip3Ihow'
    'zqvwox2tA6PnsBUzczfbrgcYiSw8shklFB0ZrzGwhzAwLl+WFLArrygti6viqKet860asKwvza5qoygtUrN4rP2y/rWIAGMkmK4w'
    'sP07JDesrvS6fT0DPsUto72yNZch66zLuse2cTw2KcO25LgvLAwqC53YIEgrBLH4KXEuJqy9IQe0YKxUty61hbbOtzat6rUutqon'
    'kDPNNFeyljrYOBm5yDKAsAMyATHOsYszziywHDWtKqwcLRq4Ia97LEI3lqn9Nfm2oq0xJ9EsJK9jKkqs/rb1sqEiOK/XMG4t0rBE'
    'KCCwa50uMAWwQLJoNlK6VrBJJ3og57Fat3mpLit/sMc0FrShIMsuvKn9quKgfa2mMUgd6K4lsDe1krXys+Yyf7RYLbaxA7EUMFOr'
    'Di32nAcbuhthMq0rxC6QKrEcf7AsK9Esg68BqUazkKvwMGosJCvKJqQnuSASKMSvWLLyrP4pu7UAtXQnhrTHlmswn6S7KAuxC7ZA'
    'sLqyMLQOM/Cq+K2UIOkuxCqeqOAtUa5wMDwyLi4DL5ww/jkSMn+UvjS1KrWo0yyortgxf6sqqdmtxw7aKiqreCzbqmyx/DXDMSS0'
    'Azeqr0My5ir1rAAy7bilsi40jjXXKImrOZzSrl0oPqwZswqyXbPKKHKt0KQtqp4sLrDPqVqzwa8cMYA4DyvbOCRAbMHTsm02/TnG'
    'O6A0azhdNnY8/7iPOzUstz3pNmiz/TE0Mu+zlb0KsbAv9rHPtEi5lbbXtfa33b1iuJ+50LrNuPk4KzkZOq07AritOK04ZDmGM3Qt'
    'hC2uGSuwkyxENuqiGK8kN7424bXMMFC1lKxZrX4wOi6vN4GxH6b7oAshqrYaqd8vD7deIUw2jS57sdOvAhQttSisfKmptCu26Tid'
    'rWo3YzRFOIqp0TUIsNwqfTQ4KaAvOzRdLPouOjHzMpk1AjTkL+4xMjKEGI6x67BQrR0wI6oTtcAjBC25q1MunaxNrDGt7q9ysjGy'
    'Xy/XMSKsH66kKrAK8Cq3Mi2uSq34rz0wWTLeHki1qLBetMa11SgjrQO4rTGRJQSxbCq1sZWoRDLxrQW2mi3rJS8vwSt6r4m0kaQG'
    'L5Ajx6xvMpKw3bAUtVuzSbN1rkusgSjCsBChLCamL3IvyJXjq7EsWDAmMOspJrNaqyCzP6u0r1mnfa9brou0v6wJs9izRDd4tYa0'
    '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'
    'Li4pKlSuArBkKPutW6yar8qy9rEqsbuwU7DiKGMrJzIKqHmwozF0HdovkTCoKZSv/qwDsMkhojOGqQ2pMSzlsWIvDTGVL0mzsC0s'
    'saKjUC9HsKorji9vrUssH6wUKdMxWSsbqZIqfDEypqeoaaj2rwwdsDDTro8xkrHuqbIyYrX4oI80O7GKsXMoMTBMoZ+vezAZro4s'
    'ZTLgrQunTrQjp8atKLRyqNGvgrEiKNiwyy8+Lfm2wrCInMmxBzFVLbyxJ7EJMAmzDzEyMoOi46HYNQEukLEGqFWs8RmzLIEqzrIT'
    'KyktWrDOKQcw0LO5LEsewLDFLPktXyyAMOeoLi9bqyEwLTXPLcil2Kczq7auMrKNqpCoSyy4saupna/zro4obrQqsqk0rjeNM2It'
    'ORa5LuUubDDFMx0eqauDLjyrrCunpc0srjYTNAI0szResF8mjZ8FrFEjDLYCqDswGrNWMFuuqTJXKmioqDO4Mqes3zAJLFygTLJo'
    'pim0KLR/qRcyJ7aTr0qtBDO4L++s0DDyLyG1Fbhur7Eq6aD+L8AzMTZEMZmvMSs1pjav+6MHrQK1lqjVsJIuFK6hKl8vHSybLtAw'
    'C6x/LnOdRa5zpPuwlBzMNlkwCK82NGuwlTTxNW4pNq4XsUAsWaasoz2iba0MtAuszTQ/MtmxoTEuNQWpJS2orw+xazK3qqOvNDcc'
    'tGSs9qgynB60i6MML40wHK/Dnbmtc6XILwqvCS/IJyQuZrABLie0mTDLLz8tKbKHsGkwrrU+IM6rO7CANrewsy56MbwsbKVWrmAZ'
    'uDNIL1eiM7YGtYQ4gjQ6NbIzUi8bswgwCzRYtncvnDBKjHgp9Ch3rk+p4jGjMHozGKpmtg4wNzCpql4YbrAxqOuwybOaJaawwKNu'
    'q+AysjR9qjEyzzTgryk0NzJnmTyxEq95MFW1b7GNp6isc7NqqlMwjSkCJ+ioGjAIr0eiVbCrHASsOSzdLvChs7GpLm2soizrJLMn'
    'JSzsILwtzi1PqKuxVTG2rDSztzPtsbapkDVQrpWvMzBjLLkx6bF5nlAqgrAIq6InJDH7nacwqDEnqYYfQDV9n+UwbTEiMEmh4rZA'
    'qRMtqiizMWgyA6RRKHmsTLBFqycp0y/pLCWvyDKfJMgwRzKrMaEjKCYvLtkuVCoLpKKxQLT4rJSy6bStphirZ6sFrBOyTrjPsw0y'
    'cy8nsnQmDDCNKVc1GyjxLv2xgiqTMhywD7LOqdmrSLLPJJwtIqp0MN4vYSYfrC6t8TOXs3WqGybgKhmlTq2yLWev0LYCrKmuj7Cf'
    'K6CsQTAEKay0lzI/ruYrh6z6qnUqdSvoKGYsmLWQrkiy+jKBKTUwl6hHLaicuSa+rLGxujP5rJartSpWqfo02yuVNBEnQSEPqQyw'
    '36pPqAOsEq6usX4qCi8bKw+pNynPqcgzYqsBMv6rHTTTLQAk6y1aJ1QlHTTIsVCwQqRjrRMsDLBpqhI0WjZDMaIwGbAgrRSzQ7QK'
    'rciwti2vtMCxK661sj2VtyczMF0xUDOOMX+y1qZzMlCsrLD2pn+vFKOYNJcqnqXgMEewN7HdLbWwwiyZKUOb1y5kqXis8xq6JTQq'
    'NLLvor2xGDD1rg4s9i87q2QygzYbKmazP59ztGStcaBlpfkzOKpmMrCj2rGDJoytE6Varc2xN64jtGiseTAdMIYwGDCDK+KyCLA8'
    'IW2qxzFzJSikn6XbJowxaCglqK4nm7BmGRoxDixqMoGwKbNStJw0saw9qeEpK7DNFuEubjNKNaKt/S10tI6xbbWAs2Gt3bPro4op'
    'GDbCEJCjTDJSMoEpUClNMgkwfy7zrrIoZqvdmrmfRTG7r7QpgiwkL+qnr64rsCWkszINJ2atDy4QqoquDK6xrmuVA6SGqyYs06SL'
    'LTmk+Cx+LE4m2izCK2UnGDN8sC6sObDXJkyglCv+LiCvkKAurR4qILF4K4sqYykDqQEskS8gI6sqNi/ZpIqlKrNKI0Ih+xRVLbUg'
    'ozBArVMqOaq/p+yh8DQysKGpRR9aLSmdm6/QIEisVR7cIrYl+620JpGmHbSItmwxHjQOtVg0hSjhtFervDHzLAWutKwHKBGwfxdq'
    'NPUn4KoDnaI1xilMJi6mwC97MFoz3zCvsBEo+LIAJEQq7ywEKF6rfqY0qAssvzC4piQqYy5GsVUpki9UJykwaqGaKwmp7y/Iq+Kk'
    'UK3SslcxRqgaq9kjUqiRqVUp96ybrRojZSVQJlAy6SqCreystCkCLPywWSuQKH6guCduqTaeUzJaofGlkKPSrKSmxCkFKUGqki9u'
    'KjKttyu9MKko+LPFLykhXikGGduw0bKVqd8spLFEMfAxF7Edr46d+bckqQ0wjK8NKHC1KTG3MAQtVjncMyWxpqw0niCtAzEHLI+3'
    '/So2LXmxWyFVrxG1ejDpstSomiH2tKGu/bFJs9Gy/7RxNHSwuLQUuXUgobXuMoiryi/0L+itRzAKMHivSzIjpoQ0mbIJtEIz2bSf'
    'p+GswSyKMia0ySX2Keyxb7SXMsapCLYZsYQon7T9tEE5+7ALNzksEjINtCIqGS8ysvMreLA2MvMur6WUMWChoTPdL6E0GShsrRap'
    '3zempHSuC7CKrI+vkTE0shWutrQKIDctZbItMFKoyC4RsP60MbQHrywwNrIwsH8rALNmsroo6izIoIc0wLGGqIgy4TVXtLCrrzFg'
    'LFAvySpmq5wp4i3ALwaZlK67JxStcDNLplgwN7c4MrmPsCxUnOasY6cBqE40czgRq8IupzVaLzOz7y/lMAC4vTYdrvww3agMs4ws'
    'ZCvprekqqbR5stCtRy78KUSxcDWPMvq4ITXNs8urFLTBMGi0xa9Fr2+naDFRI4ItGzIBpSwyYbKksRG09Ta0L3ky/rd4KcupzLMs'
    'qIWw2q3CrHCwuLFVqZewB7RwsJ4gMbWkqU+gITT0rUuVdzHTsrwwjK+pr3Cmk7DBHH+uOzRjJqQYCa7AMCocnbCJLnEx3jC5NMWs'
    'gyvALmOvbLLSJ40swK0BsmGqQKy5sXKtpS0LLKEnqbB4LPirhjUMMJSlPLCFMqQqC6VHrOym/7BZJyQqLysapY6bMStYL1Mt76MD'
    'rY+wnjEwqYswR67OpJUwnKutorUwWTAIr0ysVjPFrqciwSUFLnmqpjDDhBywqqwAsDAuWaRErEUwOC5SKIqsqTB5JRqsFihIrI8n'
    'brNCKMIv8TEou468TbrWLoayILVrsBs2mDXzJcKoqLEYKTW0CLF7MSKpFqqXrAUyqa3fIXUwdzWzNCov8KBRLeSw1rKgKDiqQ7O8'
    'LN8eGLT8o8axc7AyMYu2kLEyMfO0vyrtKTGs2Ss0KMuhGLP3Lmgqg7HBr++tUTGSMU4xbCTzIHywjzCRNDKwl65IrzWqRKwQseut'
    'jjEZL+OoQqHgMve0BjBiMLkr2jI/Mq0jV7IknZu0rTF+ocCxA68SrHeqnC/yLqewFqklNHIvWzMmMToo/azCrK2tyLV8NtGuLa17'
    'JTmwmCx1MAmwJbChrkixrLHFMBCxlCSDNUguZy3TNKGo9Ti3Mhy1ky3FsreuO7DttK2rbjRmMrmstSKRNTS0DTWDLZWyATF1m3mo'
    'WjG0L4OycrSHr3KVRrC3NC2zS7jHrc60E7lmstmvGSzQNOwmpRlYsdssoyqtNP80Kyvyr7EmzbAztD2uULhotcQtu6UENPoyL6iI'
    'q+0v1agTr78kuqxLq3IwXLGYpQ4X1DQrMv+qETNpICKwD7V0q+i0szSVtDcsOrReovCqlbFIrDmxsLLxrI6v/y0dqhKzFC18MWuu'
    'bzY7ou4zW6jwMYEfH7ILqwwmcK9NLTmptDHYNdMxYbBSNCMvPKhXsOAoMCRMNIWzZLElMC2pBaVGnyawxCTwMKKaPCXzMPIxtbtt'
    'NKkwIjBeLfCtlDA7NIqzS7C7NQW0S7DPMd2l5zMJL2qtnyx1pSw3NzSWteezPrNALm+z7rS+MNimaacwLryo7a5hrAe3BrDgsKgm'
    '4qhaLX8nyrKssh22kKtBq+O2ibExsxg0mi7isTE4bi6lsvE10pKItbsunif5rX40GDOEs5ax5iW3rcazpyxoLYgpG6i2IieuMbNK'
    'JcWpB7PCoxqoNbS+rNy2Yyk0pcysDS2VtKy45TEzuByogDC6rTqnc7T9MeAq/bRUssUsnK3OKauypKzdrBWuBzlxqEQK8rEorZSy'
    'eqZDs7Sq+y1NM6SuL7LOsi4nsat3psswGDPaLZEncDgWtTQvLag/qvsy7S2AsyAklLTVKAElPTKnKTM45C1WMDwyQi6tLMCmFatx'
    'L9wzojADMSEwzjEYMDshADMJKOMye7kCLGWxQjN1sHGtwaVEIFauh7qVKYurlaYcqwytFrDdpK+uaDatsE+eJCmgL0Or/jQ1MWKp'
    'yTQ9LXMtJy/uMN2yAiWvLhawDzKzqb6zl6pjLSy5HipuLrS1CRj2qdmmNS0VKji09qfSJUc5rrdGr7at16l2KSGw+iwYNGS0TTBT'
    'NsK0nLeIGwQqMrJzKuMXarQ1L4ExKqxttQawbjB0L/Ow6bRlrXuwqqjgLBosXSZeMJQ2GCWkLvQp76mQp6Wt1JcVMh+4vyAGtdMy'
    'vCULogQwQyt6Lqc4riaVq84yZCy0LdUpyzD0q8614jH7ISctPDNAsBSs5a0HL7M0iK41Lgo2ojV3MoSwybA/stA5eDAwM+ackrMN'
    'tAUyICxFMY4tBbeqLlYwwyhMNAI2Bqs2K9CkrrTOqc2qmbCxpXMvz6LipvAgZblNMOGq1hr+rSAv8rA/qVQwUiZjrFIxHLDisVqs'
    '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'
)


_NET = None
_TRK = None
_FALLBACK = [1, 0, 0, 0, 0]


def _setup():
    global _NET
    _NET = Net(base64.b64decode(_BLOB), 8, 14, 10)
    _NET._p0(21, 21)
    _NET._p0(18, 18)


def act(observation):
    global _TRK
    try:
        obs = observation
        H, W, t = int(obs["height"]), int(obs["width"]), int(obs["turn"])
        if _NET is None:
            _setup()
        if _TRK is None or t == 0 or _TRK.H != H or _TRK.W != W:
            _TRK = Tracker(H, W)
        P, sc, (O, A, T) = _TRK.update(obs)
        legal = legal_moves(O, A, T, H, W, P)
        idx, _ = _NET.policy(P, sc, legal, H, W)
        return [int(v) for v in index_action(idx)]
    except Exception:
        return list(_FALLBACK)


try:
    _setup()
except Exception:
    pass
gc.disable()
