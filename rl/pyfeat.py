"""Stdlib port of learn/bc_features.py (Tracker, legal moves, action indexing).

Planes are produced in a PADDED flat layout used by the integer inference engine:
canvas cell (r, c) lives at index (r + 1) * ST + c with ST = 22 (one guard column that doubles as
left/right zero padding, one guard row above and below), total NP = 23 * 22 = 506 entries per plane.
`dense()` converts back to the 16 x 441 layout of the numpy reference for parity testing.
"""
import math

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
        self.H, self.W = H, W
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
        H, W = self.H, self.W
        n = H * W
        T = _flat(obs["type"])
        O = _flat(obs["owner"])
        A = _flat(obs["army"])
        t = int(obs["turn"])
        self.t = t
        mount, castle = self.mountain, self.castle
        first = t == 0 or not any(mount)
        last_seen, last_enemy, lea = self.last_seen, self.last_enemy, self.last_enemy_army
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
            if ty == 3 or (ty == 5 and not mount[i]):
                castle[i] = 1
            if ty != 0 and ty != 5:
                last_seen[i] = t
                if o == 2:
                    last_enemy[i] = 1
                    lea[i] = A[i]
                else:
                    last_enemy[i] = 0
                    lea[i] = 0
                if ty == 4 and o == 2 and egen is None:
                    egen = (i // W, i % W)
                if o == 1 and (ty == 3 or ty == 4):
                    structs.append((i // W, i % W))
        if egen is not None:
            self.egen = egen
        eg = self.egen
        sur = {}
        for (r, c) in structs:
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
            if vis and o == 0 and not mount[i]:
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
            if o == 1 and ty == 1 and A[i] >= 35 + s:
                P15[p] = 255
        if eg is not None:
            P10[pos[eg[0] * W + eg[1]]] = 255
        my_a, op_a = max(1, int(obs["my_army"])), max(1, int(obs["opp_army"]))
        my_l, op_l = max(1, int(obs["my_land"])), max(1, int(obs["opp_land"]))
        scal = [t / 1200.0, (t % 50) / 50.0, float(t >= 800), math.log(my_a / op_a),
                math.log(my_l / op_l), math.log(my_a) / 8.0, math.log(op_a) / 8.0,
                float(eg is not None)]
        return P, scal, (O, A, T)


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
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        ty = T[rr * W + cc]
                        if ty != 2 and ty != 5:
                            out.append(base + d * 2)
                            if a >= 3:
                                out.append(base + d * 2 + 1)
            if P15[(r + 1) * ST + c]:
                out.append(base + 8)
    return out
