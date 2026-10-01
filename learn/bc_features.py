"""Observation -> tensor features for the behaviour-cloned sparring bot.

Works from event-style observation dicts only (so the same code runs inside
the opponent bot), keeping a small fog memory. Spatial planes are uint8 on a
21x21 canvas; scalars are a small float vector.
"""
import math

import numpy as np

S = 21
NPLANES = 16
NSCAL = 8
NACT = S * S * 9 + 1   # per cell: 4 dirs x 2 splits + build ; plus pass
PASS_IDX = S * S * 9


def lg(a):
    return np.minimum(255, (np.log1p(np.maximum(a, 0)) * 32.0)).astype(np.uint8)


class Tracker:
    def __init__(self, H, W):
        self.H, self.W = H, W
        self.mountain = np.zeros((H, W), bool)
        self.castle = np.zeros((H, W), bool)
        self.last_enemy = np.zeros((H, W), bool)
        self.last_enemy_army = np.zeros((H, W), np.int32)
        self.last_seen = np.full((H, W), -1, np.int32)
        self.egen = None
        self.t = 0

    def update(self, obs):
        H, W = self.H, self.W
        T = np.asarray(obs["type"], dtype=np.int32).reshape(H, W)
        O = np.asarray(obs["owner"], dtype=np.int32).reshape(H, W)
        A = np.asarray(obs["army"], dtype=np.int32).reshape(H, W)
        t = int(obs["turn"])
        self.t = t
        vis = (T != 0) & (T != 5)
        if t == 0 or not self.mountain.any():
            self.mountain |= (T == 2) | (T == 5)
        self.mountain |= T == 2
        newstruct = (T == 5) & ~self.mountain
        self.castle |= (T == 3) | newstruct
        self.last_seen[vis] = t
        self.last_enemy[vis] = O[vis] == 2
        self.last_enemy_army[vis] = np.where(O[vis] == 2, A[vis], 0)
        eg = np.argwhere((T == 4) & (O == 2))
        if len(eg):
            self.egen = (int(eg[0][0]), int(eg[0][1]))
        planes = np.zeros((NPLANES, S, S), np.uint8)
        planes[0, :H, :W] = 255
        planes[1, :H, :W] = self.mountain * 255
        planes[2, :H, :W] = (O == 1) * 255
        planes[3, :H, :W] = (O == 2) * 255
        planes[4, :H, :W] = (vis & (O == 0) & ~self.mountain) * 255
        planes[5, :H, :W] = (~vis) * 255
        planes[6, :H, :W] = lg(np.where(O == 1, A, 0))
        planes[7, :H, :W] = lg(np.where(O == 2, A, 0))
        planes[8, :H, :W] = self.castle * 255
        planes[9, :H, :W] = ((T == 4) & (O == 1)) * 255
        if self.egen is not None:
            planes[10, self.egen[0], self.egen[1]] = 255
        age = np.where(self.last_seen >= 0, t - self.last_seen, 999)
        planes[11, :H, :W] = (self.last_enemy & ~vis) * np.clip(255 - age * 2, 0, 255).astype(np.uint8)
        planes[12, :H, :W] = lg(np.where(~vis, self.last_enemy_army, 0))
        planes[13, :H, :W] = np.clip(age, 0, 255).astype(np.uint8)
        # castle price surcharge from own structures
        own_struct = np.argwhere((O == 1) & ((T == 3) | (T == 4)))
        sur = np.zeros((H, W), np.int32)
        rr, cc = np.mgrid[0:H, 0:W]
        for r, c in own_struct:
            d = np.abs(rr - r) + np.abs(cc - c)
            sur += np.maximum(0, 14 - 2 * d)
        planes[14, :H, :W] = np.clip(sur * 4, 0, 255).astype(np.uint8)
        can_build = (O == 1) & (T == 1) & (A >= 35 + sur)
        planes[15, :H, :W] = can_build * 255
        my_a, op_a = max(1, int(obs["my_army"])), max(1, int(obs["opp_army"]))
        my_l, op_l = max(1, int(obs["my_land"])), max(1, int(obs["opp_land"]))
        scal = np.array([t / 1200.0, (t % 50) / 50.0, float(t >= 800), math.log(my_a / op_a),
                         math.log(my_l / op_l), math.log(my_a) / 8.0, math.log(op_a) / 8.0,
                         float(self.egen is not None)], np.float32)
        return planes, scal, (O, A, T)


def action_index(a, W):
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


def legal_mask(O, A, T, H, W, planes):
    """bool[NACT]: moves from own cells with >=2 army into passable in-board
    cells, builds where affordable, pass always."""
    m = np.zeros((S, S, 9), bool)
    own2 = (O == 1) & (A >= 2)
    passable = ~((T == 2) | (T == 5))
    pas = np.zeros((S + 2, S + 2), bool)
    pas[1:H + 1, 1:W + 1] = passable
    for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        dest = pas[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
        ok = own2 & dest
        m[:H, :W, d * 2] = ok
        m[:H, :W, d * 2 + 1] = ok & (A >= 3)
    m[:H, :W, 8] = planes[15, :H, :W] > 0
    flat = np.zeros(NACT, bool)
    flat[:PASS_IDX] = m.reshape(-1)
    flat[PASS_IDX] = True
    return flat
