"""Scripted corner cases (ported from upstream tests/) checked differentially:
sim/engine.py must give the same winner / done / armies / owners as the
pinned JAX transition (build_castles + deathtouch)."""
import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "generals-bots"))

from generals.core.game import create_initial_state  # noqa: E402
from generals.modifiers import build_castles as _bc  # noqa: E402
from generals.modifiers import deathtouch as _dt  # noqa: E402

from sim import engine as E  # noqa: E402

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
PASS = [1, 0, 0, 0, 0]


def mv(i, j, d, split=0):
    return [0, i, j, d, split]


def scenario(size=6, time=800, cells=(), gen_army=(1, 1), castles=(), mountains=()):
    """P0 general (0,0), P1 general (0,size-1). cells: (player, (i,j), army)."""
    grid = [[0] * size for _ in range(size)]
    grid[0][0], grid[0][size - 1] = 1, 2
    for (i, j) in mountains:
        grid[i][j] = -2
    js = create_initial_state(jnp.asarray(grid, dtype=jnp.int32))
    s = E.from_grid(grid)
    js = js._replace(time=jnp.int32(time))
    s.time = time
    arm = np.asarray(js.armies).copy()
    own = np.asarray(js.ownership).copy()
    neu = np.asarray(js.ownership_neutral).copy()
    cas = np.asarray(js.castles).copy()
    arm[0, 0], arm[0, size - 1] = gen_army
    s.army[0], s.army[size - 1] = gen_army
    for p, (i, j), a in cells:
        arm[i, j] = a
        own[:, i, j] = False
        own[p, i, j] = True
        neu[i, j] = False
        k = i * size + j
        s.army[k], s.owner[k] = a, p
    for (i, j) in castles:
        cas[i, j] = True
        s.castle[i * size + j] = True
    js = js._replace(armies=jnp.asarray(arm), ownership=jnp.asarray(own),
                     ownership_neutral=jnp.asarray(neu), castles=jnp.asarray(cas))
    return js, s


def run_both(js, s, acts):
    st, a2 = _bc.apply_build_actions(js, jnp.asarray(acts, dtype=jnp.int32))
    ns, info = _dt.step(st, a2, 800)
    E.step(s, acts)
    jdone, jwin = bool(info.is_done), int(info.winner)
    assert (jdone, jwin) == (s.done, s.winner), f"jax={(jdone, jwin)} sim={(s.done, s.winner)}"
    if not jdone:
        assert np.asarray(ns.armies).reshape(-1).tolist() == s.army
        own = np.asarray(ns.ownership)
        owner = [0 if own[0].reshape(-1)[k] else 1 if own[1].reshape(-1)[k] else -1
                 for k in range(len(s.army))]
        assert owner == s.owner
    return jdone, jwin


CASES = {
    "touch_beats_stacked_general": (dict(time=800, cells=[(0, (0, 4), 5)], gen_army=(1, 50)),
                                    [mv(0, 4, RIGHT), PASS], (True, 0)),
    "no_touch_at_799": (dict(time=799, cells=[(0, (0, 4), 5)], gen_army=(1, 50)),
                        [mv(0, 4, RIGHT), PASS], (False, -1)),
    "capture_after_threshold": (dict(time=900, cells=[(0, (0, 4), 100)], gen_army=(1, 3)),
                                [mv(0, 4, RIGHT), PASS], (True, 0)),
    "chase_kills_source": (dict(cells=[(0, (0, 4), 5), (1, (1, 4), 10)]),
                           [mv(0, 4, RIGHT), mv(1, 4, UP)], (False, -1)),
    "chase_leaves_two": (dict(cells=[(0, (0, 4), 5), (1, (1, 4), 3)]),
                         [mv(0, 4, RIGHT), mv(1, 4, UP)], (True, 0)),
    "chase_leaves_one": (dict(cells=[(0, (0, 4), 5), (1, (1, 4), 5)]),
                         [mv(0, 4, RIGHT), mv(1, 4, UP)], (False, -1)),
    "mutual_touch_draw": (dict(cells=[(0, (0, 4), 5), (1, (0, 1), 5)]),
                          [mv(0, 4, RIGHT), mv(0, 1, LEFT)], (True, -1)),
    "general_counter_bigger_fails": (dict(cells=[(0, (0, 4), 5)], gen_army=(1, 40)),
                                     [mv(0, 4, RIGHT), mv(0, 5, LEFT)], (True, 0)),
    "general_counter_equal_p1": (dict(cells=[(0, (0, 4), 6)], gen_army=(1, 6)),
                                 [mv(0, 4, RIGHT), mv(0, 5, LEFT)], None),
    "general_counter_equal_p0": (dict(cells=[(1, (0, 1), 6)], gen_army=(6, 1)),
                                 [mv(0, 0, RIGHT), mv(0, 1, LEFT)], None),
    "mutual_capture_early_draw": (dict(time=100, cells=[(0, (0, 4), 9), (1, (0, 1), 9)], gen_army=(2, 2)),
                                  [mv(0, 4, RIGHT), mv(0, 1, LEFT)], (True, -1)),
    "build_next_to_general_not_touch": (dict(time=900, cells=[(0, (0, 4), 60)]),
                                        [[2, 0, 4, 0, 0], PASS], (False, -1)),
    "race_for_neutral_bigger_last": (dict(time=10, cells=[(0, (2, 1), 9), (1, (2, 3), 5)]),
                                     [mv(2, 1, RIGHT), mv(2, 3, LEFT)], (False, -1)),
    "dodge_fails_chaser_first": (dict(time=10, cells=[(0, (2, 2), 9), (1, (2, 3), 5)]),
                                 [mv(2, 2, RIGHT), mv(2, 3, RIGHT)], (False, -1)),
    "reinforce_before_attack": (dict(time=10, cells=[(0, (2, 2), 20), (1, (2, 3), 5), (1, (2, 4), 30)]),
                                [mv(2, 2, RIGHT), mv(2, 4, LEFT)], (False, -1)),
    "castle_zero_snipe": (dict(time=11, cells=[(0, (3, 3), 35), (1, (3, 4), 2)]),
                          [[2, 3, 3, 0, 0], mv(3, 4, LEFT)], (False, -1)),
    "split_half": (dict(time=10, cells=[(0, (2, 2), 9)]), [mv(2, 2, DOWN, 1), PASS], (False, -1)),
    "growth_odd_turn": (dict(time=49, cells=[(0, (2, 2), 3), (1, (3, 3), 3)], castles=[(2, 2)]),
                        [PASS, PASS], (False, -1)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_case(name):
    kw, acts, expect = CASES[name]
    js, s = scenario(**kw)
    got = run_both(js, s, acts)
    if expect is not None:
        assert got == expect


if __name__ == "__main__":
    for n in sorted(CASES):
        test_case(n)
        print("ok", n)
