"""Scripted tactical positions: the bot must find forced wins and forced defenses."""
import importlib.util
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from sim import engine as E  # noqa: E402

BOT = os.path.join(ROOT, "bots", "participant.py")
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def fresh():
    spec = importlib.util.spec_from_file_location("pt_%d" % random.randrange(10 ** 9), BOT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.DEBUG = True
    return m


def board(H=18, W=18, g0=(5, 5), g1=(14, 14), time=800):
    grid = [[0] * W for _ in range(H)]
    grid[g0[0]][g0[1]] = 1
    grid[g1[0]][g1[1]] = 2
    s = E.from_grid(grid)
    s.time = time
    return s


def give(s, p, rc, a):
    i = rc[0] * s.W + rc[1]
    s.owner[i] = p
    s.army[i] = a


def act_for(s, p=0):
    m = fresh()
    return m.act(E.observe(s, p))


def test_deathtouch_win():
    s = board()
    give(s, 0, (14, 13), 3)
    s.army[14 * 18 + 14] = 200
    a = act_for(s)
    assert a == [0, 14, 13, RIGHT, 0] or a == [0, 14, 13, RIGHT, 1]


def test_capture_win_before_800():
    s = board(time=300)
    give(s, 0, (14, 13), 30)
    s.army[14 * 18 + 14] = 12
    a = act_for(s)
    assert a[:4] == [0, 14, 13, RIGHT]


def test_strip_adjacent_toucher():
    s = board(time=800)
    s.army[5 * 18 + 5] = 30
    give(s, 1, (5, 6), 6)      # enemy next to our general
    give(s, 0, (4, 6), 9)      # our third tile next to the enemy source
    a = act_for(s)
    assert a[:4] == [0, 4, 6, DOWN], a
    # verify it actually saves us against the touch
    E.step(s, [a, [0, 5, 6, LEFT, 0]])
    assert s.winner != 1


def test_reinforce_against_capture():
    s = board(time=300)
    s.army[5 * 18 + 5] = 10
    give(s, 1, (5, 6), 25)     # enemy can capture our general (24 > 10)
    give(s, 0, (5, 4), 30)     # we can reinforce from the left
    a = act_for(s)
    E.step(s, [a, [0, 5, 6, LEFT, 0]])
    assert s.winner != 1, a


def test_no_suicidal_general_move():
    s = board(time=300)
    s.army[5 * 18 + 5] = 40
    give(s, 1, (5, 7), 30)     # enemy two cells away
    give(s, 0, (5, 6), 1)
    a = act_for(s)
    # whatever we do, the enemy must not be able to take the general next turn
    for ea in ([0, 5, 7, LEFT, 0], [1, 0, 0, 0, 0]):
        t = s.copy()
        E.step(t, [a, ea])
        assert t.winner != 1
