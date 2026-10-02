"""Generic adapter: run an external bot that speaks the generals.bot stdio wire
protocol as a subprocess, exposed through our act(observation).

Wire protocol (competition/protocol.py):
  handshake (once):    "<player_id> <H> <W>"
  per turn:            "<turn> <my_land> <my_army> <opp_land> <opp_army>" + 3 x H lines of W ints
                       (type grid, owner grid, army grid; same codes as our observation dict)
  reply:               "<kind> <row> <col> <dir> <split>"

Usage from a per-bot wrapper (load this file by path, never by module name):
    act = ext_stdio.make_act(["./agent"], cwd="/abs/dir", env={...})
or directly with PARAMS = {"cmd": [...], "cwd": "...", "env": {...}}.

One subprocess per game: the arena re-imports a wrapper module for every game,
so each module instance owns its own process; a new process is started at the
first act() call of a game (or whenever the turn counter goes backwards).
Processes of finished games are killed when a new game starts (process-wide
registry: any registered process that has advanced past turn 0 belongs to an
older game) and at interpreter exit. Child stderr goes to /dev/null unless
EXT_STDIO_STDERR=<file> is set. A dead child or a reply timeout raises, so the
arena records a visible forfeit instead of silently passing.
"""
import atexit
import os
import select
import subprocess
import sys
import time

PARAMS = {"cmd": None, "cwd": None, "env": None, "first_timeout": 60.0, "timeout": 10.0}

_REG = sys.__dict__.setdefault("_porter_stdio_registry", [])  # list of [proc, last_turn]


def _kill(entry):
    p = entry[0]
    if p.poll() is None:
        try:
            p.stdin.close()
        except Exception:
            pass
        try:
            p.wait(timeout=0.2)
        except Exception:
            p.kill()
            try:
                p.wait(timeout=1.0)
            except Exception:
                pass


def _kill_all():
    for e in list(_REG):
        _kill(e)
    del _REG[:]


if not sys.__dict__.get("_porter_stdio_atexit"):
    atexit.register(_kill_all)
    sys._porter_stdio_atexit = True


def _reap_old_games():
    keep = []
    for e in _REG:
        if e[1] > 0 or e[0].poll() is not None:
            _kill(e)
        else:
            keep.append(e)
    _REG[:] = keep


class StdioBot:
    def __init__(self, cmd, cwd=None, env=None, first_timeout=60.0, timeout=10.0):
        self.cmd, self.cwd = list(cmd), cwd
        self.env = dict(os.environ, **(env or {}))
        self.first_timeout, self.timeout = first_timeout, timeout
        self.entry = None
        self.buf = b""
        self.last_turn = -1

    def _start(self, obs):
        if self.entry is not None:
            _kill(self.entry)
            if self.entry in _REG:
                _REG.remove(self.entry)
        _reap_old_games()
        err = os.environ.get("EXT_STDIO_STDERR")
        errf = open(err, "ab") if err else subprocess.DEVNULL
        p = subprocess.Popen(self.cmd, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=errf, bufsize=0)
        if err:
            errf.close()
        self.entry = [p, 0]
        _REG.append(self.entry)
        self.buf = b""
        p.stdin.write(f"{obs['player_id']} {obs['height']} {obs['width']}\n".encode())

    def _readline(self, timeout):
        p = self.entry[0]
        fd = p.stdout.fileno()
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"ext_stdio: no reply within {timeout}s from {self.cmd}")
            r, _, _ = select.select([fd], [], [], left)
            if r:
                chunk = os.read(fd, 65536)
                if not chunk:
                    raise RuntimeError(f"ext_stdio: child exited (rc={p.poll()}) {self.cmd}")
                self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line

    def act(self, obs):
        t = obs["turn"]
        first = self.entry is None or t <= self.last_turn or self.entry[0].poll() is not None
        if first:
            self._start(obs)
        self.last_turn = t
        self.entry[1] = t
        parts = [f"{t} {obs['my_land']} {obs['my_army']} {obs['opp_land']} {obs['opp_army']}"]
        for key in ("type", "owner", "army"):
            for row in obs[key]:
                parts.append(" ".join(map(str, row)))
        parts.append("")
        self.entry[0].stdin.write("\n".join(parts).encode())
        while True:
            line = self._readline(self.first_timeout if first else self.timeout).strip()
            if line:
                break
        vals = [int(v) for v in line.split()]
        if len(vals) != 5:
            raise ValueError(f"ext_stdio: malformed reply {line!r}")
        return vals


def make_act(cmd, cwd=None, env=None, first_timeout=60.0, timeout=10.0):
    bot = StdioBot(cmd, cwd, env, first_timeout, timeout)
    return bot.act


_default = []


def act(obs):
    if not _default:
        _default.append(StdioBot(PARAMS["cmd"], PARAMS["cwd"], PARAMS["env"],
                                 PARAMS["first_timeout"], PARAMS["timeout"]))
    return _default[0].act(obs)
