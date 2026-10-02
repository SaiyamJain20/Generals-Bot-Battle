"""Embed RO weights into a copy of bot_ro.py -> single submission-ready file.

    .venv312/bin/python rl/hier/build_ro_submission.py rl/hier/runs/<name>/weights_<it>.json out.py [--games 2]

Runs static checks (stdlib-only imports, no open()/print(), size <= 1 MiB, act defined) and then
tools/check_submission.py (which also plays a few games).
"""
import argparse
import ast
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MARK = "RO_W = None          # weights dict (see rl/hier/ro_ppo.py::default_weights) or None"


def rnd(x):
    if isinstance(x, list):
        return [rnd(v) for v in x]
    return round(float(x), 6)


def build(weights_path, out_path):
    w = json.load(open(weights_path))
    w = {k: (rnd(v) if k not in ("alpha0",) else float(v)) for k, v in w.items()}
    w["alpha"] = float(w["alpha"])
    src = open(os.path.join(HERE, "bot_ro.py")).read()
    assert src.count(MARK) == 1, "RO_W marker not found in bot_ro.py"
    lit = "RO_W = " + json.dumps(w, separators=(",", ":")) + "  # trained RO-PPO weights"
    out = src.replace(MARK, lit)
    open(out_path, "w").write(out)
    return out


def static_checks(src):
    ok = True
    raw = src.encode("utf-8")
    if len(raw) > 1 << 20:
        print("FAIL size", len(raw))
        ok = False
    tree = ast.parse(src)
    std = set(sys.stdlib_module_names)
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module.split(".")[0]]
        for m in mods:
            if m not in std:
                print("FAIL non-stdlib import", m)
                ok = False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id in ("open", "print", "input", "exec", "eval", "__import__", "compile"):
            print("FAIL banned call", node.func.id, "line", node.lineno)
            ok = False
    if not any(isinstance(n, ast.FunctionDef) and n.name == "act" for n in tree.body):
        print("FAIL no act()")
        ok = False
    print("static checks %s; size %d bytes" % ("ok" if ok else "FAILED", len(raw)))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--no-dynamic", action="store_true")
    a = ap.parse_args()
    src = build(a.weights, a.out)
    ok = static_checks(src)
    if ok and not a.no_dynamic:
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "vendor/generals-bots") + ":" + ROOT)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools/check_submission.py"), a.out,
                            "--games", str(a.games)], cwd=ROOT, env=env)
        ok = r.returncode == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
