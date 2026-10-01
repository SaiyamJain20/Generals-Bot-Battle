"""Static + dynamic checks for the Code Bot submission file.

    python tools/check_submission.py bots/participant.py [--games 4]

Checks: UTF-8, size <= 1 MiB, compiles on this interpreter (run with 3.12),
imports only from the standard library, defines act(observation), header
docstring mentions attribution + AI assistance, no print() calls / file or
network I/O, and plays a few full games returning valid actions.
"""
import argparse
import ast
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

STDLIB = set(sys.stdlib_module_names)
BANNED_CALLS = {"print", "open", "input", "exec", "eval", "__import__", "compile"}
BANNED_MODULES = {"socket", "subprocess", "urllib", "http", "requests", "multiprocessing", "threading",
                  "ctypes", "os", "shutil", "pathlib", "signal"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--games", type=int, default=4)
    args = ap.parse_args()
    ok = True
    raw = open(args.path, "rb").read()
    print(f"size {len(raw)} bytes (limit 1048576)")
    if len(raw) > 1048576:
        print("FAIL size")
        ok = False
    try:
        src = raw.decode("utf-8")
    except UnicodeDecodeError:
        print("FAIL not utf-8")
        return 1
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    for kw in ("participant", "AI assistance", "Sources"):
        if kw.lower() not in doc.lower():
            print(f"WARN header docstring lacks '{kw}'")
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            print(f"FAIL banned call {node.func.id}() at line {node.lineno}")
            ok = False
    nonstd = [m for m in mods if m not in STDLIB]
    if nonstd:
        print("FAIL non-stdlib imports:", nonstd)
        ok = False
    bad = [m for m in mods if m in BANNED_MODULES]
    if bad:
        print("FAIL risky modules:", bad)
        ok = False
    print("imports:", sorted(mods))
    if not any(isinstance(n, ast.FunctionDef) and n.name == "act" for n in tree.body):
        print("FAIL no top-level act()")
        ok = False
    if "TODO" in src or "[PARTICIPANT_ID]" in src:
        print("WARN placeholders/TODO present (fill participant id + bot name before submitting)")
    # dynamic: import time + a few games vs expander
    t0 = time.perf_counter()
    spec = importlib.util.spec_from_file_location("sub_check", args.path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print(f"import {1000 * (time.perf_counter() - t0):.1f} ms")
    if getattr(m, "DEBUG", False):
        print("FAIL DEBUG is True")
        ok = False
    import arena.run as R
    res = R.run_match(args.path, os.path.join(ROOT, "bots", "opp", "expander.py"), args.games,
                      workers=min(4, args.games), limit_ms=150, quiet=False)
    summ = R.summarize(res)
    print(json.dumps(summ))
    if summ.get("forfeits"):
        ok = False
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
