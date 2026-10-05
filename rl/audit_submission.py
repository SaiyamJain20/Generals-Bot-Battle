"""Audit a candidate submission against the organiser rules: one .py, <= 1 MiB, Python 3.12 standard library
only, no separate assets/model files, no file/network/exec calls.

    python rl/audit_submission.py <file.py> [...]
"""
import ast
import sys

BANNED_CALLS = {"open", "exec", "eval", "__import__", "compile", "input"}
RISKY_MODULES = {"socket", "subprocess", "urllib", "http", "ctypes", "multiprocessing", "threading", "os",
                 "pathlib", "shutil", "importlib", "pickle", "marshal"}


def audit(path):
    raw = open(path, "rb").read()
    src = raw.decode("utf-8")
    tree = ast.parse(src)
    mods, calls = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in BANNED_CALLS:
            calls.add(n.func.id)
    nonstd = sorted(m for m in mods if m not in sys.stdlib_module_names)
    risky = sorted(m for m in mods if m in RISKY_MODULES)
    has_act = any(isinstance(n, ast.FunctionDef) and n.name == "act" for n in tree.body)
    ok = len(raw) <= 1048576 and not nonstd and not risky and not calls and has_act
    print(f"{path}: {len(raw)} bytes; imports {sorted(mods)}; non-stdlib {nonstd or 'none'}; "
          f"risky {risky or 'none'}; banned calls {sorted(calls) or 'none'}; top-level act {has_act}; "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if all([audit(p) for p in sys.argv[1:]]) else 1)
