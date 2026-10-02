"""Assemble the single-file stdlib-only bot rl/bots/pystudent.py from pyfeat.py + pyinfer.py + packed weights.
   python rl/build_pystudent.py [ckpt=rl/students/ResBot_12x1.pt] [out=rl/bots/pystudent.py] [mode=e] [SA KS KH]"""
import ast
import base64
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "rl"))
import torch  # noqa: E402
import pack_student as PK  # noqa: E402

HEADER = '''"""Code Bot submission: behaviour-cloned tiny residual CNN (12 channels x 1 block, 6.8k params), pure stdlib.

participant: filled in by tools/build_submission.py   bot: pystudent
AI assistance: written with Claude Code (Anthropic). Network trained by behaviour cloning from our own
heuristic bots' games; weights are embedded below (BN folded, float16, base64).
Sources: generals.bot environment rules; no third-party code. The forward pass runs on big-integer SIMD lanes
(each feature map is one Python int), features are a stdlib port of our numpy feature extractor.
"""
'''

GLUE = '''

_NET = None
_TRK = None
_FALLBACK = [1, 0, 0, 0, 0]


def _setup():
    global _NET
    _NET = Net(base64.b64decode(_BLOB), %d, %d, %d)
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
'''


def body(path):
    tree = ast.parse(open(path).read())
    keep = []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant) and isinstance(n.value.value, str):
            continue
        keep.append(n)
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def main():
    a = sys.argv[1:]
    ck = a[0] if len(a) > 0 else os.path.join(ROOT, "rl/students/ResBot_12x1.pt")
    out = a[1] if len(a) > 1 else os.path.join(ROOT, "rl/bots/pystudent.py")
    mode = a[2] if len(a) > 2 else "e"
    SA, KS, KH = (int(x) for x in a[3:6]) if len(a) >= 6 else (8, 14, 10)
    c = torch.load(ck, map_location="cpu")
    blob = base64.b64encode(PK.pack(c, mode)).decode()
    src = HEADER + "import base64\nimport gc\nimport math\nimport struct\nimport sys\nfrom array import array\nfrom operator import mul\n\n"
    src += body(os.path.join(ROOT, "rl/pyfeat.py")) + "\n\n" + body(os.path.join(ROOT, "rl/pyinfer.py")) + "\n\n"
    src += "_BLOB = (\n"
    for i in range(0, len(blob), 100):
        src += f"    '{blob[i:i+100]}'\n"
    src += ")\n" + GLUE % (SA, KS, KH)
    with open(out, "w") as f:
        f.write(src)
    print(out, len(src.encode()), "bytes")


if __name__ == "__main__":
    main()
