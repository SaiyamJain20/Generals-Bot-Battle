"""Generate an RO bot from a heuristic bot file (+ RO hook).

    python rl/hier/gen_bot_ro.py                                   # tune_base12g -> rl/hier/bot_ro.py
    python rl/hier/gen_bot_ro.py bots/versions/F2.py rl/hier/bot_ro_F2.py
"""
import os
import sys

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = sys.argv[1] if len(sys.argv) > 1 else "bots/versions/tune_base12g.py"
out_path = sys.argv[2] if len(sys.argv) > 2 else "rl/hier/bot_ro.py"
src = open(os.path.join(R, src_path)).read()
blk = open(os.path.join(R, "rl/hier/ro_block.txt")).read()
a = "class Bot:\n    def __init__"
assert src.count(a) == 1
src = src.replace(a, blk + a)
old = """        options.sort(key=lambda x: -x[0])
        choice = options[0][1]
        self.last_label = options[0][2]
"""
new = """        options.sort(key=lambda x: -x[0])
        pick = 0
        if RO_W is not None and len(options) > 1:
            pick = _ro_pick(self, options, need_g)
        choice = options[pick][1]
        self.last_label = options[pick][2]
"""
assert src.count(old) == 1
src = src.replace(old, new)
open(os.path.join(R, out_path), "w").write(src)
print("wrote", out_path, "from", src_path)
