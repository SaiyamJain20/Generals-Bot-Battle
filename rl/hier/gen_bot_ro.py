"""Generate rl/hier/bot_ro.py from bots/versions/tune_base12g.py (+ RO hook)."""
import os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src = open(os.path.join(R, "bots/versions/tune_base12g.py")).read()
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
open(os.path.join(R, "rl/hier/bot_ro.py"), "w").write(src)
