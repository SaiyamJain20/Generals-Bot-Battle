"""Write a copy of a bot file with PARAMS overrides: make_variant.py src.py dst.py '{"k": v}'"""
import json
import sys

src, dst, spec = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
code = open(src).read()
marker = "\nDIRS = "
k = code.index(marker)
code = code[:k] + "\nPARAMS.update(" + repr(spec) + ")\n" + code[k:]
open(dst, "w").write(code)
