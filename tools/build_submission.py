"""Build the single-file submission from bots/participant.py + a tuned params JSON.

    python tools/build_submission.py --params runs/final_params.json --id <participant_id> \
        --name "<bot name>" [--out submission/<participant_id>.py]

Writes the tuned values straight into the PARAMS literal (no runtime update),
forces DEBUG = False, fills the header placeholders, then runs the static
checks from tools/check_submission.py.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "bots", "participant.py"))
    ap.add_argument("--params", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    src = open(args.src).read()
    tuned = json.load(open(args.params))
    # locate the PARAMS dict literal
    m = re.search(r"^PARAMS = \{", src, re.M)
    start = m.start()
    depth, i = 0, src.index("{", start)
    while True:
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    lit = src[src.index("{", start):i + 1]
    params = ast.literal_eval(lit)
    unknown = [k for k in tuned if k not in params]
    if unknown:
        print("WARNING: tuned keys not in PARAMS (ignored):", unknown)
    for k, v in tuned.items():
        if k in params:
            params[k] = v
    body = "PARAMS = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in params.items()) + "}"
    out_src = src[:start] + body + src[i + 1:]
    out_src = re.sub(r"^DEBUG = True", "DEBUG = False", out_src, flags=re.M)
    out_src = out_src.replace("[PARTICIPANT_ID]", args.id).replace("[BOT_NAME]", args.name)
    out = args.out or os.path.join(ROOT, "submission", f"{args.id}.py")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(out_src)
    print("wrote", out, len(out_src.encode()), "bytes, sha256", hashlib.sha256(out_src.encode()).hexdigest())
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_submission.py"), out, "--games", "4"])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
