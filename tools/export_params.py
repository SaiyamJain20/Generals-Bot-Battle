"""Export the effective PARAMS of a bot file (after any PARAMS.update) to JSON.

    python tools/export_params.py bots/versions/F0.py runs/final_params.json

Feed the JSON to tools/build_submission.py, which bakes it into the PARAMS literal.
"""
import importlib.util
import json
import sys


def main():
    src, out = sys.argv[1], sys.argv[2]
    spec = importlib.util.spec_from_file_location("export_params_bot", src)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    params = {k: (float(v) if isinstance(v, float) else v) for k, v in m.PARAMS.items()}
    with open(out, "w") as f:
        json.dump(params, f, indent=1, sort_keys=True)
    print(f"wrote {out}: {len(params)} params from {src}")


if __name__ == "__main__":
    main()
