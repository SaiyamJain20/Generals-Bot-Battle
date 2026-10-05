"""Pack a learn/bc_train.py Net checkpoint into a compact blob (BN folded into the convs).

    python rl/pack_student.py rl/students/ResBot_12x1.pt [--mode e|f|q] [--out file.b64]

Modes: e = float16 (struct 'e'), f = float32, q = int8 with per-output-row fp16 scale.
Blob = bytes: magic 'S', mode byte, ch, blocks, then the arrays in the order given by `spec(ch, blocks)`.
"""
import argparse
import base64
import struct
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch


def spec(ch, blocks):
    """(name, nrows, rowlen) of every array, in storage order."""
    s = [("stem_w", ch, 24 * 9), ("stem_b", 1, ch)]
    for i in range(blocks):
        s += [(f"b{i}w1", ch, ch * 9), (f"b{i}b1", 1, ch), (f"b{i}w2", ch, ch * 9), (f"b{i}b2", 1, ch)]
    s += [("polw", 9, ch), ("polb", 1, 9), ("pasw", 1, ch + 8), ("pasb", 1, 1)]
    return s


def fold(st, key, bnkey, eps=1e-5):
    w = st[key + ".weight"].double().numpy()
    g = st[bnkey + ".weight"].double().numpy()
    b = st[bnkey + ".bias"].double().numpy()
    m = st[bnkey + ".running_mean"].double().numpy()
    v = st[bnkey + ".running_var"].double().numpy()
    sc = g / np.sqrt(v + eps)
    return w * sc[:, None, None, None], b - m * sc


def arrays(ck):
    st = ck["state"]
    ch, blocks = ck["ch"], ck["blocks"]
    out = {}
    w, b = fold(st, "stem.0", "stem.1")
    out["stem_w"], out["stem_b"] = w.reshape(ch, -1), b
    for i in range(blocks):
        for j in (1, 2):
            w, b = fold(st, f"blocks.{i}.c{j}", f"blocks.{i}.b{j}")
            out[f"b{i}w{j}"], out[f"b{i}b{j}"] = w.reshape(ch, -1), b
    out["polw"] = st["pol.weight"].double().numpy().reshape(9, ch)
    out["polb"] = st["pol.bias"].double().numpy()
    out["pasw"] = st["pas.weight"].double().numpy().reshape(1, -1)
    out["pasb"] = st["pas.bias"].double().numpy()
    return out


def pack(ck, mode="e"):
    ch, blocks = ck["ch"], ck["blocks"]
    arr = arrays(ck)
    buf = bytearray(b"S" + mode.encode() + bytes([ch, blocks]))
    for name, nr, rl in spec(ch, blocks):
        a = arr[name].reshape(nr, rl)
        if mode == "q":
            for row in a:
                sc = float(np.abs(row).max()) / 127.0 or 1.0
                sc16 = float(np.float16(sc))
                if sc16 * 127 < np.abs(row).max():
                    sc16 = float(np.float16(sc * 1.001))
                buf += struct.pack("<e", sc16)
                buf += struct.pack(f"<{rl}b", *[int(round(x / sc16)) for x in row])
        else:
            buf += struct.pack(f"<{nr * rl}{mode}", *[float(x) for x in a.reshape(-1)])
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--mode", default="e")
    ap.add_argument("--out")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    blob = pack(ck, a.mode)
    b64 = base64.b64encode(blob).decode()
    print(f"{a.ckpt}: ch {ck['ch']} blocks {ck['blocks']} mode {a.mode}: {len(blob)} bytes, base64 {len(b64)}")
    if a.out:
        with open(a.out, "w") as f:
            f.write(b64)


if __name__ == "__main__":
    main()
