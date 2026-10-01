"""Train the behaviour-cloning policy (+ win-probability head) on data/bc/<name>/.

    python learn/bc_train.py --name resbot --epochs 4 --out data/bc/resbot_net.pt
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

S = 21
NACT = S * S * 9 + 1


class Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = Fn.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return Fn.relu(x + y)


class Net(nn.Module):
    def __init__(self, cin=16, nscal=8, ch=64, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin + nscal, ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(ch) for _ in range(blocks)])
        self.pol = nn.Conv2d(ch, 9, 1)
        self.pas = nn.Linear(ch + nscal, 1)
        self.val = nn.Sequential(nn.Linear(ch + nscal, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, planes, scal):
        x = planes.float() / 255.0
        b = scal[:, :, None, None].expand(-1, -1, S, S)
        h = self.blocks(self.stem(torch.cat([x, b], 1)))
        pol = self.pol(h).permute(0, 2, 3, 1).reshape(h.shape[0], -1)
        g = torch.cat([h.mean((2, 3)), scal], 1)
        logits = torch.cat([pol, self.pas(g)], 1)
        return logits, self.val(g).squeeze(1)


def load(name):
    d = os.path.join("data", "bc", name)
    shards = sorted(glob.glob(os.path.join(d, "act_*.npy")))
    P, Sc, M, A, Wn = [], [], [], [], []
    for f in shards:
        k = f[-7:-4]
        P.append(np.load(os.path.join(d, f"planes_{k}.npy"), mmap_mode="r"))
        Sc.append(np.load(os.path.join(d, f"scal_{k}.npy")))
        M.append(np.load(os.path.join(d, f"mask_{k}.npy"), mmap_mode="r"))
        A.append(np.load(os.path.join(d, f"act_{k}.npy")))
        Wn.append(np.load(os.path.join(d, f"win_{k}.npy")))
    return P, Sc, M, A, Wn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="resbot")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P, Sc, M, A, Wn = load(args.name)
    n_sh = len(A)
    val_sh = {n_sh - 1} if n_sh > 1 else set()
    net = Net(ch=args.ch, blocks=args.blocks).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_total = args.epochs * sum(len(A[i]) for i in range(n_sh) if i not in val_sh) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, steps_total))
    print(f"device {dev} shards {n_sh} samples {sum(len(a) for a in A)} params "
          f"{sum(p.numel() for p in net.parameters())}", flush=True)

    def batches(sh_ids, shuffle=True):
        order = list(sh_ids)
        if shuffle:
            np.random.shuffle(order)
        for si in order:
            planes = np.asarray(P[si])
            masks = np.unpackbits(np.asarray(M[si]), axis=1)[:, :NACT].astype(bool)
            idx = np.random.permutation(len(A[si])) if shuffle else np.arange(len(A[si]))
            for k in range(0, len(idx), args.bs):
                j = idx[k:k + args.bs]
                yield (torch.from_numpy(planes[j]).to(dev), torch.from_numpy(Sc[si][j]).to(dev),
                       torch.from_numpy(masks[j]).to(dev), torch.from_numpy(A[si][j].astype(np.int64)).to(dev),
                       torch.from_numpy(Wn[si][j].astype(np.float32)).to(dev))

    def evaluate():
        net.eval()
        tot = cor = 0
        vl = 0.0
        with torch.no_grad():
            for planes, scal, mask, act, win in batches(val_sh, shuffle=False):
                lg, v = net(planes, scal)
                lg = lg.masked_fill(~mask, -1e9)
                cor += (lg.argmax(1) == act).sum().item()
                tot += len(act)
                vl += Fn.binary_cross_entropy_with_logits(v, win, reduction="sum").item()
        net.train()
        return cor / max(1, tot), vl / max(1, tot)

    train_sh = [i for i in range(n_sh) if i not in val_sh]
    step = 0
    for ep in range(args.epochs):
        t0 = time.time()
        for planes, scal, mask, act, win in batches(train_sh):
            lg, v = net(planes, scal)
            lg = lg.masked_fill(~mask, -1e9)
            loss = Fn.cross_entropy(lg, act) + 0.3 * Fn.binary_cross_entropy_with_logits(v, win)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step < steps_total - 1:
                sched.step()
            step += 1
            if step % 500 == 0:
                print(f"ep {ep} step {step} loss {loss.item():.3f}", flush=True)
        acc, vl = evaluate()
        print(f"epoch {ep} val top1 {acc:.3f} value bce {vl:.3f} time {time.time() - t0:.0f}s", flush=True)
        out = args.out or os.path.join("data", "bc", f"{args.name}_net.pt")
        torch.save({"state": net.state_dict(), "ch": args.ch, "blocks": args.blocks}, out)


if __name__ == "__main__":
    main()
