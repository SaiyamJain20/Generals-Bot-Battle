"""Uniform random valid-looking move (mirrors generals/agents/random_agent)."""
import random

_rng = random.Random(12345)


def act(obs):
    H, W = obs["height"], obs["width"]
    own, typ, arm = obs["owner"], obs["type"], obs["army"]
    cands = []
    for r in range(H):
        for c in range(W):
            if own[r][c] == 1 and arm[r][c] > 1:
                for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and typ[rr][cc] not in (2, 5):
                        cands.append((r, c, d))
    if not cands or _rng.random() < 0.05:
        return [1, 0, 0, 0, 0]
    r, c, d = _rng.choice(cands)
    return [0, r, c, d, 1 if _rng.random() < 0.25 else 0]
