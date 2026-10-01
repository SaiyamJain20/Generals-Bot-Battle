"""Port of competition/agents/expander_python/agent.py (generals-bots@13db8f69, MIT)."""
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def act(obs):
    H, W = obs["height"], obs["width"]
    own, typ, arm = obs["owner"], obs["type"], obs["army"]
    best_score, best_move, first_valid = -1.0, None, None
    for r in range(H):
        for c in range(W):
            if own[r][c] != 1:
                continue
            src = arm[r][c]
            if src <= 1:
                continue
            for d, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                t = typ[nr][nc]
                if t == 2 or t == 5:
                    continue
                move = [0, r, c, d, 0]
                if first_valid is None:
                    first_valid = move
                if src <= arm[nr][nc] + 1:
                    continue
                is_opp = own[nr][nc] == 2
                is_vis_neutral = own[nr][nc] == 0 and t not in (0, 5)
                score = float(src)
                if is_opp or is_vis_neutral:
                    score *= 10.0
                if is_opp:
                    score *= 2.0
                if score > best_score:
                    best_score, best_move = score, move
    if best_move is not None:
        return best_move
    if first_valid is not None:
        return first_valid
    return [1, 0, 0, 0, 0]
