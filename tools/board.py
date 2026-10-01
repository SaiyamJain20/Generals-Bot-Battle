"""ASCII dump of the true board: our cells lowercase/digits, enemy uppercase."""
def dump(s, me, vis=None, mark=()):
    W = s.W
    out = []
    for r in range(s.H):
        row = []
        for c in range(W):
            i = r * W + c
            if s.mountain[i]:
                ch = " ##"
            else:
                o, a = s.owner[i], s.army[i]
                tag = "."
                if o == me:
                    tag = "G" if s.general[i] else ("C" if s.castle[i] else "m")
                elif o == 1 - me:
                    tag = "g" if s.general[i] else ("c" if s.castle[i] else "e")
                ch = f"{tag}{min(a, 99):2d}" if o >= 0 else ("  ." if not s.castle[i] else "  c")
                if vis is not None and not vis[i] and o != me:
                    ch = ch.lower().replace(".", ",") if o < 0 else ch + ""
                    ch = "~" + ch[1:] if o >= 0 else "  ~"
            if i in mark:
                ch = "*" + ch[1:]
            row.append(ch)
        out.append(" ".join(row))
    return "\n".join(out)
