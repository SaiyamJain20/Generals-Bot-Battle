"""Verbatim-copy audit: does the submission share long token runs with any third-party bot source?

Tokens: identifiers, numbers, strings collapsed, operators; comments/whitespace removed.
A shared run of >= K consecutive tokens is reported with the file and the longest match.
"""
import hashlib
import io
import os
import re
import sys
import tokenize

K = 25
SUB = sys.argv[1]
ROOTS = sys.argv[2:]
TOK_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*|==|!=|<=|>=|//|\*\*|->|[^\sA-Za-z0-9_]")


def py_tokens(src):
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                          tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            if t.type == tokenize.STRING:
                out.append("STR")
            else:
                out.append(t.string)
    except Exception:
        return generic_tokens(src)
    return out


def generic_tokens(src):
    src = re.sub(r"//[^\n]*|/\*.*?\*/|#[^\n]*", " ", src, flags=re.S)
    src = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", " STR ", src)
    return TOK_RE.findall(src)


def shingles(toks):
    return {hashlib.md5(" ".join(toks[i:i + K]).encode()).hexdigest(): i for i in range(len(toks) - K + 1)}


def longest_run(a, b):
    # longest common contiguous token run (DP over the shared region; inputs are modest)
    best, pos = 0, 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, pos = cur[j], i
        prev = cur
    return best, pos


sub_src = open(SUB, encoding="utf-8").read()
sub_toks = py_tokens(sub_src)
sub_sh = shingles(sub_toks)
print(f"submission tokens: {len(sub_toks)}; shingles: {len(sub_sh)}")
hits = []
nfiles = 0
for root in ROOTS:
    for dp, dn, fn in os.walk(root):
        if any(x in dp for x in ("site-packages", "node_modules", ".venv", "_pylib", "__pycache__", ".git")):
            continue
        for f in fn:
            if not f.endswith((".py", ".cpp", ".hpp", ".h", ".rs", ".ts", ".js", ".go", ".java")):
                continue
            path = os.path.join(dp, f)
            try:
                src = open(path, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            nfiles += 1
            toks = py_tokens(src) if f.endswith(".py") else generic_tokens(src)
            sh = shingles(toks)
            common = [i for h, i in sub_sh.items() if h in sh]
            if common:
                hits.append((len(common), path, toks))
print(f"files scanned: {nfiles}; files sharing a >= {K}-token run: {len(hits)}")
for n, path, toks in sorted(hits, reverse=True)[:15]:
    L, pos = longest_run(sub_toks, toks) if len(toks) < 60000 else (None, None)
    snippet = " ".join(sub_toks[pos - L:pos])[:300] if L else ""
    print(f"\n{n:5d} shared {K}-grams | longest run {L} tokens | {path}\n   e.g.: {snippet}")
