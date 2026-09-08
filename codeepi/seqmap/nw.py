from __future__ import annotations

from typing import List, Tuple, Dict



_BLOSUM62_RAW = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4
B -2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
Z -1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
X  0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4
* -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
"""


def _load_blosum62() -> Dict[Tuple[str, str], int]:
    lines = [ln for ln in _BLOSUM62_RAW.strip().splitlines() if ln.strip()]
    header = lines[0].split()
    m: Dict[Tuple[str, str], int] = {}
    for row in lines[1:]:
        parts = row.split()
        r = parts[0]
        for c, v in zip(header, parts[1:]):
            m[(r, c)] = int(v)
    return m


BLOSUM62: Dict[Tuple[str, str], int] = _load_blosum62()


def _score(a: str, b: str) -> int:
    if (a, b) in BLOSUM62:
        return BLOSUM62[(a, b)]
    if (b, a) in BLOSUM62:
        return BLOSUM62[(b, a)]
    return BLOSUM62[("X", "X")]


_NEG_INF = -(10 ** 15)


def needleman_wunsch(
    seq1: str,
    seq2: str,
    open_gap: float = -10.0,
    extend_gap: float = -1.0,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], float]:
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return [], [], 0.0

    M = [[_NEG_INF] * (m + 1) for _ in range(n + 1)]
    X = [[_NEG_INF] * (m + 1) for _ in range(n + 1)]
    Y = [[_NEG_INF] * (m + 1) for _ in range(n + 1)]
    TM = [[0] * (m + 1) for _ in range(n + 1)]
    TX = [[0] * (m + 1) for _ in range(n + 1)]
    TY = [[0] * (m + 1) for _ in range(n + 1)]

    M[0][0] = 0.0
    for i in range(1, n + 1):
        X[i][0] = open_gap + extend_gap * (i - 1)
        TX[i][0] = 2
    for j in range(1, m + 1):
        Y[0][j] = open_gap + extend_gap * (j - 1)
        TY[0][j] = 3

    for i in range(1, n + 1):
        a = seq1[i - 1]
        for j in range(1, m + 1):
            b = seq2[j - 1]
            s = _score(a, b)
            cand = (M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1])
            best = max(cand)
            M[i][j] = best + s
            TM[i][j] = 1 + cand.index(best)  
            cand_x = (M[i - 1][j] + open_gap, X[i - 1][j] + extend_gap)
            if cand_x[0] >= cand_x[1]:
                X[i][j] = cand_x[0]; TX[i][j] = 1
            else:
                X[i][j] = cand_x[1]; TX[i][j] = 2
            cand_y = (M[i][j - 1] + open_gap, Y[i][j - 1] + extend_gap)
            if cand_y[0] >= cand_y[1]:
                Y[i][j] = cand_y[0]; TY[i][j] = 1
            else:
                Y[i][j] = cand_y[1]; TY[i][j] = 3

    end_scores = (M[n][m], X[n][m], Y[n][m])
    state = 1 + end_scores.index(max(end_scores))
    score = max(end_scores)

    i, j = n, m
    pairs: List[Tuple[int, int, int]] = []   
    while i > 0 or j > 0:
        if state == 1:                      
            pairs.append((i - 1, j - 1, 1))
            frm = TM[i][j]
            i -= 1; j -= 1
            state = frm if frm in (1, 2, 3) else 1
        elif state == 2:                    
            pairs.append((i - 1, -1, 2))
            frm = TX[i][j]
            i -= 1
            state = frm
        elif state == 3:                    
            pairs.append((-1, j - 1, 3))
            frm = TY[i][j]
            j -= 1
            state = frm
        else:
            break
    pairs.reverse()

    aligned1: List[Tuple[int, int]] = []
    aligned2: List[Tuple[int, int]] = []
    k = 0
    while k < len(pairs):
        if pairs[k][2] != 1:
            k += 1; continue
        s0 = pairs[k][0]; a0 = pairs[k][1]
        s1 = s0 + 1;      a1 = a0 + 1
        k += 1
        while k < len(pairs) and pairs[k][2] == 1 and pairs[k][0] == s1 and pairs[k][1] == a1:
            s1 += 1; a1 += 1; k += 1
        aligned1.append((s0, s1))
        aligned2.append((a0, a1))
    return aligned1, aligned2, float(score)



def greedy_same_char_map(seqres: str, atmseq: str) -> dict:
    m: dict = {}
    i = j = 0
    n, N = len(seqres), len(atmseq)
    while i < n and j < N:
        if seqres[i] == atmseq[j]:
            m[i] = j
            i += 1
            j += 1
        else:
            i += 1
    return m


def nw_aligned_map(seqres: str, atmseq: str,
                   open_gap: float = -10.0,
                   extend_gap: float = -1.0) -> dict:
    a1, a2, _ = needleman_wunsch(seqres, atmseq,
                                 open_gap=open_gap,
                                 extend_gap=extend_gap)
    out: dict = {}
    for (s0, s1), (b0, b1) in zip(a1, a2):
        for k in range(s1 - s0):
            out[s0 + k] = b0 + k
    return out
