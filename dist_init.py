# ----------------------------------------------------------------------
# dist_init.py   – generate initial 65×65 bias matrix for Chess‑ViT
# ----------------------------------------------------------------------
import heapq, math, numpy as np

# ---------- parameters (hard‑coded) -----------------------------------
W_KING   = 1.0
W_KNIGHT = 1.3
W_SLIDE  = 0.05          # incremental cost per step beyond the first
SIG_K    = 6.0           # sigmoid slope
SIG_X0   = 3.0           # sigmoid shift

# ---------- helper: moves for a square --------------------------------
def in_board(r, f): return 0 <= r < 8 and 0 <= f < 8

dirs_king  = [(dr, df) for dr in (-1,0,1) for df in (-1,0,1) if not dr == df == 0]
dirs_slide = dirs_king                           # same eight directions
dirs_knight= [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]

def neighbours(r, f):
    # king steps
    for dr, df in dirs_king:
        rr, ff = r+dr, f+df
        if in_board(rr, ff):
            yield (rr, ff), W_KING
    # knight jumps
    for dr, df in dirs_knight:
        rr, ff = r+dr, f+df
        if in_board(rr, ff):
            yield (rr, ff), W_KNIGHT
    # sliding moves (rook / bishop / queen)
    for dr, df in dirs_slide:
        rr, ff = r+dr, f+df
        length = 1
        while in_board(rr, ff):
            cost = 1.0 + W_SLIDE*length
            yield (rr, ff), cost
            rr += dr;  ff += df;  length += 1

# ---------- pre‑compute index <-> square mapping ----------------------
idx2sq = [(r,f) for r in range(8) for f in range(8)]  # rank major
sq2idx = {sq: i for i, sq in enumerate(idx2sq)}

# ---------- Dijkstra distances (64×64) --------------------------------
dist = np.full((64, 64), np.inf, dtype=np.float32)

for s in range(64):
    dist[s, s] = 0.0
    pq = [(0.0, s)]                     # (distance, node)
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u != dist[s, u]:           # stale entry
            continue
        r_u, f_u = idx2sq[u]
        for (r_v, f_v), w in neighbours(r_u, f_u):
            v = sq2idx[(r_v, f_v)]
            if d_u + w < dist[s, v]:
                dist[s, v] = d_u + w
                heapq.heappush(pq, (dist[s, v], v))

# ---------- convert to bias in [-1, 0] --------------------------------
d_norm  = np.clip((dist - 1.0) / 0.35, 0.0, 1.0)      # rescale to [0,1]
bias64  = -1.0 / (1.0 + np.exp(-SIG_K * d_norm + SIG_X0))

# ---------- set diagonals to zero -------------------------------------
for i in range(64):
    bias64[i, i] = 0.0

# ---------- assemble 65×65 matrix (index 0 = CLS) ---------------------
bias = np.zeros((65, 65), dtype=np.float32)
bias[1:, 1:] = bias64

np.save('dist_init.npy', bias)

# ---------- sanity checks ---------------------------------------------
B = bias
assert B.shape == (65, 65)
assert np.allclose(B, B.T, atol=1e-6)
assert np.allclose(B.diagonal(), 0.0)
assert (B[0] == 0).all() and (B[:, 0] == 0).all()
assert -1.0001 <= B.min() <= -0.94 and B.max() <= 0.0
print("✔ dist_init.npy saved   (min %.3f, max %.3f)" % (B.min(), B.max()))

# ---------- pretty‑print one example row/col for manual inspection ----
ref_sq = (3, 5)                         # user‑chosen reference
ref_id = sq2idx[ref_sq]

def to_board(arr):
    return arr.reshape(8, 8)[::-1]      # flip for traditional chess view

print("\nGraph‑distance from square (3,5):")
print(np.round(to_board(dist[ref_id]), 2))

print("\nBias values from square (3,5):")
print(np.round(to_board(bias64[ref_id].reshape(8,8)[::-1]), 3))