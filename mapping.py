import numpy as np

##############################################################################
#  Move-type  <-->  plane lookup tables
##############################################################################

# −1  means "illegal / not represented"
MOVE2PLANE = np.full((64, 64), -1, dtype=np.int8)

PROM_PIECE_TO_IDX = {'n': 0, 'N': 0,
                     'b': 1, 'B': 1,
                     'r': 2, 'R': 2}

FILE_CHANGE_TO_IDX = {-1: 0, 0: 1, 1: 2}        # dst_file - src_file


def underpromotion_plane(src_rank: int, src_file: int,
                         dst_rank: int, dst_file: int,
                         promo_piece: str) -> int:
    """
    Return the policy-plane index (64-72) for a legal pawn under-promotion.
    Returns −1 if the move is not an under-promotion covered by the scheme.

    Parameters
    ----------
    src_rank, src_file : int   0 … 7  coordinates of the pawn before the move
    dst_rank, dst_file : int   0 … 7  coordinates after the move
    promo_piece        : str   one of 'n','b','r' (case-insensitive)

    Example
    -------
    >>> underpromotion_plane(6, 4, 7, 4, 'n')   # e7-e8 = Knight
    65
    >>> underpromotion_plane(1, 2, 0, 1, 'R')   # c2-b1 = Rook
    70
    """
    # 1) correct ranks (white 6→7, black 1→0)?
    if not ((src_rank == 6 and dst_rank == 7) or
            (src_rank == 1 and dst_rank == 0)):
        return -1

    # 2) piece type mapped to 0,1,2
    piece_idx = PROM_PIECE_TO_IDX.get(promo_piece)
    if piece_idx is None:
        return -1

    # 3) file change mapped to 0,1,2
    file_diff = dst_file - src_file
    fc_idx = FILE_CHANGE_TO_IDX.get(file_diff)
    if fc_idx is None:
        return -1

    # 4) build plane number
    return 64 + 3 * piece_idx + fc_idx

# ---------------------------------------------------------------------------
# 1) knight moves – planes 56-63
# ---------------------------------------------------------------------------
KNIGHT_OFFSETS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  ( 1, -2), ( 1, 2), ( 2, -1), ( 2, 1)]
for plane, (dr, df) in enumerate(KNIGHT_OFFSETS, start=56):
    for src in range(64):
        r0, f0 = divmod(src, 8)
        r1, f1 = r0 + dr, f0 + df
        if 0 <= r1 < 8 and 0 <= f1 < 8:
            MOVE2PLANE[src, r1 * 8 + f1] = plane

# ---------------------------------------------------------------------------
# 2) sliding rays – planes 0-55   (same order used by `move_type()`)
# ---------------------------------------------------------------------------
def fill_ray(first_plane: int, dr: int, df: int):
    # write the 7 distances of one direction
    for src in range(64):
        r0, f0 = divmod(src, 8)
        for dist in range(1, 8):                         # 1 … 7
            r1, f1 = r0 + dr * dist, f0 + df * dist
            if not (0 <= r1 < 8 and 0 <= f1 < 8):
                break
            MOVE2PLANE[src, r1 * 8 + f1] = first_plane + (dist - 1)

fill_ray( 0,  1,  0)   # north (corrected dr)
fill_ray( 7,  1,  1)   # north-east (corrected dr)
fill_ray(14,  0,  1)   # east (was correct)
fill_ray(21, -1,  1)   # south-east (corrected dr)
fill_ray(28, -1,  0)   # south (corrected dr)
fill_ray(35, -1, -1)   # south-west (corrected dr)
fill_ray(42,  0, -1)   # west (was correct)
fill_ray(49,  1, -1)   # north-west (corrected dr)

# ---------------------------------------------------------------------------
# 3) under-promotion moves – planes 64-72
# ---------------------------------------------------------------------------
# Iterate through possible promotion scenarios
# (White promotions: pawn on rank 6 moving to rank 7)
# (Black promotions: pawn on rank 1 moving to rank 0)

for piece_char_lower in ['n', 'b', 'r']: # Iterate over 'n', 'b', 'r'
    # White promotions
    for src_f_idx in range(8): # Source file index
        src_rank_idx = 6 # 7th rank
        src_sq = src_rank_idx * 8 + src_f_idx
        for file_offset in [-1, 0, 1]: # Pawn captures left, moves straight, or captures right
            dst_f_idx = src_f_idx + file_offset
            if 0 <= dst_f_idx < 8:
                dst_rank_idx = 7 # 8th rank
                dst_sq = dst_rank_idx * 8 + dst_f_idx
                # Check if this is a valid pawn move (diagonal capture or straight push)
                if abs(src_f_idx - dst_f_idx) == 1 or src_f_idx == dst_f_idx: # Capture or push
                    plane = underpromotion_plane(src_rank_idx, src_f_idx, dst_rank_idx, dst_f_idx, piece_char_lower)
                    # if plane != -1: # Do not overwrite MOVE2PLANE here for underpromotions
                        # MOVE2PLANE[src_sq, dst_sq] = plane 

    # Black promotions
    for src_f_idx in range(8): # Source file index
        src_rank_idx = 1 # 2nd rank
        src_sq = src_rank_idx * 8 + src_f_idx
        for file_offset in [-1, 0, 1]: # Pawn captures left, moves straight, or captures right
            dst_f_idx = src_f_idx + file_offset
            if 0 <= dst_f_idx < 8:
                dst_rank_idx = 0 # 1st rank
                dst_sq = dst_rank_idx * 8 + dst_f_idx
                if abs(src_f_idx - dst_f_idx) == 1 or src_f_idx == dst_f_idx: # Capture or push
                    plane = underpromotion_plane(src_rank_idx, src_f_idx, dst_rank_idx, dst_f_idx, piece_char_lower)
                    # if plane != -1: # Do not overwrite MOVE2PLANE here for underpromotions
                        # MOVE2PLANE[src_sq, dst_sq] = plane

# ---------------------------------------------------------------------------
# 4) inverse table  PLANE2MOVE[plane] -> list of (src, dst) tuples
#    The original PLANE2MOVE was a dict {src: dst} which only works if each plane maps to one src.
#    For underpromotions, a single plane can be reached from multiple src squares (e.g., e7d8=N and c7b8=N might be the same plane)
#    Changing to list of tuples for each plane.
# ---------------------------------------------------------------------------
max_plane_val = int(np.max(MOVE2PLANE)) # Recalculate max_plane after adding underpromotions
PLANE2MOVE = [[] for _ in range(max_plane_val + 1)] # Initialize list of lists
for src_sq_iter in range(64):
    for dst_sq_iter in range(64):
        plane_val = MOVE2PLANE[src_sq_iter, dst_sq_iter]
        if plane_val != -1:
            # Ensure the list is long enough
            while len(PLANE2MOVE) <= plane_val:
                PLANE2MOVE.append([])
            PLANE2MOVE[plane_val].append((src_sq_iter, dst_sq_iter))

# ---------------------------------------------------------------------------
# Under-promotion plane helper
# ---------------------------------------------------------------------------
#
#  • Plane layout wanted by the question
#        64 … 72  (inclusive)
#        64-66 : promote to Knight  (file −1,0,+1)
#        67-69 : promote to Bishop  (file −1,0,+1)
#        70-72 : promote to Rook    (file −1,0,+1)
#
#  • The function returns -1 for "not an under-promotion".
# ---------------------------------------------------------------------------

# Ensure all underpromotion moves are covered in MOVE2PLANE and PLANE2MOVE
for src_sq in range(64):
    src_rank, src_file = divmod(src_sq, 8)
    for dst_sq in range(64):
        dst_rank, dst_file = divmod(dst_sq, 8)
        for promo_piece in ['n', 'b', 'r']:
            plane = underpromotion_plane(src_rank, src_file, dst_rank, dst_file, promo_piece)
            if plane != -1:
                # MOVE2PLANE[src_sq, dst_sq] = plane # Do not overwrite MOVE2PLANE here
                # Ensure PLANE2MOVE list is large enough
                while len(PLANE2MOVE) <= plane:
                    PLANE2MOVE.append([])
                if (src_sq, dst_sq) not in PLANE2MOVE[plane]:
                    PLANE2MOVE[plane].append((src_sq, dst_sq))