# stream_worker.py
import chess
import chess.pgn
import numpy as np
import collections
from typing import Iterator, Iterable, Dict, Any, Tuple, Optional, List
import os

# Assuming mapping.py is in the same directory or accessible in PYTHONPATH
import mapping # This is the mapping.py file you provided
# Assuming stream_reader.py is in the same directory or accessible
import stream_reader

# Constants for bitboard encoding
PIECE_TYPES_COUNT = 6  # P, N, B, R, Q, K (white: planes 0–5, black: planes 6–11)
NUM_BITBOARD_PLANES = 14 # 12 piece planes + castling + en-passant
CASTLING_PLANE_IDX = 12  # castling rights plane
EP_PLANE_IDX = 13        # en-passant target plane
NUM_POLICY_PLANES = 73  # 0–72 for moves

# --- Helper function for algebraic bitboard flipping ---
def _flip_bitboards_algebraic(bitboards: np.ndarray, flip_type: str) -> np.ndarray:
    """
    Flips bitboards algebraically (H, V, HV) for contrastive learning.
    'v' also swaps piece color planes.

    Args:
        bitboards (np.ndarray): The original bitboard tensor (planes x 8 x 8).
        flip_type (str): 'h' for horizontal, 'v' for vertical, 'hv' for both.

    Returns:
        np.ndarray: The flipped bitboard tensor.
    """
    flipped_bb = bitboards.copy()

    if 'h' in flip_type:  # Horizontal flip
        for i in range(flipped_bb.shape[0]):
            flipped_bb[i, :, :] = np.fliplr(flipped_bb[i, :, :])

    if 'v' in flip_type:  # Vertical flip
        # Spatial flipud all planes first
        for i in range(flipped_bb.shape[0]):
            flipped_bb[i, :, :] = np.flipud(flipped_bb[i, :, :])
        
        # Swap White piece planes (0-5) with Black piece planes (6-11)
        temp_white_planes = flipped_bb[0:PIECE_TYPES_COUNT, :, :].copy()
        flipped_bb[0:PIECE_TYPES_COUNT, :, :] = flipped_bb[PIECE_TYPES_COUNT : PIECE_TYPES_COUNT*2, :, :]
        flipped_bb[PIECE_TYPES_COUNT : PIECE_TYPES_COUNT*2, :, :] = temp_white_planes
        
        # Castling (CASTLING_PLANE_IDX) and EP (EP_PLANE_IDX) are already spatially flipped.
    return flipped_bb

# --- Core Feature Encoding Functions (mostly unchanged) ---

def _get_plane_for_move(board_turn: chess.Color, move: chess.Move, piece_type_at_from_sq: Optional[chess.PieceType]) -> int:
    """
    Determines the policy plane (0-72) for a given move using mapping.py logic.
    `board_turn` is chess.WHITE or chess.BLACK.
    `piece_type_at_from_sq` is the type of the piece making the move.
    Returns:
        int: Plane index (0-72) or -1 if not representable.
    """
    if move == chess.Move.null():
        return -1

    if move.promotion and move.promotion != chess.QUEEN:
        from_r, from_f = divmod(move.from_square, 8)
        to_r, to_f = divmod(move.to_square, 8)
        promo_symbol = chess.piece_symbol(move.promotion).lower()
        plane = mapping.underpromotion_plane(from_r, from_f, to_r, to_f, promo_symbol)
        if plane != -1:
            return plane

    if move.from_square is None or move.to_square is None:
        return -1
    
    plane = mapping.MOVE2PLANE[move.from_square, move.to_square]
    return int(plane)


def _encode_bitboards(board: chess.Board, rep_count: int) -> np.ndarray:
    """
    Encodes the board state into a bitboard tensor without global planes.
    Returns:
        np.ndarray: The bitboard tensor of shape (14, 8, 8).
    """
    bitboards = np.zeros((NUM_BITBOARD_PLANES, 8, 8), dtype=np.float32)

    for piece_type_val in chess.PIECE_TYPES: 
        plane_offset_white = piece_type_val - 1
        plane_offset_black = piece_type_val - 1 + PIECE_TYPES_COUNT
        for sq in board.pieces(piece_type_val, chess.WHITE):
            r, f = divmod(sq, 8)
            bitboards[plane_offset_white, r, f] = 1.0
        for sq in board.pieces(piece_type_val, chess.BLACK):
            r, f = divmod(sq, 8)
            bitboards[plane_offset_black, r, f] = 1.0

    # Castling rights plane
    if board.has_queenside_castling_rights(chess.WHITE): bitboards[CASTLING_PLANE_IDX, 0, 0] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):  bitboards[CASTLING_PLANE_IDX, 0, 7] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): bitboards[CASTLING_PLANE_IDX, 7, 0] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):  bitboards[CASTLING_PLANE_IDX, 7, 7] = 1.0

    # En-passant target plane
    if board.ep_square is not None:
        r, f = divmod(board.ep_square, 8)
        bitboards[EP_PLANE_IDX, r, f] = 1.0
        
    return bitboards


def _encode_policy_target(board: chess.Board, move: chess.Move) -> np.ndarray:
    """
    Encodes the played move into a one-hot policy tensor.
    Returns:
        np.ndarray: The policy tensor.
    """
    policy = np.zeros((NUM_POLICY_PLANES, 8, 8), dtype=np.float32)
    piece_at_from = board.piece_at(move.from_square)
    piece_type_at_from_sq = piece_at_from.piece_type if piece_at_from else None

    plane = _get_plane_for_move(board.turn, move, piece_type_at_from_sq)

    if plane != -1 and 0 <= plane < NUM_POLICY_PLANES:
        r_from, f_from = divmod(move.from_square, 8)
        policy[plane, r_from, f_from] = 1.0
    return policy


def _generate_legal_mask(board: chess.Board) -> np.ndarray:
    """
    Generates a mask of legal moves.
    Returns:
        np.ndarray: The legal move mask tensor (dtype=bool).
    """
    legal_mask = np.zeros((NUM_POLICY_PLANES, 8, 8), dtype=bool)
    for legal_move in board.legal_moves:
        piece_at_from = board.piece_at(legal_move.from_square)
        piece_type_at_from_sq = piece_at_from.piece_type if piece_at_from else None
        plane = _get_plane_for_move(board.turn, legal_move, piece_type_at_from_sq)
        if plane != -1 and 0 <= plane < NUM_POLICY_PLANES:
            r_from, f_from = divmod(legal_move.from_square, 8)
            legal_mask[plane, r_from, f_from] = True
    return legal_mask


def _calculate_value_target(result_str: str) -> np.ndarray:
    """
    Calculates the one-hot value target: [Win, Draw, Loss] for White.
    Returns:
        np.ndarray: The value target vector.
    """
    value = np.zeros((3,), dtype=np.uint8) 
    if result_str == "1-0": value[0] = 1
    elif result_str == "1/2-1/2": value[1] = 1
    elif result_str == "0-1": value[2] = 1
    return value


def _calculate_material(board: chess.Board) -> Tuple[np.int8, np.uint8]:
    """
    Calculates raw material difference and its categorical representation.
    Returns:
        Tuple[np.int8, np.uint8]: Raw material diff, categorical material.
    """
    val_map = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    white_material = sum(len(board.pieces(pt, chess.WHITE)) * val_map[pt] for pt in val_map)
    black_material = sum(len(board.pieces(pt, chess.BLACK)) * val_map[pt] for pt in val_map)
    
    material_raw = white_material - black_material
    
    cat = 0 
    if material_raw == -4: cat = 1
    elif material_raw == -3: cat = 2
    elif material_raw == -2: cat = 3
    elif material_raw == -1: cat = 4
    elif material_raw == 0:  cat = 5
    elif material_raw == 1:  cat = 6
    elif material_raw == 2:  cat = 7
    elif material_raw == 3:  cat = 8
    elif material_raw == 4:  cat = 9
    elif material_raw >= 5: cat = 10
        
    return np.int8(material_raw), np.uint8(cat)

# --- New Helper for WDL Target from Current Player's Perspective ---
def _calculate_player_wdl_target(result_str: str, player_turn: chess.Color) -> np.int8:
    """Calculates WDL target from the current player's perspective.
    0: Loss, 1: Draw, 2: Win for the current player.
    """
    if result_str == "1/2-1/2":
        return np.int8(1) # Draw
    
    white_won = result_str == "1-0"
    
    if player_turn == chess.WHITE:
        if white_won:
            return np.int8(2) # White (current player) won
        else:
            return np.int8(0) # White (current player) lost (Black won)
    else: # player_turn == chess.BLACK
        if not white_won: # Black won
            return np.int8(2) # Black (current player) won
        else:
            return np.int8(0) # Black (current player) lost (White won)

# --- Main Streaming Function ---

def stream_samples(paths: Iterable[str], *, flips: bool = True, plys_to_end_cap: int = 512, is_lichess_source_override: Optional[bool] = None) -> Iterable[List[Dict[str, Any]]]:
    """
    High-level generator: takes one or many input paths, yields lists of feature dicts (one list per game).
    
    NOTE: The 'flips' parameter is now ignored - flipped bitboards are computed on-the-fly 
    in the dataloader for memory efficiency.
    
    Returns:
        Iterable[List[Dict[str, Any]]]: An iterator over lists of sample dictionaries, where each list corresponds to one game.
    """
    for path_item in paths:
        # Determine if this source should be treated as Lichess
        treat_as_lichess_source: bool
        if is_lichess_source_override is not None:
            treat_as_lichess_source = is_lichess_source_override
        else:
            # Fallback: inspect the path_item itself (original behavior for direct calls)
            filename_lower = os.path.basename(path_item.lower())
            treat_as_lichess_source = "lichess" in filename_lower

        for game in stream_reader.iter_games(path_item):
            samples_for_current_game: List[Dict[str, Any]] = []
            # DEBUG PRINT:
            # print(f"Processing game from {path_item} with headers: {game.headers}")
            try:
                result_str = game.headers.get("Result", "*")
                # DEBUG PRINT:
                # print(f"  Result: {result_str}")
                if result_str == "*": 
                    # print("  Skipping: Result is *")
                    continue
                
                value_target_orig = _calculate_value_target(result_str)
                if np.sum(value_target_orig) == 0: continue 

                # Determine source_flag based on treat_as_lichess_source
                # This flag is for the output sample. The Lichess *filtering* below also uses treat_as_lichess_source.
                current_source_flag = np.uint8(1 if treat_as_lichess_source else 0)

                # Filter Lichess games based on event type and player ratings
                if treat_as_lichess_source: # Use the determined flag for filtering
                    event = game.headers.get("Event", "")
                    if event not in ["Rated Blitz game", "Rated Classical game"]:
                        continue
                    
                    # Check both players' ratings
                    white_rating = int(game.headers.get("WhiteElo", "0"))
                    black_rating = int(game.headers.get("BlackElo", "0"))
                    if white_rating < 1800 or black_rating < 1800:
                        continue

                initial_board_for_game = game.board() 
                is960_game = initial_board_for_game.chess960 # Renamed for clarity in this scope

                # source_flag is already determined as current_source_flag above
                variant_flag = np.uint8(1 if is960_game else 0)

                repetition_map = collections.Counter()
                
                # Create a unique game_id
                game_headers = game.headers
                game_id_parts = [
                    path_item,
                    game_headers.get("Site", "UnknownSite"),
                    game_headers.get("Date", "UnknownDate"),
                    game_headers.get("Round", "?"),
                    game_headers.get("White", "UnknownWhite"),
                    game_headers.get("Black", "UnknownBlack"),
                    game_headers.get("Result", "*")
                ]
                game_id = "_".join(part.replace(" ", "_") for part in game_id_parts)

                # Correctly get ply count and node iterator
                mainline_moves_list = list(game.mainline_moves()) # Get all moves first for ply count
                total_plys_in_game = len(mainline_moves_list)
                node_iterator = game.mainline() # Get the iterator for nodes

                # DEBUG PRINT:
                # print(f"[stream_samples] Processing game: {game.headers.get('Event', '?')} | Moves: {total_plys_in_game}")

                if total_plys_in_game == 0: 
                    # print("  Skipping: Total plys is 0")
                    continue

                current_board_state = initial_board_for_game.copy()

                # Iterate using the moves list to avoid issues with node iterator after board manipulation
                for ply_idx, move_obj_orig in enumerate(mainline_moves_list):
                    # Fetch the corresponding node from the game if needed for other attributes, though move is primary here
                    # For this loop, we primarily need the move object itself.
                    # The board state is advanced manually.
                    
                    # Ensure move_obj_orig is a chess.Move object, as game.mainline_moves() yields them.
                    if not isinstance(move_obj_orig, chess.Move):
                        # This might happen if the PGN is structured with nodes that aren't simple moves (e.g. null moves in odd places)
                        # Or if the mainline_moves() iterator behaves unexpectedly. For safety, skip such entries.
                        # print(f"DEBUG: Expected a chess.Move object, got {type(move_obj_orig)}. Skipping this entry.")
                        if current_board_state.is_game_over(): break # Stop if game already ended due to this
                        continue

                    try:
                        current_zobrist_key = current_board_state.transposition_key
                    except AttributeError: 
                        current_zobrist_key = (
                            current_board_state.board_fen(),
                            current_board_state.castling_rights, 
                            current_board_state.ep_square, 
                            current_board_state.turn
                        )
                    rep_count_orig = repetition_map[current_zobrist_key]

                    # Global flags: side-to-move and repetition indicators
                    stm_flag = np.uint8(1 if current_board_state.turn == chess.WHITE else 0)
                    rep1_flag = np.uint8(1 if rep_count_orig >= 1 else 0)
                    rep2_flag = np.uint8(1 if rep_count_orig >= 2 else 0)

                    bb_orig = _encode_bitboards(current_board_state, rep_count_orig)
                    pt_orig = _encode_policy_target(current_board_state, move_obj_orig)
                    lm_orig = _generate_legal_mask(current_board_state)
                    
                    plys_remaining = total_plys_in_game - (ply_idx + 1)
                    current_ply_target = np.uint16(min(max(0, plys_remaining), plys_to_end_cap)) # Renamed
                    material_raw, material_cat = _calculate_material(current_board_state)
                    current_wdl_target = _calculate_player_wdl_target(result_str, current_board_state.turn)

                    # Metadata for the original sample (only essential data)
                    original_sample_metadata = {
                        "total_plys_in_game": total_plys_in_game,
                        "ply_target": current_ply_target,
                        "material_raw": material_raw,
                        "material_cat": material_cat,
                        "source_flag": current_source_flag,
                        "is960": is960_game,
                        "stm": stm_flag,
                        "rep1": rep1_flag,
                        "rep2": rep2_flag,
                        "wdl_target": current_wdl_target,
                    }
                    
                    output_sample = {
                        **original_sample_metadata,
                        "bitboards_original": bb_orig, # Renamed for clarity
                        "policy": pt_orig, 
                        "legal_mask": lm_orig,
                    }

                    # NOTE: Flipped bitboards are now computed on-the-fly in the dataloader
                    # instead of being pre-computed here to save memory
                    
                    # yield output_sample # MODIFIED: Do not yield individual sample here
                    samples_for_current_game.append(output_sample) # MODIFIED: Add to list

                    repetition_map[current_zobrist_key] += 1
                    current_board_state.push(move_obj_orig)

                # MODIFIED: Yield all samples for the game if any were collected
                if samples_for_current_game:
                    # print(f"[stream_samples] Yielding {len(samples_for_current_game)} samples for game. Keys in first sample: {list(samples_for_current_game[0].keys()) if samples_for_current_game else 'N/A'}")
                    yield samples_for_current_game

            except chess.IllegalMoveError:
                continue 
            except ValueError: 
                continue
            except Exception as e_inner: # Named the exception for logging
                # Enhanced debug print
                current_ply_for_debug = locals().get('ply_idx', -1)
                current_move_for_debug = locals().get('move_obj_orig', 'N/A')
                # print(f"DEBUG: stream_samples caught: {type(e_inner).__name__} - {e_inner} (Game: {path_item}, Ply_idx: {current_ply_for_debug}, Move: {current_move_for_debug})")
                # import traceback
                # traceback.print_exc() # Uncomment for full traceback if needed
                continue