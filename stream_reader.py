# stream_reader.py
import chess
import chess.pgn
import tarfile
import io
import gzip
import bz2
from typing import Iterator, TextIO, BinaryIO, Optional

# Constants
# MIN_GAME_LENGTH_FOR_ERROR_SKIP = 5 # If a game has fewer moves before error, skip_game might be less effective # Removed

def _iter_games_from_pgn_stream(pgn_stream: TextIO, file_path_for_error_msg: str = "stream") -> Iterator[chess.pgn.Game]:
    """Helper to yield games from an already opened PGN text stream."""
    games_yielded = 0
    while True:
        game_node = None # Initialize game_node
        pgn_stream_tell = -1 # Initialize, may not be usable if stream not seekable
        try:
            # Try to get current stream position if seekable.
            # This is for potential rewind if read_game fails very early.
            if pgn_stream.seekable():
                try:
                    pgn_stream_tell = pgn_stream.tell()
                except io.UnsupportedOperation:
                    pgn_stream_tell = -1 # Mark as unusable/failed to get position

            game_node = chess.pgn.read_game(pgn_stream)

            if game_node is None:
                break # End of file or unrecoverable stream error

            # If python-chess reported errors during parsing of this game's content
            if game_node.errors:
                # print(f"Info: Skipping game from {file_path_for_error_msg} due to parsing errors: {game_node.errors}")
                continue
            
            # Check for common PGN errors that python-chess might tolerate but we consider invalid
            if not game_node.mainline_moves() and game_node.headers.get("Result") == "*":
                # print(f"Info: Skipping game with no moves and '*' result from {file_path_for_error_msg}.")
                continue

            # Apply Lichess-specific filters
            # TODO: This lichess check is very basic.
            # Consider a more robust way if PGNs can come from mixed sources within one file
            # or if "lichess" might appear spuriously in other filenames.
            is_lichess_file = "lichess" in file_path_for_error_msg.lower()
            if is_lichess_file:
                termination = game_node.headers.get("Termination", "").lower()
                if termination != "normal":
                    # print(f"Info: Skipping Lichess game from {file_path_for_error_msg} due to termination: {game_node.headers.get('Termination')}")
                    continue

            games_yielded += 1
            yield game_node

        except (ValueError, IndexError) as parse_err: # Catch common PGN parsing errors
            error_message = f"PGN Parsing Error ({type(parse_err).__name__}) in {file_path_for_error_msg}: {parse_err}."
            print(f"ERROR_CONTEXT: {error_message}")

            # Check if the source was likely gzipped and add a diagnostic note
            if any(ext in file_path_for_error_msg.lower() for ext in (".gz", ".tgz")):
                print("ERROR_CONTEXT: Note: The input stream was from a GZip compressed source.")
                print("ERROR_CONTEXT: The following dump is the content *after* GZip decompression and UTF-8 text decoding (with replacements for errors).")
                print("ERROR_CONTEXT: If this dump appears as binary garbage, the .gz archive likely does not contain UTF-8 encoded PGN text or is corrupted.")
            
            lines_to_print_on_error = 30 # Number of lines of PGN to dump
            
            # Try to rewind to the start of the problematic game if possible
            if game_node is None and pgn_stream_tell != -1 and pgn_stream.seekable():
                try:
                    pgn_stream.seek(pgn_stream_tell)
                    print(f"ERROR_CONTEXT: Rewound to stream position {pgn_stream_tell} to show PGN context.")
                except io.UnsupportedOperation:
                    print(f"ERROR_CONTEXT: Stream seekable but seek to {pgn_stream_tell} failed (UnsupportedOperation). Printing from current error point.")
                except Exception as e_seek:
                    print(f"ERROR_CONTEXT: Error seeking to {pgn_stream_tell}: {e_seek}. Printing from current error point.")
            elif game_node is not None: # Error occurred after game_node was successfully read, so pgn_stream_tell is for next game
                 print(f"ERROR_CONTEXT: Error occurred after game was partially read. PGN dump will be from point of error, not game start.")
            else: # game_node is None, and pgn_stream_tell was not usable or stream not seekable
                print(f"ERROR_CONTEXT: Cannot rewind to game start (stream not seekable or position not recorded). Printing from current error point.")

            print(f"ERROR_CONTEXT: Dumping up to {lines_to_print_on_error} lines of PGN content from error point:")
            print("---------- PGN DUMP START ----------")
            try:
                for i in range(lines_to_print_on_error):
                    line = pgn_stream.readline()
                    if not line: # readline() returns empty string at EOF
                        print(f"ERROR_CONTEXT: <EOF reached after {i} lines of dump>")
                        break
                    print(line, end='') # Print line as is, readline() includes newline
            except Exception as e_read:
                print(f"ERROR_CONTEXT: Exception during PGN dump: {e_read}")
            print("----------- PGN DUMP END -----------")
            
            print(f"ERROR_CONTEXT: Aborting processing for {file_path_for_error_msg} due to PGN parsing error.")
            break # Abort processing this specific PGN stream / file member
        except Exception as e: # Catch other, more critical/unexpected errors (e.g., IO issues not caught by above)
            print(f"Critical Error: Unhandled exception reading/processing game from {file_path_for_error_msg}: {e}. Aborting this stream/file.")
            # import traceback # Uncomment for detailed debugging
            # traceback.print_exc() # Uncomment for detailed debugging
            break # Abort processing this specific PGN stream


def iter_games(path: str) -> Iterator[chess.pgn.Game]:
    """
    Detects file type (.pgn, .tar, .tar.gz, .tar.bz2, .pgn.gz, .pgn.bz2) 
    and yields chess.pgn.Game objects one by one.

    Returns:
        Iterator[chess.pgn.Game]: An iterator over chess games.
    """
    low_path = path.lower()
    
    if low_path.endswith(".pgn"):
        try:
            with open(path, 'rt', encoding='utf-8', errors='replace') as f:
                yield from _iter_games_from_pgn_stream(f, path)
        except Exception as e:
            print(f"Error opening/reading PGN file {path}: {e}")
    elif low_path.endswith((".tar", ".tar.gz", ".tar.bz2", ".tgz", ".tbz2")):
        # Use r|* for streaming mode if possible, though tarfile may still read some index data.
        # For true streaming of tar, one might need external libraries or more complex handling.
        # "r:*" attempts to auto-detect compression for non-streamed mode.
        # "r|gz", "r|bz2" for streamed.
        mode = "r:"
        if low_path.endswith(".gz") or low_path.endswith(".tgz"):
            mode = "r:gz"
        elif low_path.endswith(".bz2") or low_path.endswith(".tbz2"):
            mode = "r:bz2"
        
        try:
            # Using "r:*" for general tar files; tarfile handles compression detection.
            # For explicit streaming, use "r|gz" or "r|bz2" if extension is known.
            # Let's stick to "r:*" for simplicity unless streaming mode `r|*` is confirmed better.
            # The review mentioned `r|*` for streaming but tarfile itself might not fully support it for all ops.
            # Let's try specific modes for gz/bz2 if it's a tar archive.
            if low_path.endswith(".tar.gz") or low_path.endswith(".tgz"):
                open_mode = "r:gz"
            elif low_path.endswith(".tar.bz2") or low_path.endswith(".tbz2"):
                open_mode = "r:bz2"
            elif low_path.endswith(".tar"):
                 open_mode = "r:" # No compression
            else: # Should not happen due to outer if
                print(f"Warning: Unexpected tar extension {path}")
                open_mode = "r:*"


            with tarfile.open(path, open_mode) as tar:
                # print(f"DEBUG: Successfully opened TAR file: {path} with mode {open_mode}")
                for member_idx, member in enumerate(tar): # This part might not be fully streaming for all tarfile versions/modes
                    # print(f"DEBUG: Processing TAR member {member_idx}: {member.name}, isfile: {member.isfile()}")
                    if member.isfile():
                        member_binary_stream: Optional[BinaryIO] = None # Initialize to None
                        try:
                            member_binary_stream = tar.extractfile(member)
                            if member_binary_stream:
                                member_name_lower = member.name.lower()
                                stream_id_for_errors = f"{path}::{member.name}"
                                # print(f"DEBUG: Extracted TAR member {member.name}. Lowercase: {member_name_lower}")

                                # Outer try/finally ensures member_binary_stream is closed
                                try:
                                    pgn_text_stream_to_iterate: Optional[TextIO] = None
                                    if member_name_lower.endswith(".pgn"):
                                        # print(f"DEBUG: Member {member.name} is a .pgn file. Wrapping in TextIOWrapper.")
                                        pgn_text_stream_to_iterate = io.TextIOWrapper(member_binary_stream, encoding='utf-8', errors='replace')
                                    elif member_name_lower.endswith(".gz"): # Check for .pgn.gz or just .gz containing PGN
                                        # print(f"DEBUG: Member {member.name} is a .gz file. Opening with gzip.")
                                        # Ensure it's treated as a PGN if it's .gz (could be .pgn.gz)
                                        pgn_text_stream_to_iterate = gzip.open(member_binary_stream, 'rt', encoding='utf-8', errors='replace')
                                    elif member_name_lower.endswith(".bz2"): # Check for .pgn.bz2 or just .bz2 containing PGN
                                        # print(f"DEBUG: Member {member.name} is a .bz2 file. Opening with bz2.")
                                        pgn_text_stream_to_iterate = bz2.open(member_binary_stream, 'rt', encoding='utf-8', errors='replace')
                                    
                                    if pgn_text_stream_to_iterate:
                                        # print(f"DEBUG: Prepared to iterate games from member {member.name} in TAR {path}")
                                        try: # Inner try for specific stream processing
                                            with pgn_text_stream_to_iterate: # Ensures pgn_text_stream_to_iterate is closed
                                                yield from _iter_games_from_pgn_stream(pgn_text_stream_to_iterate, stream_id_for_errors)
                                        except Exception as member_processing_e: # Catch errors from this specific member processing or _iter_games_from_pgn_stream
                                            print(f"Error processing content of member {member.name} in TAR file {path}: {member_processing_e}")
                                            # pgn_text_stream_to_iterate is closed by its `with` statement, even if an error occurs within.
                                    else:
                                        print(f"Info: Skipping member '{member.name}' in TAR '{path}' as it's not a recognized PGN or compressed PGN type (.pgn, .gz, .bz2).")
                                finally:
                                    if member_binary_stream:
                                        # Crucially, close the binary stream extracted from the tar member in all cases
                                        member_binary_stream.close()
                                        # print(f"DEBUG: Closed binary stream for TAR member {member.name}")
                            else:
                                # print(f"DEBUG: Failed to extract TAR member {member.name} or it's not a file/streamable.")
                                pass # Silently skip if extractfile returns None
                        except KeyError as ke: # tarfile can raise KeyError for certain types of invalid tar files (e.g. PAX headers it can't handle)
                            print(f"Warning: Skipping member {member.name} in TAR {path} due to KeyError (possibly unsupported TAR feature or corruption): {ke}")
                        except Exception as extraction_e:
                             print(f"Error extracting or preparing member {member.name} from TAR {path}: {extraction_e}")
                             # Ensure stream is closed if it was opened before error
                             if member_binary_stream:
                                 member_binary_stream.close()
                                 # print(f"DEBUG: Closed binary stream for TAR member {member.name} after extraction error.")
                    else:
                        # print(f"DEBUG: Skipping TAR member {member.name} as it is not a file (e.g., directory).")
                        pass # Silently skip non-file members
        except tarfile.ReadError as tar_read_err:
            print(f"Error (tarfile.ReadError) reading TAR-like file {path}: {tar_read_err}. This might indicate a corrupted or non-standard TAR file.")
        except Exception as e:
            print(f"Error reading TAR-like file {path}: {e}")
            # import traceback
            # traceback.print_exc()
            
    elif low_path.endswith(".pgn.gz"):
        try:
            with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
                yield from _iter_games_from_pgn_stream(f, path)
        except Exception as e:
            print(f"Error opening/reading gzipped PGN file {path}: {e}")
    elif low_path.endswith(".pgn.bz2"):
        try:
            with bz2.open(path, 'rt', encoding='utf-8', errors='replace') as f:
                yield from _iter_games_from_pgn_stream(f, path)
        except Exception as e:
            print(f"Error opening/reading bzipped PGN file {path}: {e}")
    else:
        print(f"Error: Unsupported file type: {path}.")