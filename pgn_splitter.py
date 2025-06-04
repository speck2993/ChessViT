"""pgn_splitter.py
=================
Fast, parallel PGN sharder for preprocessing.

Features
--------
* Accepts **.pgn**, **.pgn.gz**, **.pgn.bz2**, **.tar(.gz/.bz2)** containing PGNs.
* Splits every source file into *N‑game* shards (default = 2 500).
* Preserves *lichess* / *lc0* indicator in output filename.
* Optional quality filter for Lichess games: Elo ≥ 1800, Termination=="Normal",
  Event matches Rated *(Classical/Blitz)* (skip otherwise).
* Uses a task‑queue with `multiprocessing.Pool` so **any** number of workers is
  safe regardless of file count.
* Per‑worker memory ≈ < 50 MB (stores ≤ games_per_chunk RAM).
* Live progress printed every shard flush.

Example
-------
```bash
python pgn_splitter.py \
       --in-dir  raw_pgns \
       --out-dir shards \
       --games-per-chunk 2500 \
       --workers 8
```
"""
from __future__ import annotations

import argparse, os, sys, gzip, bz2, tarfile, queue, time, logging, re, io, hashlib, random
from pathlib import Path
from multiprocessing import Pool, Manager, current_process
import chess.pgn  # type: ignore

# ───────────────────────────── Config & helpers ────────────────────────────────
GAME_FILTER_RE = re.compile(r"Rated (?:Classical|Blitz) game", re.I)
LICHESS_RE     = re.compile(r"lichess", re.I)


def is_good_lichess(game: chess.pgn.Game) -> bool:
    """Apply 1800+ Elo, Normal termination, rated event filter."""
    try:
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))
        if min(white_elo, black_elo) < 1800:
            return False
        if game.headers.get("Termination", "") != "Normal":
            return False
        if not GAME_FILTER_RE.fullmatch(game.headers.get("Event", "")):
            return False
    except Exception:
        return False
    return True


def open_pgn_stream(path: Path):
    """Yield (filename, text_file_like_object) for any supported PGN or archive type."""
    file_name_lower = path.name.lower()

    # Tar archives (potentially compressed)
    # Check for .tar and its compressed variants first.
    if file_name_lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
        try:
            # "r:*" auto-detects compression (gz, bz2)
            with tarfile.open(path, "r:*") as tar:
                for member in tar.getmembers():
                    # Process if it's a file and ends with .pgn (case-insensitive)
                    if member.isfile() and member.name.lower().endswith(".pgn"):
                        binary_file = tar.extractfile(member)
                        if binary_file:  # Successfully extracted
                            try:
                                # Wrap the binary stream in a TextIOWrapper for text-mode reading
                                # The TextIOWrapper will close the binary_file when it is closed
                                text_stream = io.TextIOWrapper(binary_file, encoding="utf-8", errors="ignore")
                                yield (member.name, text_stream) # member.name is full path within tar
                            except Exception as e_inner:
                                print(f"Warning: Could not create text stream for member {member.name} in {path}: {e_inner}", file=sys.stderr)
                                # Ensure binary_file is closed if TextIOWrapper instantiation failed or was not used
                                if hasattr(binary_file, 'close'):
                                    binary_file.close()
            # After processing a tar file (successfully or not, members yielded or not),
            # we are done with this path. The 'with' statement handles tar.close().
            return
        except (tarfile.ReadError, EOFError, FileNotFoundError, IsADirectoryError) as e:
            print(f"Warning: Could not open or read TAR archive {path}: {e}", file=sys.stderr)
            return # Stop processing this path if tar opening itself fails
        except Exception as e_outer: # Catch any other unexpected errors during tar processing
            print(f"Warning: Unexpected error processing TAR archive {path}: {e_outer}", file=sys.stderr)
            return

    # Single compressed PGN files (e.g., foo.pgn.gz)
    elif file_name_lower.endswith(".pgn.gz"):
        try:
            # gzip.open in text mode ("rt") returns a TextIOBase compatible stream
            stream = gzip.open(path, "rt", encoding="utf-8", errors="ignore")
            yield (path.name, stream)
        except (gzip.BadGzipFile, EOFError, FileNotFoundError, IsADirectoryError) as e:
            print(f"Warning: Could not open or read GZIP PGN file {path}: {e}", file=sys.stderr)
        # Done with this path, whether stream was yielded or error occurred.
        return

    elif file_name_lower.endswith(".pgn.bz2"):
        try:
            # bz2.open in text mode ("rt") returns a TextIOBase compatible stream
            stream = bz2.open(path, "rt", encoding="utf-8", errors="ignore")
            yield (path.name, stream)
        except (OSError, EOFError, FileNotFoundError, IsADirectoryError) as e: # bz2 can raise OSError for bad file format
            print(f"Warning: Could not open or read BZ2 PGN file {path}: {e}", file=sys.stderr)
        # Done with this path.
        return

    # Plain PGN files (e.g., foo.pgn)
    elif file_name_lower.endswith(".pgn"):
        try:
            stream = open(path, "rt", encoding="utf-8", errors="ignore")
            yield (path.name, stream)
        except (FileNotFoundError, IsADirectoryError, OSError) as e:
            print(f"Warning: Could not open plain PGN file {path}: {e}", file=sys.stderr)
        # Done with this path.
        return


# ───────────────────────────── Worker routine ─────────────────────────────────

def worker(job_q, out_dir: str, games_per_chunk: int, stats_dict, 
           test_suffixes_set: set[str], test_set_hash_chars: int, is_test_split_active: bool):
    out_base = Path(out_dir)
    pid = current_process()._identity[0] if current_process()._identity else 0
    
    # Per-worker, per-split-type chunk ID and line buffers
    chunk_id_train = 0
    chunk_id_test = 0
    out_lines_train: list[str] = []
    out_lines_test: list[str] = []
    gcount_train = 0
    gcount_test = 0

    while True:
        try:
            src_path_str = job_q.get_nowait()
        except queue.Empty:
            break

        src_path = Path(src_path_str)
        is_lichess_file_label = bool(LICHESS_RE.search(src_path.name))
        label = "lichess" if is_lichess_file_label else "lc0"

        for inner_name, stream in open_pgn_stream(src_path):
            # inner_name from open_pgn_stream could be different from src_path.name if it's a tar member
            # For the Lichess/LC0 label, src_path.name (the original file/archive) is more indicative.
            # For filtering (is_good_lichess), the game headers are what matter.

            for game in iter(lambda: chess.pgn.read_game(stream), None):
                if is_lichess_file_label and not is_good_lichess(game):
                    continue

                game_pgn_str = str(game) + "\n\n" # Ensure double newline for PGN standard
                current_data_split_type = "train"

                if is_test_split_active and test_suffixes_set: # Only hash if test splitting is on
                    game_hash = hashlib.sha1(game_pgn_str.encode('utf-8', errors='ignore')).hexdigest()
                    suffix = game_hash[-test_set_hash_chars:]
                    if suffix in test_suffixes_set:
                        current_data_split_type = "test"
                
                stats_dict["total_games_processed"] += 1 # Increment for overall progress monitoring

                if current_data_split_type == "train":
                    out_lines_train.append(game_pgn_str)
                    gcount_train += 1
                    if gcount_train >= games_per_chunk:
                        write_chunk(out_base, label, pid, chunk_id_train, out_lines_train, "train")
                        stats_dict[label + "_train"] += gcount_train
                        chunk_id_train += 1
                        gcount_train = 0
                        out_lines_train.clear()
                else: # Test game
                    out_lines_test.append(game_pgn_str)
                    gcount_test += 1
                    if gcount_test >= games_per_chunk:
                        write_chunk(out_base, label, pid, chunk_id_test, out_lines_test, "test")
                        stats_dict[label + "_test"] += gcount_test
                        chunk_id_test += 1
                        gcount_test = 0
                        out_lines_test.clear()

            if hasattr(stream, "close"):
                stream.close()
        
        # Flush remaining games from this source file for both splits
        if gcount_train > 0:
            write_chunk(out_base, label, pid, chunk_id_train, out_lines_train, "train")
            stats_dict[label + "_train"] += gcount_train
            chunk_id_train += 1
            gcount_train = 0 # Reset counter
            out_lines_train.clear()
        
        if gcount_test > 0:
            write_chunk(out_base, label, pid, chunk_id_test, out_lines_test, "test")
            stats_dict[label + "_test"] += gcount_test
            chunk_id_test += 1
            gcount_test = 0 # Reset counter
            out_lines_test.clear()
            
        print(f"[W{pid}] Finished {src_path}")


def write_chunk(base_output_dir: Path, label: str, worker_id: int, chunk_idx: int, lines: list[str], data_split_type: str):
    # Construct path: base_output_dir / data_split_type / label / chunk_...
    # e.g., shards/train/lichess/chunk_lichess_00_000000.pgn
    target_dir = base_output_dir / data_split_type / label
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Filename itself does not need data_split_type, its location determines it.
    file_name = target_dir / f"chunk_{label}_{worker_id:02d}_{chunk_idx:06d}.pgn"
    
    with open(file_name, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[W{worker_id}] wrote {file_name} ({len(lines)} games to {data_split_type}/{label}) ")


# ──────────────────────────── Progress monitoring ───────────────────────────────

def progress_monitor(stats_dict, stop_event, report_interval=10):
    """Monitor processing progress and print games/second periodically."""
    prev_total_processed = 0 # Use the new total_games_processed stat
    prev_time = time.time()
    
    while True:
        # Wait for the event to be set or for the timeout
        # If event is set during wait, wait() returns True.
        # If timeout occurs, wait() returns False.
        if stop_event.wait(timeout=report_interval):
            break  # Event was set, exit loop

        # Timeout occurred, event not set, so do the work
        current_time = time.time()
        current_total_processed = stats_dict.get("total_games_processed", 0)
        
        # Gather current train/test counts for detailed logging
        lichess_train_count = stats_dict.get("lichess_train", 0)
        lc0_train_count = stats_dict.get("lc0_train", 0)
        lichess_test_count = stats_dict.get("lichess_test", 0)
        lc0_test_count = stats_dict.get("lc0_test", 0)
        total_train_current = lichess_train_count + lc0_train_count
        total_test_current = lichess_test_count + lc0_test_count
        
        if current_total_processed > prev_total_processed:
            elapsed = current_time - prev_time
            games_processed_interval = current_total_processed - prev_total_processed
            rate = games_processed_interval / elapsed if elapsed > 0 else 0
            
            print(f"[PROGRESS] Total Processed: {current_total_processed:,} | Rate: {rate:,.0f} games/sec | "
                  f"Train (L:{lichess_train_count:,}, LC0:{lc0_train_count:,}) = {total_train_current:,} | "
                  f"Test (L:{lichess_test_count:,}, LC0:{lc0_test_count:,}) = {total_test_current:,}")
            
            prev_total_processed = current_total_processed
            prev_time = current_time
        elif current_total_processed > 0: # No new games, but some have been processed
            print(f"[PROGRESS] Total Processed: {current_total_processed:,} | Processing... | "
                  f"Train (L:{lichess_train_count:,}, LC0:{lc0_train_count:,}) = {total_train_current:,} | "
                  f"Test (L:{lichess_test_count:,}, LC0:{lc0_test_count:,}) = {total_test_current:,}")
    print("[PROGRESS_MONITOR] Stop signal received, exiting.")


# ──────────────────────────── Main orchestrator ───────────────────────────────

def collect_source_files(in_dir: Path):
    exts = (".pgn", ".pgn.gz", ".pgn.bz2", ".tar", ".tar.gz", ".tar.bz2", ".tgz", ".tbz2")
    return [str(p) for p in in_dir.rglob("*") if p.suffix.lower() in exts or any(str(p).lower().endswith(x) for x in exts)]


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in-dir", default="raw_data", help="Directory with raw PGN / tar files")
    ap.add_argument("--out-dir", default="shards", help="Output directory for shards. Will contain \'train\' and \'test\' subdirs.")
    ap.add_argument("--games-per-chunk", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    
    # Test set arguments (mirrored from preprocess_tensors.py logic)
    ap.add_argument("--test-fraction", type=float, default=0.0,
                    help="Fraction of games to allocate to the test set (e.g., 0.1 for 10%). "
                         "If 0, no test set is created. Default: 0.0")
    ap.add_argument("--test-set-seed", type=int, default=42,
                    help="RNG seed for selecting games for the test set, ensuring reproducibility. Default: 42")
    ap.add_argument("--test-set-hash-chars", type=int, default=2, choices=[2, 3, 4],
                    help="Number of trailing hex characters from SHA1 hash of PGN game string to use for train/test split. "
                         "2 chars = 256 buckets, 3 chars = 4096, 4 chars = 65536. Default: 2")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    # Base output directory will be created by write_chunk as needed (e.g., out_dir/train/label)
    # out_dir.mkdir(parents=True, exist_ok=True) # No longer creating main out_dir here

    files = collect_source_files(in_dir)
    if not files:
        print("No PGN-compatible files found.")
        sys.exit(1)

    # Generate test set suffixes if test_fraction > 0
    selected_test_suffixes_set = set()
    if args.test_fraction > 0:
        if not (0 < args.test_fraction < 1):
            print("Error: --test_fraction must be between 0 (exclusive, if creating test set) and 1. Disabling test set.", file=sys.stderr)
            args.test_fraction = 0.0
        else:
            num_chars = args.test_set_hash_chars
            possible_suffixes = [f"{i:0{num_chars}x}" for i in range(16**num_chars)]
            num_to_select = int(round(len(possible_suffixes) * args.test_fraction))
            
            if num_to_select == 0 and args.test_fraction > 0:
                num_to_select = 1 # Select at least one suffix
            
            if num_to_select > len(possible_suffixes):
                print(f"Warning: Requested test_fraction {args.test_fraction} results in selecting all {len(possible_suffixes)} "
                                f"possible suffixes for {num_chars} hash characters. Test set will effectively be the entire dataset.", file=sys.stderr)
                selected_test_suffixes_set = set(possible_suffixes)
            elif num_to_select > 0:
                rng_test_set = random.Random(args.test_set_seed)
                selected_test_suffixes_list = rng_test_set.sample(possible_suffixes, num_to_select)
                selected_test_suffixes_set = set(selected_test_suffixes_list)
                print(f"Info: Selected {len(selected_test_suffixes_set)} suffixes for the test set using {num_chars} hash chars "
                             f"(approx. {args.test_fraction*100:.2f}% of games). Seed: {args.test_set_seed}")
            else:
                print("Info: Test fraction is too small to select any suffixes, no test set will be generated.")
                args.test_fraction = 0.0


    print(f"Found {len(files)} source files. Spawning {args.workers} workers …")
    manager = Manager()
    q = manager.Queue()
    # Update stats to track train/test separately
    stats = manager.dict(lichess_train=0, lc0_train=0, lichess_test=0, lc0_test=0, total_games_processed=0)
    stop_monitor_event = manager.Event() # Create event for stopping the monitor
    for p in files:
        q.put(p)

    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        # Pass the stop_monitor_event to progress_monitor
        monitor_async_result = pool.apply_async(progress_monitor, (stats, stop_monitor_event))
        
        worker_results = []
        for _ in range(args.workers):
            # Pass test split params to worker
            result = pool.apply_async(worker, (q, str(out_dir), args.games_per_chunk, stats, 
                                               selected_test_suffixes_set, args.test_set_hash_chars, args.test_fraction > 0))
            worker_results.append(result)
        
        for result in worker_results:
            result.wait()
        
        stop_monitor_event.set() # Signal the progress_monitor to stop
        # monitor_async_result.get() # Optionally wait for monitor to finish, good for catching exceptions from it
        pool.close()
        pool.join()
    dt = time.time() - t0

    total_train = stats["lichess_train"] + stats["lc0_train"]
    total_test = stats["lichess_test"] + stats["lc0_test"]
    total_games = total_train + total_test
    
    print(f"Done. Total Games: {total_games:,}")
    print(f"  Train Games: {total_train:,} (Lichess: {stats['lichess_train']:,}, Lc0: {stats['lc0_train']:,})")
    if args.test_fraction > 0:
        print(f"  Test Games:  {total_test:,} (Lichess: {stats['lichess_test']:,}, Lc0: {stats['lc0_test']:,})")

    # Count actual shards written
    num_train_shards = len(list(out_dir.glob('train/*/*.pgn')))
    num_test_shards = len(list(out_dir.glob('test/*/*.pgn')))
    total_shards = num_train_shards + num_test_shards

    print(f"Shards written: {total_shards:,} (Train: {num_train_shards:,}, Test: {num_test_shards:,}) in {dt/60:.1f} min")
    if dt > 0 and total_games > 0:
        print(f"Processing speed ≈ {total_games/dt:,.0f} games/s")


if __name__ == "__main__":
    main()