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

import argparse, os, sys, gzip, bz2, tarfile, queue, time, logging, re, io
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

def worker(job_q, out_dir: str, games_per_chunk: int, stats_dict):
    out_base = Path(out_dir)
    pid = current_process()._identity[0] if current_process()._identity else 0
    chunk_id = 0

    while True:
        try:
            src_path = job_q.get_nowait()
        except queue.Empty:
            break

        src_path = Path(src_path)
        is_lichess_file = bool(LICHESS_RE.search(src_path.name))
        label = "lichess" if is_lichess_file else "lc0"

        # Initialize accumulation variables for this source file
        gcount = 0
        out_lines = []
        
        for inner_name, stream in open_pgn_stream(src_path):
            for game in iter(lambda: chess.pgn.read_game(stream), None):
                if is_lichess_file and not is_good_lichess(game):
                    continue
                out_lines.append(str(game) + "\n\n")
                gcount += 1
                if gcount >= games_per_chunk:
                    write_chunk(out_base, label, pid, chunk_id, out_lines)
                    stats_dict[label] += gcount
                    chunk_id += 1
                    gcount = 0
                    out_lines.clear()
            if hasattr(stream, "close"):
                stream.close()
        
        # Write any remaining games from this source file
        if gcount > 0:
            write_chunk(out_base, label, pid, chunk_id, out_lines)
            stats_dict[label] += gcount
            chunk_id += 1
            out_lines.clear()
            
        print(f"[W{pid}] Finished {src_path}")


def write_chunk(base: Path, label: str, wid: int, cid: int, lines):
    dst = base / label
    dst.mkdir(parents=True, exist_ok=True)
    name = dst / f"chunk_{label}_{wid:02d}_{cid:06d}.pgn"
    with open(name, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[W{wid}] wrote {name} ({len(lines)} games)")


# ──────────────────────────── Progress monitoring ───────────────────────────────

def progress_monitor(stats_dict, report_interval=10):
    """Monitor processing progress and print games/second periodically."""
    prev_total = 0
    prev_time = time.time()
    
    while True:
        time.sleep(report_interval)
        current_time = time.time()
        current_total = stats_dict.get("lichess", 0) + stats_dict.get("lc0", 0)
        
        if current_total > prev_total:
            elapsed = current_time - prev_time
            games_processed = current_total - prev_total
            rate = games_processed / elapsed if elapsed > 0 else 0
            
            print(f"[PROGRESS] Total games: {current_total:,} | Rate: {rate:,.0f} games/sec | "
                  f"Lichess: {stats_dict.get('lichess', 0):,} | LC0: {stats_dict.get('lc0', 0):,}")
            
            prev_total = current_total
            prev_time = current_time
        elif current_total == prev_total and current_total > 0:
            # No progress made, might be finishing up
            print(f"[PROGRESS] Total games: {current_total:,} | Processing...")


# ──────────────────────────── Main orchestrator ───────────────────────────────

def collect_source_files(in_dir: Path):
    exts = (".pgn", ".pgn.gz", ".pgn.bz2", ".tar", ".tar.gz", ".tar.bz2", ".tgz", ".tbz2")
    return [str(p) for p in in_dir.rglob("*") if p.suffix.lower() in exts or any(str(p).lower().endswith(x) for x in exts)]


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in-dir", default="raw_data", help="Directory with raw PGN / tar files")
    ap.add_argument("--out-dir", default="shards", help="Output directory for shards")
    ap.add_argument("--games-per-chunk", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_source_files(in_dir)
    if not files:
        print("No PGN-compatible files found.")
        sys.exit(1)

    print(f"Found {len(files)} source files. Spawning {args.workers} workers …")
    manager = Manager()
    q = manager.Queue()
    stats = manager.dict(lichess=0, lc0=0)
    for p in files:
        q.put(p)

    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        # Start progress monitor in a separate process
        monitor_process = pool.apply_async(progress_monitor, (stats,))
        
        # Start worker processes
        worker_results = []
        for _ in range(args.workers):
            result = pool.apply_async(worker, (q, str(out_dir), args.games_per_chunk, stats))
            worker_results.append(result)
        
        # Wait for all workers to complete
        for result in worker_results:
            result.wait()
        
        # Terminate the monitor process
        monitor_process.terminate()
        pool.close()
        pool.join()
    dt = time.time() - t0

    total = stats["lichess"] + stats["lc0"]
    print(f"Done. Lichess games: {stats['lichess']:,}  Lc0 games: {stats['lc0']:,}")
    print(f"Shards written: {len(list(out_dir.rglob('*.pgn'))):,}  in {dt/60:.1f} min ≈ {total/dt:,.0f} games/s")


if __name__ == "__main__":
    main()