"""preprocess_v3.py
=================================
Merged preprocessing pipeline ("best‑of‑both‑worlds") for converting raw PGN/ARCHIVE
files into fixed‑size *.npz* tensor chunks for training **TensorChessDataset**.

Key design decisions
--------------------
*   Per‑worker immediate flush (script #1) → predictable ≈400 MB RAM/worker.
*   Chunk size default 16 384 positions (adjustable).
*   `ProcessPoolExecutor` with *spawn* start method for cross‑platform safety.
*   Deterministic multi‑pass mix–shuffle (script #2) done **incrementally** every
    *mix_freq* new chunks to avoid a huge final I/O spike.
*   Same numpy dtypes map as original code; no flips/pre‑augmentation written –
    flips are done at train‑time.
*   Full structured logging + metadata.json manifest.

Usage example
-------------
```bash
python preprocess_v3.py \
       --raw-dir   /data/raw_pgns \
       --out-dir   /data/tensors   \
       --chunk-size 16384          \
       --jobs       48             \
       --mix-passes 3              \
       --mix-freq   100            \
       --log-level  INFO
```
"""

from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────────
# Standard lib
# ────────────────────────────────────────────────────────────────────────────────
import argparse, os, sys, time, random, json, shutil, tempfile, logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import threading

# Third‑party
import numpy as np
# python‑chess is imported indirectly by stream_worker / stream_reader

# Local helpers (provided in repo)
from stream_worker import stream_samples  # type: ignore

# ────────────────────────────────────────────────────────────────────────────────
# Constants & type maps
# ────────────────────────────────────────────────────────────────────────────────
LICHESS_IDENTIFIER = "lichess"
DEFAULT_CHUNK_SIZE = 8_192          # ≈ 93 MB raw fp16 per buffer (good for local machines)
DEFAULT_MIX_FREQ   = 100            # shuffle every N new chunks
DEFAULT_MIX_PASSES = 3
DEFAULT_LOG_LVL    = "INFO"
DEFAULT_WORKER_BUFFER_MULTIPLIER = 4 # New default

DTYPE_MAP: dict[str, np.dtype] = {
    "bitboards"          : np.float16,
    "policy_target"      : np.float16,
    "legal_mask"         : np.bool_,
    "material_raw"       : np.int8,
    "material_category"  : np.uint8,
    "value_target"       : np.int8,
    "ply_target"         : np.uint16,
    "is_960_game"        : np.bool_,
    "is_lichess_game"    : np.bool_,
    "is_lc0_game"        : np.bool_,
    "side_to_move"       : np.bool_,
    "rep_count_ge1"      : np.bool_,
    "rep_count_ge2"      : np.bool_,
    "total_plys_in_game" : np.int16,
}
NUMPY_KEYS = tuple(DTYPE_MAP.keys())

# ────────────────────────────────────────────────────────────────────────────────
# Per‑worker helpers
# ────────────────────────────────────────────────────────────────────────────────

def _get_bytes_per_position() -> int:
    """Calculate approximate bytes per position based on DTYPE_MAP."""
    bytes_val = 0
    for k, dt_val in DTYPE_MAP.items():
        dtype = np.dtype(dt_val) # Ensure it's a dtype object
        if k == "bitboards":
            # Assumes 18 planes of 8x8. This might need to be exact based on stream_worker output.
            # stream_worker.NUM_PLANES_ORIGINAL = 18
            bytes_val += dtype.itemsize * 8 * 8 * 18
        elif k == "policy_target":
            # stream_worker.POLICY_TARGET_SIZE = 1882 (for Stockfish) or specific for engine
            # Assuming a common policy size, e.g., 1882 or a board-area based one
            # Let's use a placeholder typical for chess 19x19 go style policy_length ~4672 for chess from some models
            # This needs to be accurate. For now, a placeholder from a common chess model.
            # Example: LeelaZero UCI policy map size ~1858. Let's use 1882 as a common one.
            bytes_val += dtype.itemsize * 1882 # Placeholder, adjust if policy size differs
        elif k == "legal_mask":
            bytes_val += dtype.itemsize * 1882 # Same placeholder as policy
        else:
            bytes_val += dtype.itemsize
    return bytes_val

# Global for estimated bytes, calculated once
APPROX_BYTES_PER_POSITION = _get_bytes_per_position()

def _flush_chunk_from_buffer(src: str, chunk_size: int, buf: dict[str, list], out_base: Path, 
                            chunk_counter: dict[str, int], disable_verification: bool = False) -> int:
    """Extract exactly chunk_size positions from buffer, shuffle them, and save to disk.
    Returns number of positions actually saved (may be less than chunk_size if buffer is smaller)."""
    if not buf["bitboards"]:
        return 0  # nothing to flush

    process_id = os.getpid()
    out_base.mkdir(parents=True, exist_ok=True)
    
    # Determine how many positions to save (exactly chunk_size or whatever is left)
    available_positions = len(buf["bitboards"])
    positions_to_save = min(chunk_size, available_positions)
    
    if positions_to_save == 0:
        return 0
    
    # Create filename with worker PID and chunk counter to avoid conflicts
    fname = out_base / f"{src}_w{process_id}_c{chunk_counter[src]:07d}.npz"
    chunk_counter[src] += 1
    
    logging.info(f"_flush_chunk_from_buffer: Saving {positions_to_save} positions for {src} to {fname}")

    # Extract positions to save and shuffle them
    chunk_data = {}
    for k in buf.keys():
        chunk_data[k] = buf[k][:positions_to_save]
        # Remove saved positions from buffer
        buf[k] = buf[k][positions_to_save:]
    
    # Shuffle the chunk data
    if positions_to_save > 1:
        perm_seed = os.getpid() + chunk_counter[src] + int(time.time() * 1000)
        perm_seed = perm_seed % (2**32) 
        np_rng_shuffle = np.random.RandomState(perm_seed)
        perm = np_rng_shuffle.permutation(positions_to_save)
        
        for k in chunk_data.keys():
            if chunk_data[k] and isinstance(chunk_data[k][0], np.ndarray):
                # Handle lists of numpy arrays (like bitboards)
                try:
                    stacked = np.stack(chunk_data[k])
                    shuffled_stacked = stacked[perm]
                    chunk_data[k] = [shuffled_stacked[i] for i in range(positions_to_save)]
                except ValueError as e:
                    logging.error(f"Error stacking/shuffling list of ndarrays for key {k}: {e}")
            else:
                # Handle lists of scalars
                temp_array = np.array(chunk_data[k], dtype=object)
                temp_array = temp_array[perm]
                chunk_data[k] = list(temp_array)

    # Convert to numpy arrays for saving
    arrays: dict[str, np.ndarray] = {}
    total_bytes = 0
    for k, lst in chunk_data.items():
        want = DTYPE_MAP[k]
        if k in ("bitboards", "policy_target", "legal_mask"):
            if lst:
                arrays[k] = np.stack(lst).astype(want, copy=False)
            else:
                arrays[k] = np.array([], dtype=want).reshape(0, *DTYPE_MAP[k].shape if hasattr(DTYPE_MAP[k], 'shape') else (0,))
        else:
            arrays[k] = np.asarray(lst, dtype=want)
        total_bytes += arrays[k].nbytes
    
    try:
        np.savez_compressed(fname, **arrays)
        
        if not fname.exists():
            raise FileNotFoundError(f"NPZ file was not created: {fname}")
            
        if not disable_verification:
            try:
                test_load = np.load(fname, allow_pickle=False)
                test_load.close()
            except Exception as verify_e:
                logging.error(f"_flush_chunk_from_buffer: Verification failed for {fname}: {verify_e}")
                if fname.exists():
                    fname.unlink()
                raise verify_e
        
        logging.info(f"_flush_chunk_from_buffer: Successfully saved {fname} ({total_bytes} bytes)")
        
    except Exception as e:
        logging.error(f"_flush_chunk_from_buffer: ERROR saving {fname}: {e}")
        if fname.exists():
            fname.unlink()
        raise

    return positions_to_save


def _sample_to_lists(sample: dict, lists: dict[str, list]):
    """Convert one stream_worker sample to flat lists (no flips)."""
    lists["bitboards"         ].append(sample["bitboards_original"].astype(np.float16))
    lists["policy_target"     ].append(sample["policy"           ].astype(np.float16))
    lists["legal_mask"        ].append(sample["legal_mask"       ].astype(np.bool_))
    lists["material_raw"      ].append(np.int8(sample["material_raw"]))
    lists["material_category" ].append(np.uint8(sample["material_cat"]))
    lists["value_target"      ].append(np.int8(sample["wdl_target"]))
    lists["ply_target"        ].append(np.uint16(sample["ply_target"]))
    lists["is_960_game"       ].append(bool(sample["is960"]))

    is_lichess = bool(sample["source_flag"])
    lists["is_lichess_game"   ].append(is_lichess)
    lists["is_lc0_game"       ].append(not is_lichess)

    lists["side_to_move"      ].append(bool(sample["stm"]))
    lists["rep_count_ge1"     ].append(bool(sample["rep1"]))
    lists["rep_count_ge2"     ].append(bool(sample["rep2"]))
    lists["total_plys_in_game"].append(np.int16(sample["total_plys_in_game"]))


def _worker(file_subset: list[tuple[str, str]], job_seed: int, chunk_size: int, output_dir: str, 
            worker_buffer_multiplier: int, # New arg
            disable_verification: bool = False, progress_queue=None):
    """Executed in each subprocess with real-time progress reporting via queue."""
    logger = logging.getLogger()
    
    # Calculate effective buffer size for worker's buffer
    effective_buffer_size = chunk_size * worker_buffer_multiplier
    logger.info(f"Worker starting with {len(file_subset)} files. chunk_size={chunk_size}, worker_buffer_multiplier={worker_buffer_multiplier}, effective_buffer_size={effective_buffer_size}")

    random.seed(job_seed)
    out_base = Path(output_dir)

    for i, (file_path, source_name) in enumerate(file_subset):
        logger.info(f"  File {i}: {file_path} ({source_name})")

    # Report total files count
    if progress_queue:
        try:
            progress_queue.put({'type': 'total_files', 'count': len(file_subset)})
        except:
            pass

    # Per‑source buffers
    buf = {
        "lichess": {k: [] for k in NUMPY_KEYS},
        "lc0"    : {k: [] for k in NUMPY_KEYS},
    }
    chunk_counter = {"lichess": 0, "lc0": 0}
    pos_counter = {"lichess": 0, "lc0": 0}
    chunks_saved = {"lichess": 0, "lc0": 0}

    for file_idx, (file_path, source_name) in enumerate(file_subset):
        logger.info(f"Processing file {file_idx+1}/{len(file_subset)}: {file_path} ({source_name})")
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.error(f"File does not exist: {file_path}")
            continue
            
        file_size = path_obj.stat().st_size
        logger.info(f"File size: {file_size} bytes")
        
        games_from_file = 0
        positions_from_file = 0
        positions_since_last_update = {"lichess": 0, "lc0": 0}
        
        # Determine source override based on directory structure
        is_lichess_source = (source_name == "lichess")
        
        try:
            for game_samples in stream_samples([file_path], flips=False, is_lichess_source_override=is_lichess_source):
                games_from_file += 1
                positions_from_file += len(game_samples)
                
                for s in game_samples:
                    src = "lichess" if s["source_flag"] else "lc0"
                    _sample_to_lists(s, buf[src])
                    pos_counter[src] += 1
                    positions_since_last_update[src] += 1
                    
                    # Report progress every 100 positions per source
                    if positions_since_last_update[src] >= 100 and progress_queue:
                        try:
                            progress_queue.put({'type': 'position', 'source': src, 'count': positions_since_last_update[src]})
                            positions_since_last_update[src] = 0
                        except:
                            pass
                    
                    # Check if we can save a full chunk
                    if len(buf[src]["bitboards"]) >= chunk_size:
                        saved_positions = _flush_chunk_from_buffer(src, chunk_size, buf[src], out_base / src, chunk_counter, disable_verification)
                        if saved_positions > 0:
                            chunks_saved[src] += 1
                            # Report chunk completion
                            if progress_queue:
                                try:
                                    progress_queue.put({'type': 'chunk', 'source': src})
                                except:
                                    pass
            
            # Report any remaining positions
            for src in ["lichess", "lc0"]:
                if positions_since_last_update[src] > 0 and progress_queue:
                    try:
                        progress_queue.put({'type': 'position', 'source': src, 'count': positions_since_last_update[src]})
                    except:
                        pass
            
            logger.info(f"Processed {games_from_file} games ({positions_from_file} positions) from file {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Report file completion
        if progress_queue:
            try:
                progress_queue.put({'type': 'file_completed'})
            except:
                pass

    # Flush remaining positions in buffer (may be less than chunk_size)
    for src in ("lichess", "lc0"):
        while buf[src]["bitboards"]:  # Keep flushing until buffer is empty
            saved_positions = _flush_chunk_from_buffer(src, chunk_size, buf[src], out_base / src, chunk_counter, disable_verification)
            if saved_positions > 0:
                chunks_saved[src] += 1
                # Report final chunk
                if progress_queue:
                    try:
                        progress_queue.put({'type': 'chunk', 'source': src})
                    except:
                        pass
            else:
                break  # No more positions to save

    logger.info(f"Worker finished. Positions: lichess={pos_counter['lichess']}, lc0={pos_counter['lc0']}")
    logger.info(f"Worker saved chunks: lichess={chunks_saved['lichess']}, lc0={chunks_saved['lc0']}")
    
    return {
        "lichess_chunks"   : chunks_saved["lichess"],
        "lc0_chunks"       : chunks_saved["lc0"],
        "lichess_positions": pos_counter["lichess"],
        "lc0_positions"    : pos_counter["lc0"],
    }


def worker_wrapper(args):
    """Wrapper function to unpack arguments for the worker."""
    return _worker(*args)

# ────────────────────────────────────────────────────────────────────────────────
# Shuffle helper (pair‑wise merge‑shuffle‑split) – identical to script #1
# ────────────────────────────────────────────────────────────────────────────────

def _verify_shuffle_integrity(original_files, shuffled_dir, verify_shuffle):
    """Verify that shuffling preserved data integrity."""
    if not verify_shuffle:
        return True
    
    logging.info("Verifying shuffle integrity...")
    
    # Load a sample from original and shuffled to check key consistency
    original_sample = None
    shuffled_sample = None
    
    for f in original_files[:1]:  # Just check first file
        if f.exists():
            try:
                original_sample = dict(np.load(f, allow_pickle=True))
                break
            except:
                continue
    
    shuffled_files = list(shuffled_dir.glob("*.npz"))
    for f in shuffled_files[:1]:  # Just check first file
        try:
            shuffled_sample = dict(np.load(f, allow_pickle=True))
            break
        except:
            continue
    
    if original_sample and shuffled_sample:
        if set(original_sample.keys()) != set(shuffled_sample.keys()):
            logging.error("Shuffle integrity check failed: key mismatch")
            return False
        
        for key in original_sample.keys():
            if original_sample[key].shape[1:] != shuffled_sample[key].shape[1:]:
                logging.error(f"Shuffle integrity check failed: shape mismatch for {key}")
                return False
        
        logging.info("✓ Shuffle integrity check passed")
    
    return True


def _estimate_shuffle_ram(num_files, avg_file_size_mb):
    """Estimate RAM usage for shuffle operations."""
    # Each file pair loads 2 files, merges them, then writes back
    # Peak usage: 2 * file_size (load) + merged_size (2 * file_size) = 4 * file_size per pair
    ram_per_pair_gb = (4 * avg_file_size_mb) / 1024
    return ram_per_pair_gb


def _simple_merge_shuffle_split_parallel(cat_dir: Path, rng_seed: int, shuffle_workers: int, 
                                         shuffle_batch_size: int, benchmark=False, verify_shuffle=False):
    """Simple parallel merge+shuffle: coordinator assigns file batches to workers."""
    if shuffle_batch_size < 2:
        logging.error(f"shuffle_batch_size must be at least 2, got {shuffle_batch_size}. Aborting shuffle for {cat_dir}.")
        return

    # Find all NPZ files in the directory
    files = sorted(list(cat_dir.glob("*.npz")))
    
    if len(files) < 2:
        logging.info(f"Not enough files ({len(files)}) in {cat_dir} for shuffling. Minimum 2 required.")
        return
    
    # Globally shuffle file order to break local effects (CRITICAL for randomness)
    rng = random.Random(rng_seed)
    rng.shuffle(files)
    logging.info(f"Globally shuffled {len(files)} files in {cat_dir} to break local data effects before batching.")
    
    # Create file batches
    file_batches = []
    for i in range(0, len(files), shuffle_batch_size):
        batch = files[i:i + shuffle_batch_size]
        if len(batch) >= 2: # Only process batches with at least 2 files
            file_batches.append(batch)
        elif batch: # Log if a small batch is skipped
            logging.info(f"Skipping a small batch of {len(batch)} file(s) as it's less than 2: {[f.name for f in batch]}")

    if not file_batches:
        logging.info(f"No suitable file batches (min size 2) created from {len(files)} files in {cat_dir} for shuffling.")
        return
    
    # Set up progress monitoring
    # total_files_in_valid_batches = sum(len(b) for b in file_batches)
    # shuffle_monitor = ShuffleProgressMonitor(total_files_in_valid_batches, "Batched Parallel Shuffle")
    num_actual_shuffle_ops = len(file_batches)
    shuffle_monitor = ShuffleProgressMonitor(num_actual_shuffle_ops, "Batched Parallel Shuffle (Ops)")

    start_time = time.time()
    logging.info(f"Batched parallel shuffling {len(file_batches)} file batches (each up to {shuffle_batch_size} files) in {cat_dir} using {shuffle_workers} workers")
    
    # Run parallel shuffling
    total_positions_processed_across_all_batches = 0
    completed_batch_ops = 0
    
    with ProcessPoolExecutor(max_workers=shuffle_workers) as pool:
        futures = [
            pool.submit(_simple_merge_shuffle_split_worker, batch, rng_seed + i, verify_shuffle)
            for i, batch in enumerate(file_batches)
        ]
        
        for future in as_completed(futures):
            try:
                result_msg, positions_in_completed_batch = future.result()
                if positions_in_completed_batch > 0: # Successful operation
                    completed_batch_ops += 1
                total_positions_processed_across_all_batches += positions_in_completed_batch
                
                # Update progress monitor based on completed batches/operations
                shuffle_monitor.update_progress(completed_batch_ops, total_positions_processed_across_all_batches)
                
                # Only log individual batch results if benchmarking is enabled
                if benchmark and result_msg:
                    logging.info(f"Shuffle batch result: {result_msg}") # Renamed variable for clarity
                    
            except Exception as e:
                logging.error(f"Shuffle batch worker future failed: {e}")
    
    elapsed = time.time() - start_time
    if benchmark and total_positions_processed_across_all_batches > 0:
        positions_per_second = total_positions_processed_across_all_batches / elapsed
        logging.info(f"Batched Shuffle benchmark for {cat_dir}: {positions_per_second:.0f} positions/second "
                    f"({total_positions_processed_across_all_batches} positions from {completed_batch_ops} batch ops in {elapsed:.1f}s)")


def _simple_merge_shuffle_split_worker(file_batch: list[Path], rng_seed: int, verify_shuffle: bool = False):
    """Worker function: merge a batch of files, shuffle, split back to original files."""
    if len(file_batch) < 2:
        return "Not enough files in batch to shuffle", 0

    batch_names = [f.name for f in file_batch]
    try:
        start_time = time.time()

        # Load all files in the batch
        all_data_in_batch: list[dict[str, np.ndarray]] = []
        for file_path in file_batch:
            all_data_in_batch.append(dict(np.load(file_path, allow_pickle=False)))

        if not all_data_in_batch:
            return f"No data loaded for batch {batch_names}", 0

        # Verify keys match across all files in the batch
        first_file_keys = set(all_data_in_batch[0].keys())
        for i, data_dict in enumerate(all_data_in_batch[1:], 1):
            if set(data_dict.keys()) != first_file_keys:
                raise ValueError(f"Key mismatch in batch. File {file_batch[0].name} keys vs {file_batch[i].name} keys")

        # Calculate total positions and concatenate arrays
        total_positions = sum(len(data_dict["bitboards"]) for data_dict in all_data_in_batch)
        if total_positions == 0:
            return f"Batch {batch_names} contains no positions.", 0

        # Store original checksums for verification if requested
        original_checksums = {}
        if verify_shuffle:
            for key in first_file_keys:
                concatenated_for_checksum = np.concatenate([data[key] for data in all_data_in_batch], axis=0)
                original_checksums[key] = np.sum(concatenated_for_checksum.astype(np.float64))

        # Merge all arrays from the batch
        merged_arrays: dict[str, np.ndarray] = {}
        for key in first_file_keys:
            merged_arrays[key] = np.concatenate([data[key] for data in all_data_in_batch], axis=0)

        # Generate permutation and apply to ALL merged arrays consistently
        np_rng = np.random.RandomState(rng_seed)
        perm = np_rng.permutation(total_positions)
        for key in merged_arrays:
            merged_arrays[key] = merged_arrays[key][perm]

        # Verify checksums if requested
        if verify_shuffle:
            for key, orig_checksum in original_checksums.items():
                new_checksum = np.sum(merged_arrays[key].astype(np.float64))
                # Use a slightly more tolerant comparison for float64 sums
                if not np.isclose(orig_checksum, new_checksum, rtol=1e-05, atol=1e-08):
                    raise ValueError(f"Shuffle corrupted data in {key} for batch {batch_names}. Original sum: {orig_checksum}, new sum: {new_checksum}")

        # Split the shuffled data back into parts, matching the number of input files
        num_files_in_batch = len(file_batch)
        # Calculate roughly equal split points for positions
        # E.g., if 100 positions and 3 files, splits are [33, 33, 34]
        base_split_size = total_positions // num_files_in_batch
        remainder = total_positions % num_files_in_batch
        split_sizes = [base_split_size + 1] * remainder + [base_split_size] * (num_files_in_batch - remainder)
        
        current_pos_idx = 0
        for i in range(num_files_in_batch):
            start_idx = current_pos_idx
            end_idx = current_pos_idx + split_sizes[i]
            data_to_save = {key: merged_arrays[key][start_idx:end_idx] for key in merged_arrays}
            np.savez_compressed(file_batch[i], **data_to_save)
            current_pos_idx = end_idx

        elapsed = time.time() - start_time
        result_msg = f"Shuffled batch of {num_files_in_batch} files ({batch_names}) with {total_positions} total positions in {elapsed:.2f}s"
        return result_msg, total_positions

    except Exception as e:
        error_msg = f"ERROR shuffling batch {batch_names}: {e}"
        logging.error(error_msg)
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return error_msg, 0




# ────────────────────────────────────────────────────────────────────────────────
# Progress monitoring
# ────────────────────────────────────────────────────────────────────────────────

class ProgressMonitor:
    """Thread-safe progress monitoring using a queue for real-time updates."""
    
    def __init__(self, total_workers, update_interval=10.0, progress_queue=None, 
                 chunk_size_arg=DEFAULT_CHUNK_SIZE, 
                 worker_buffer_multiplier_arg=DEFAULT_WORKER_BUFFER_MULTIPLIER): # New args
        self.total_workers = total_workers
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.progress_queue = progress_queue
        self.chunk_size_arg = chunk_size_arg
        self.worker_buffer_multiplier_arg = worker_buffer_multiplier_arg
        
        # Calculate estimated buffer memory per worker
        self.estimated_peak_buffer_ram_mb = 0
        if self.worker_buffer_multiplier_arg > 0:
            effective_buffer_sz = self.chunk_size_arg * self.worker_buffer_multiplier_arg
            # Approx bytes per position * effective buffer size * 2 sources (lichess, lc0)
            self.estimated_peak_buffer_ram_mb = (APPROX_BYTES_PER_POSITION * effective_buffer_sz * 2) / (1024 * 1024)
        
        # Local tracking for all stats
        self.stats = {
            'lichess_positions': 0,
            'lc0_positions': 0,
            'lichess_chunks': 0,
            'lc0_chunks': 0,
            'completed_workers': 0,
            'files_processed': 0,
            'total_files': 0
        }
        self.last_stats = self.stats.copy()
        
        # Progress thread control
        self.running = False
        self.progress_thread = None
    
    def start_monitoring(self):
        """Start the progress monitoring thread."""
        self.running = True
        self.progress_thread = threading.Thread(target=self._progress_loop, daemon=True)
        self.progress_thread.start()
        logging.info("Real-time progress monitoring started (updates every %.1fs)", self.update_interval)
    
    def stop_monitoring(self):
        """Stop the progress monitoring thread."""
        self.running = False
        if self.progress_thread:
            self.progress_thread.join(timeout=2.0)
    
    def update_worker_completion(self, worker_stats):
        """Update stats when a worker completes (final update)."""
        # Only update completed workers count, not position counts
        # Position counts are already tracked via queue updates
        self.stats['completed_workers'] += 1
    
    def _process_queue_updates(self):
        """Process all available updates from the queue."""
        if not self.progress_queue:
            return
        
        try:
            while True:
                update = self.progress_queue.get_nowait()
                if update['type'] == 'position':
                    self.stats[f"{update['source']}_positions"] += update['count']
                elif update['type'] == 'chunk':
                    self.stats[f"{update['source']}_chunks"] += 1
                elif update['type'] == 'file_completed':
                    self.stats['files_processed'] += 1
                elif update['type'] == 'total_files':
                    self.stats['total_files'] += update['count']
        except:
            # Queue is empty, which is normal
            pass
    
    def _progress_loop(self):
        """Background thread that prints progress updates."""
        while self.running:
            time.sleep(self.update_interval)
            if not self.running:
                break
            
            # Process queue updates
            self._process_queue_updates()
            
            current_time = time.time()
            current_stats = self.stats.copy()
            
            # Calculate rates since last update
            time_delta = current_time - self.last_update_time
            total_positions = current_stats['lichess_positions'] + current_stats['lc0_positions']
            last_total = self.last_stats['lichess_positions'] + self.last_stats['lc0_positions']
            
            if time_delta > 0 and total_positions > last_total:
                recent_rate = (total_positions - last_total) / time_delta
                overall_rate = total_positions / (current_time - self.start_time)
                efficiency = overall_rate / self.total_workers if self.total_workers > 0 else 0
                
                # Worker progress
                worker_progress = (current_stats['completed_workers'] / self.total_workers * 100) if self.total_workers > 0 else 0
                
                # File progress (if available)
                file_progress_str = ""
                if current_stats['total_files'] > 0:
                    file_pct = (current_stats['files_processed'] / current_stats['total_files'] * 100)
                    file_progress_str = f"Files: {current_stats['files_processed']}/{current_stats['total_files']} ({file_pct:.1f}%) | "
                
                # Memory estimate string
                mem_estimate_str = f"Est. peak buffer RAM/worker: {self.estimated_peak_buffer_ram_mb:.1f} MB | " if self.estimated_peak_buffer_ram_mb > 0 else ""

                logging.info(
                    f"PROGRESS: {current_stats['completed_workers']}/{self.total_workers} workers ({worker_progress:.1f}%) | "
                    f"{file_progress_str}"
                    f"Positions: {total_positions:,} total "
                    f"(L:{current_stats['lichess_positions']:,}, LC0:{current_stats['lc0_positions']:,}) | "
                    f"Rate: {recent_rate:.0f}/s recent, {overall_rate:.0f}/s avg | "
                    f"{mem_estimate_str}"
                    f"Efficiency: {efficiency:.0f} pos/s/worker | "
                    f"Chunks (buffers): L:{current_stats['lichess_chunks']}, LC0:{current_stats['lc0_chunks']}"
                )
            
            self.last_update_time = current_time
            self.last_stats = current_stats
    
    def get_final_stats(self):
        """Get final statistics."""
        # Process any remaining queue updates
        self._process_queue_updates()
        return self.stats.copy()


class ShuffleProgressMonitor:
    """Progress monitoring specifically for shuffle operations."""
    
    def __init__(self, total_operations: int, operation_name="Shuffling"):
        self.total_operations = total_operations # Renamed from total_files
        self.operation_name = operation_name
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.completed_ops = 0 # Renamed from completed_pairs
        self.total_positions = 0
        self.last_update_time = self.start_time
        self.last_completed_ops = 0 # Renamed from last_completed
        self.last_positions = 0
    
    def update_progress(self, completed_ops: int, positions_processed: int):
        """Update shuffle progress."""
        with self.lock:
            self.completed_ops = completed_ops
            self.total_positions = positions_processed
            
            current_time = time.time()
            time_delta = current_time - self.last_update_time
            
            if time_delta >= 5.0:  # Update every 5 seconds for shuffle
                ops_delta = completed_ops - self.last_completed_ops
                positions_delta = positions_processed - self.last_positions
                
                if time_delta > 0 and self.total_operations > 0:
                    op_rate = ops_delta / time_delta
                    position_rate = positions_delta / time_delta
                    
                    progress_pct = (completed_ops / self.total_operations * 100)
                    
                    logging.info(
                        f"{self.operation_name} PROGRESS: {completed_ops}/{self.total_operations} ops ({progress_pct:.1f}%) | "
                        f"Rate: {op_rate:.1f} ops/s, {position_rate:.0f} pos/s | "
                        f"Total positions: {positions_processed:,}"
                    )
                
                self.last_update_time = current_time
                self.last_completed_ops = completed_ops
                self.last_positions = positions_processed

# ────────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ────────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--raw-dir",  required=True, help="Directory with PGN / archives")
    ap.add_argument("--out-dir",  required=True, help="Output dir for tensors")
    ap.add_argument("--chunk-size",   type=int, default=DEFAULT_CHUNK_SIZE,
                    help="Positions per .npz chunk")
    ap.add_argument("--jobs",         type=int, default=mp.cpu_count(),
                    help="Parallel worker processes")
    ap.add_argument("--mix-passes",   type=int, default=DEFAULT_MIX_PASSES,
                    help="Shuffle passes (merge‑shuffle‑split)")
    ap.add_argument("--mix-freq",    type=int, default=DEFAULT_MIX_FREQ,
                    help="Run shuffler every N new chunks (per source)")
    ap.add_argument("--seed",        type=int, default=2024,
                    help="Global RNG seed")
    ap.add_argument("--log-level",   type=str, default=DEFAULT_LOG_LVL,
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging detail")
    
    # High-performance cluster options
    ap.add_argument("--shuffle-workers", type=int, default=None,
                    help="Parallel workers for shuffling (default: min(8, jobs//4))")
    ap.add_argument("--shuffle-batch-size", type=int, default=2,
                    help="Number of files each shuffle worker merges and splits at a time (min: 2).")
    ap.add_argument("--large-chunks", action="store_true",
                    help="Use larger chunk sizes for fewer files (good for high-core runs)")
    ap.add_argument("--disable-verification", action="store_true",
                    help="Skip NPZ verification to speed up I/O (use with caution)")
    
    # New arguments for in-worker buffered shuffling
    ap.add_argument("--worker-buffer-multiplier", type=int, default=DEFAULT_WORKER_BUFFER_MULTIPLIER,
                    help="Multiplier for chunk_size to determine worker's internal buffer size before shuffling and flushing. "
                         "Higher values increase RAM per worker but reduce disk I/O. (Default: %(default)s)")
    ap.add_argument("--skip-final-shuffle-pass", action="store_true",
                    help="If set, skips the final disk-based merge-shuffle pass. "
                         "Worker-generated (and internally shuffled) buffer files will be collected and renamed sequentially. "
                         "Recommended if worker_buffer_multiplier is high.")
    
    # Benchmarking and debugging options
    ap.add_argument("--benchmark", action="store_true",
                    help="Enable detailed timing and throughput benchmarking")
    ap.add_argument("--verify-shuffle", action="store_true",
                    help="Verify data integrity after shuffling (slow but safe)")
    
    return ap.parse_args()


def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(processName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    args = parse_args()
    setup_logging(args.log_level)

    # Cluster optimizations
    if args.large_chunks:
        # For high-core runs, use larger chunks to reduce file count
        chunk_size = max(args.chunk_size, 65536)  # At least 64K positions
        logging.info(f"Large chunks mode: Using chunk size {chunk_size} instead of {args.chunk_size}")
    else:
        chunk_size = args.chunk_size
    
    # Configure shuffle workers for parallel shuffling
    if args.shuffle_workers is None:
        shuffle_workers = min(8, max(1, args.jobs // 4))
    else:
        shuffle_workers = args.shuffle_workers
    
    logging.info(f"Cluster config: jobs={args.jobs}, shuffle_workers={shuffle_workers}, "
                f"disable_verification={args.disable_verification}")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"DEBUG: preprocess_tensors.py received raw_dir: {raw_dir.resolve()}")
    logging.info(f"DEBUG: raw_dir exists: {raw_dir.exists()}")

    # Log initial contents of raw_dir seen by preprocess_tensors.py
    logging.info(f"DEBUG: Initial raw scan of {raw_dir} before collecting PGNs:")
    if raw_dir.exists():
        for item in raw_dir.rglob("*"):
            logging.info(f"  DEBUG_SCAN: Found item: {item} (is_file: {item.is_file()})")
    else:
        logging.info(f"  DEBUG_SCAN: {raw_dir} does not exist at scan time.")

    # Collect PGN / archive paths from subdirectories
    exts = (".pgn", ".pgn.gz", ".pgn.bz2", ".tar", ".tar.gz", ".tar.bz2", ".tgz", ".tbz2")
    
    # Create separate lists for each source
    lc0_files_with_source: list[tuple[str, str]] = []
    lichess_files_with_source: list[tuple[str, str]] = []

    # Process lc0 files
    source_name_lc0 = "lc0"
    source_dir_lc0 = raw_dir / source_name_lc0
    logging.info(f"Searching for {source_name_lc0} files in: {source_dir_lc0}")
    if source_dir_lc0.exists():
        current_lc0_paths = []
        for item in source_dir_lc0.rglob("*"):
            if (item.is_file() and 
                (item.suffix.lower() in exts or any(str(item).lower().endswith(x) for x in exts))):
                current_lc0_paths.append(str(item))
                lc0_files_with_source.append((str(item), source_name_lc0))
        logging.info(f"Found {len(current_lc0_paths)} {source_name_lc0} files. Shuffling them...")
        random.Random(args.seed).shuffle(lc0_files_with_source)
    else:
        logging.warning(f"Source directory {source_dir_lc0} does not exist, skipping {source_name_lc0} files.")

    # Process lichess files
    source_name_lichess = "lichess"
    source_dir_lichess = raw_dir / source_name_lichess
    logging.info(f"Searching for {source_name_lichess} files in: {source_dir_lichess}")
    if source_dir_lichess.exists():
        current_lichess_paths = []
        for item in source_dir_lichess.rglob("*"):
            if (item.is_file() and 
                (item.suffix.lower() in exts or any(str(item).lower().endswith(x) for x in exts))):
                current_lichess_paths.append(str(item))
                lichess_files_with_source.append((str(item), source_name_lichess))
        logging.info(f"Found {len(current_lichess_paths)} {source_name_lichess} files. Shuffling them...")
        random.Random(args.seed + 1).shuffle(lichess_files_with_source) # Use a different seed for independence
    else:
        logging.warning(f"Source directory {source_dir_lichess} does not exist, skipping {source_name_lichess} files.")

    if not lc0_files_with_source and not lichess_files_with_source:
        logging.error("No PGN / archive files found in %s/lc0 or %s/lichess", raw_dir, raw_dir)
        sys.exit(1)

    # Distribute each source's files round-robin into per-source worker buckets
    lc0_worker_assignments: list[list[tuple[str, str]]] = [[] for _ in range(args.jobs)]
    for idx, file_tuple in enumerate(lc0_files_with_source):
        lc0_worker_assignments[idx % args.jobs].append(file_tuple)

    lichess_worker_assignments: list[list[tuple[str, str]]] = [[] for _ in range(args.jobs)]
    # Distribute Lichess files in reverse worker order for better load balancing
    if args.jobs > 0: # Ensure args.jobs is positive to prevent potential issues with modulo or negative indexing
        for idx, file_tuple in enumerate(lichess_files_with_source):
            target_worker_idx = (args.jobs - 1) - (idx % args.jobs)
            lichess_worker_assignments[target_worker_idx].append(file_tuple)
    else: # Should ideally not happen as jobs usually > 0, but handle defensively
        if lichess_files_with_source: # If there are lichess files but no jobs, log a warning
            logging.warning("args.jobs is 0 or less, cannot assign Lichess files. They will be skipped.")

    # Combine assignments: each worker gets their lc0 files, then their lichess files
    final_buckets: list[list[tuple[str, str]]] = [[] for _ in range(args.jobs)]
    for i in range(args.jobs):
        final_buckets[i].extend(lc0_worker_assignments[i])
        final_buckets[i].extend(lichess_worker_assignments[i])

    logging.info(f"Distributed files to {args.jobs} workers. Each worker processes Lc0 files then Lichess files.")
    total_files_assigned = 0
    for i, bucket_content in enumerate(final_buckets):
        if bucket_content: # Only log if bucket is not empty
            lc0_count = sum(1 for _, src in bucket_content if src == source_name_lc0)
            lichess_count = sum(1 for _, src in bucket_content if src == source_name_lichess)
            logging.info(f"  Worker {i}: {len(bucket_content)} total files ({lc0_count} Lc0, {lichess_count} Lichess). Buffer multiplier: {args.worker_buffer_multiplier}, effective buffer size: {chunk_size * args.worker_buffer_multiplier} positions")
            # Optional: log actual file names if needed for debugging, can be verbose
            # for file_path, source_name_in_bucket in bucket_content:
            #     logging.debug(f"    Worker {i} assigned: {Path(file_path).name} ({source_name_in_bucket})")
            total_files_assigned += len(bucket_content)
        else:
            logging.info(f"  Worker {i}: 0 files assigned.")
    
    # Verify all files were assigned (sanity check)
    expected_total_files = len(lc0_files_with_source) + len(lichess_files_with_source)
    if total_files_assigned != expected_total_files:
        logging.warning(f"File assignment mismatch: Expected {expected_total_files} files, but assigned {total_files_assigned} to workers.")

    # Create progress queue for real-time monitoring
    manager = mp.Manager()
    progress_queue = manager.Queue()

    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
        futures = [
            pool.submit(worker_wrapper, (
                bucket, 
                args.seed + i, 
                chunk_size, # Original chunk_size
                str(out_dir),
                args.worker_buffer_multiplier, # New arg
                args.disable_verification, 
                progress_queue
            ))
            for i, bucket in enumerate(final_buckets) if bucket
        ]
        
        logging.info(f"Submitted {len(futures)} worker tasks")

        # Set up progress monitoring with progress queue
        progress_monitor = ProgressMonitor(
            len(futures), 
            update_interval=10.0, 
            progress_queue=progress_queue,
            chunk_size_arg=chunk_size, # Pass for RAM estimation
            worker_buffer_multiplier_arg=args.worker_buffer_multiplier # Pass for RAM estimation
        )
        progress_monitor.start_monitoring()

        # Collect results from completed futures
        total_stats = defaultdict(int)
        start_time = time.time()
        
        # Wait for all futures to complete and collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                
                # Update progress monitor (final completion)
                progress_monitor.update_worker_completion(result)
                
                for k, v in result.items():
                    total_stats[k] += v
                    
                # Log completion (less verbose now that we have progress monitor)
                if args.benchmark:
                    elapsed = time.time() - start_time
                    total_positions = total_stats['lichess_positions'] + total_stats['lc0_positions']
                    if elapsed > 0:
                        positions_per_second = total_positions / elapsed
                        logging.info(f"Worker completed. Final totals: {total_positions:,} positions ({positions_per_second:.0f} pos/sec avg)")
                           
            except Exception as e:
                logging.error(f"Worker failed with exception: {e}")
                # Still update progress monitor for failed worker
                progress_monitor.update_worker_completion({})

        # Stop progress monitoring
        progress_monitor.stop_monitoring()
        
        # Get final stats from monitor
        final_monitor_stats = progress_monitor.get_final_stats()
        logging.info(f"Final progress monitor stats (real-time tracking): {dict(final_monitor_stats)}")

    processing_elapsed = time.time() - start_time
    total_positions = total_stats['lichess_positions'] + total_stats['lc0_positions']
    
    if args.benchmark:
        positions_per_second = total_positions / processing_elapsed if processing_elapsed > 0 else 0
        logging.info(f"Processing benchmark: {positions_per_second:.0f} positions/second average "
                    f"({total_positions:,} positions in {processing_elapsed:.1f}s with {args.jobs} workers)")

    logging.info(f"Final worker stats (actual totals): {dict(total_stats)}")
    
    # Sanity check on positions per game
    total_worker_positions = total_stats['lichess_positions'] + total_stats['lc0_positions']
    if total_worker_positions > 0:
        avg_positions_per_game = total_worker_positions / max(1, len(final_buckets))  # rough estimate
        logging.info(f"Sanity check: ~{avg_positions_per_game:.0f} positions per input file (not per game)")
        if avg_positions_per_game > 500000:  # Very rough heuristic
            logging.warning("⚠️  Position count seems unusually high - please verify data integrity")

    # Final mixing pass - files are already in output directory
    # This section is now conditional based on args.skip_final_shuffle_pass
    
    if args.skip_final_shuffle_pass:
        logging.info("Skipping final disk-based shuffle pass as requested.")
        for src in ("lichess", "lc0"):
            source_dir = out_dir / src
            if not source_dir.exists():
                logging.info(f"No {src} directory found at {source_dir}, skipping collection.")
                continue

            # Files are named like {src}_w{PID}_c{ID}.npz
            worker_chunk_files = sorted(list(source_dir.glob(f"{src}_w*_c*.npz")))
            
            if worker_chunk_files:
                logging.info(f"Collecting and renaming {len(worker_chunk_files)} worker-generated chunk files for {src}.")
                for i, old_file_path in enumerate(worker_chunk_files):
                    new_file_name = source_dir / f"{src}_{i:07d}.npz"
                    try:
                        old_file_path.rename(new_file_name)
                    except Exception as e:
                        logging.error(f"Error renaming {old_file_path} to {new_file_name}: {e}")
                logging.info(f"Finished collecting and renaming files for {src}.")
            else:
                logging.info(f"No worker-generated chunk files found for {src} in {source_dir} (expected format: {src}_w*_c*.npz).")
    else:
        logging.info("Proceeding with final disk-based shuffle pass.")
        for src in ("lichess", "lc0"):
            source_dir = out_dir / src
            if not source_dir.exists():
                logging.info(f"No {src} directory found, skipping shuffle")
                continue
            
            # Files are named like {src}_w{PID}_c{ID}.npz
            initial_file_pattern = f"{src}_w*_c*.npz"
            all_npz_files = list(source_dir.glob(initial_file_pattern))
            
            if len(all_npz_files) > 1:
                logging.info(f"Final mixing pass for {src} with {len(all_npz_files)} files (input pattern: {initial_file_pattern})")
                for _ in range(args.mix_passes):
                    _simple_merge_shuffle_split_parallel(source_dir, args.seed, shuffle_workers, args.shuffle_batch_size, args.benchmark, args.verify_shuffle)
            else:
                logging.info(f"Skipping mixing for {src}: only {len(all_npz_files)} files found")
            
            # Still rename single file for consistency
            if len(all_npz_files) == 1:
                single_file = all_npz_files[0]
                final_name = source_dir / f"{src}_0000000.npz"
                if single_file != final_name:
                    single_file.rename(final_name)

    # Metadata manifest
    metadata = {
        "chunk_size"             : args.chunk_size,
        "total_lichess_positions": total_stats["lichess_positions"],
        "total_lc0_positions"    : total_stats["lc0_positions"],
        "lichess_chunks"         : total_stats["lichess_chunks"], # Now refers to # of buffers
        "lc0_chunks"             : total_stats["lc0_chunks"],   # Now refers to # of buffers
        "mix_passes"             : args.mix_passes if not args.skip_final_shuffle_pass else 0,
        "mix_freq"               : args.mix_freq if not args.skip_final_shuffle_pass else 0,
        "worker_buffer_multiplier": args.worker_buffer_multiplier,
        "final_shuffle_skipped"  : args.skip_final_shuffle_pass,
        "preprocessing_date"     : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Done.  Manifest: %s", json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()