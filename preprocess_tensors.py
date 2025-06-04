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
import zipfile # Added for specific error handling
import gc # For explicit garbage collection
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

def _flush_chunk_from_buffer(chunk_id_key: str, chunk_size: int, buf: dict[str, list], 
                             output_path_for_chunk_type_and_src: Path, 
                             chunk_counter_for_src_type_and_split: dict[str, int], 
                             chunk_file_prefix: str, # e.g., "train_lichess" or "test_lc0"
                             disable_verification: bool = False) -> int:
    """Extract exactly chunk_size positions from buffer, shuffle them, and save to disk.
    Returns number of positions actually saved (may be less than chunk_size if buffer is smaller)."""
    if not buf["bitboards"]:
        return 0

    process_id = os.getpid()
    output_path_for_chunk_type_and_src.mkdir(parents=True, exist_ok=True)
    
    available_positions = len(buf["bitboards"])
    positions_to_save = min(chunk_size, available_positions)
    
    if positions_to_save == 0:
        return 0
    
    # Filename: {prefix}_w{PID}_c{counter}.npz (e.g., train_lichess_w123_c0000001.npz)
    # src_label is already incorporated into chunk_file_prefix by the caller
    fname = output_path_for_chunk_type_and_src / f"{chunk_file_prefix}_w{process_id}_c{chunk_counter_for_src_type_and_split[chunk_id_key]:07d}.npz"
    chunk_counter_for_src_type_and_split[chunk_id_key] += 1
    
    logging.info(f"_flush_chunk_from_buffer: Saving {positions_to_save} positions for {chunk_file_prefix} to {fname}")

    # Extract positions to save and shuffle them
    chunk_data = {}
    for k in buf.keys():
        chunk_data[k] = buf[k][:positions_to_save]
        buf[k] = buf[k][positions_to_save:]
    
    if positions_to_save > 1:
        perm_seed = os.getpid() + chunk_counter_for_src_type_and_split[chunk_id_key] + int(time.time() * 1000)
        perm_seed = perm_seed % (2**32) 
        np_rng_shuffle = np.random.RandomState(perm_seed)
        perm = np_rng_shuffle.permutation(positions_to_save)
        
        for k in chunk_data.keys():
            if chunk_data[k] and isinstance(chunk_data[k][0], np.ndarray):
                try:
                    stacked = np.stack(chunk_data[k])
                    shuffled_stacked = stacked[perm]
                    chunk_data[k] = [shuffled_stacked[i] for i in range(positions_to_save)]
                except ValueError as e:
                    logging.error(f"Error stacking/shuffling list of ndarrays for key {k}: {e}")
            else:
                temp_array = np.array(chunk_data[k], dtype=object)
                temp_array = temp_array[perm]
                chunk_data[k] = list(temp_array)

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
                if fname.exists(): fname.unlink()
                raise verify_e
        logging.info(f"_flush_chunk_from_buffer: Successfully saved {fname} ({total_bytes} bytes)")
    except Exception as e:
        logging.error(f"_flush_chunk_from_buffer: ERROR saving {fname}: {e}")
        if fname.exists(): fname.unlink()
        raise
    return positions_to_save


def _sample_to_lists(sample: dict, lists: dict[str, list]):
    """Convert one stream_worker sample to flat lists (no flips)."""
    # Arrays are already fp16 from stream_worker; astype with copy=False avoids extra copy
    lists["bitboards"         ].append(sample["bitboards_original"].astype(np.float16, copy=False))
    lists["policy_target"     ].append(sample["policy"           ].astype(np.float16, copy=False))
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


def _worker(files_for_worker: list[tuple[str, str, str]], # List of (file_path, source_name, data_split_type)
            job_seed: int, 
            chunk_size: int, 
            base_output_dir_str: str, 
            worker_buffer_multiplier: int, 
            disable_verification: bool = False, 
            progress_queue=None,
            use_intermediate_dir: bool = False):
    """Executed in each subprocess. Processes files assigned to it, placing outputs in correct train/test subdirs."""
    logger = logging.getLogger()
    effective_buffer_size = chunk_size * worker_buffer_multiplier
    logger.info(f"Worker starting with {len(files_for_worker)} files. Effective buffer: {effective_buffer_size} positions.")

    random.seed(job_seed)
    base_output_dir = Path(base_output_dir_str)

    # Buffers and counters are now per-source AND per-data_split_type for this worker.
    # However, a worker processes files that are ALREADY designated as train or test.
    # So, it only needs one set of buffers/counters, but the output path and stats keys change.
    
    # Let's simplify: the worker processes one file at a time, and each file has a defined data_split_type.
    # We will report stats with keys like "lichess_train_positions", "lc0_test_chunks", etc.

    # These will store the aggregated results from this worker for all files it processes.
    worker_total_stats = defaultdict(int)

    if progress_queue:
        try: progress_queue.put({'type': 'total_files', 'count': len(files_for_worker)}) 
        except: pass

    # Temporary buffer and counters for the current file's data_split_type and source_type.
    # These get reset or re-targeted for each file if its split/source changes (though round-robin should group them somewhat).
    current_processing_buf = {k: [] for k in NUMPY_KEYS}
    # Chunk counter needs to be unique per (data_split_type, source_name) across the lifetime of this worker.
    # Example: { "train_lichess": 0, "test_lc0": 1, ... }
    chunk_counters_map = defaultdict(int) 

    for file_idx, (file_path, source_name, data_split_type) in enumerate(files_for_worker):
        logger.info(f"Processing file {file_idx+1}/{len(files_for_worker)}: {file_path} ({source_name}, {data_split_type})")
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.error(f"File does not exist: {file_path}")
            continue
        
        # Define output directory for chunks from THIS specific file
        # If using intermediate directory, write to intermediate/train/lichess/ etc.
        # Otherwise, write directly to train/lichess/ etc.
        if use_intermediate_dir:
            current_output_path_for_npz = base_output_dir / "intermediate" / data_split_type / source_name
        else:
            current_output_path_for_npz = base_output_dir / data_split_type / source_name
            
        # Define a unique key for the chunk counter for this file's characteristics
        chunk_counter_key = f"{data_split_type}_{source_name}"
        # Define a prefix for the .npz filenames, e.g., "train_lichess" or "test_lc0"
        npz_file_prefix = f"{data_split_type}_{source_name}"
        # Define stat keys for progress reporting, e.g., lichess_train, lc0_test
        stat_source_prefix = f"{source_name}_{data_split_type}"

        positions_from_file = 0
        games_from_file = 0
        positions_since_last_update = 0
        
        # Clear buffer for this new file's data type (or if switching types)
        for k_buf in current_processing_buf: current_processing_buf[k_buf].clear()
        
        try:
            # stream_samples expects an iterable of paths. Here, one PGN shard file.
            # is_lichess_source_override ensures correct internal filtering if stream_samples uses it.
            for game_samples_list_for_one_game in stream_samples([file_path], flips=False, is_lichess_source_override=(source_name == "lichess")):
                games_from_file += 1
                # game_samples_list_for_one_game is the list of samples for the current game.
                if not game_samples_list_for_one_game: continue

                positions_from_file += len(game_samples_list_for_one_game)
                
                for s_data in game_samples_list_for_one_game:
                    # The s_data["source_flag"] (0 for LC0, 1 for Lichess) from stream_worker
                    # should align with `source_name` because pgn_splitter already labeled shards.
                    # We use `source_name` (from the input file path) for directory structure.
                    _sample_to_lists(s_data, current_processing_buf)
                    worker_total_stats[f"{stat_source_prefix}_positions"] += 1
                    positions_since_last_update += 1
                    
                    if positions_since_last_update >= 100 and progress_queue:
                        try:
                            progress_queue.put({
                                'type': 'position', # Generic type, specific keys in worker_total_stats distinguish train/test
                                'source_split_key': stat_source_prefix, # e.g. lichess_train
                                'count': positions_since_last_update
                            })
                            positions_since_last_update = 0
                        except: pass
                    
                    if len(current_processing_buf["bitboards"]) >= chunk_size:
                        saved_count = _flush_chunk_from_buffer(
                            chunk_counter_key, # Pass the combined key for the counter
                            chunk_size, 
                            current_processing_buf, 
                            current_output_path_for_npz, # Specific path like .../train/lichess
                            chunk_counters_map, # Pass the map, _flush_chunk uses chunk_counter_key
                            npz_file_prefix, # e.g. train_lichess for filename
                            disable_verification
                        )
                        if saved_count > 0:
                            worker_total_stats[f"{stat_source_prefix}_chunks"] += 1
                            if progress_queue:
                                try:
                                    progress_queue.put({
                                        'type': 'chunk', # Generic type
                                        'source_split_key': stat_source_prefix # e.g. lichess_train
                                    })
                                except: pass
            
            if positions_since_last_update > 0 and progress_queue:
                try:
                    progress_queue.put({
                        'type': 'position',
                        'source_split_key': stat_source_prefix,
                        'count': positions_since_last_update
                    })
                except: pass
            
            logger.info(f"Processed {games_from_file} games ({positions_from_file} positions) from file {file_path}")
            
        except Exception as e_file_proc:
            logger.error(f"Error processing PGN shard {file_path}: {e_file_proc}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        if progress_queue: 
            try: progress_queue.put({'type': 'file_completed'}) 
            except: pass

    # After all files for this worker are done, flush any remaining data in the last used buffer.
    # This requires knowing the last `source_name` and `data_split_type` to correctly flush.
    # This part is tricky because current_processing_buf holds data for the *last* file processed.
    # We need to flush it to the correct corresponding directory.
    if files_for_worker: # Check if any files were processed
        last_file_path, last_source_name, last_data_split_type = files_for_worker[-1]
        last_output_path_for_npz = base_output_dir / last_data_split_type / last_source_name
        last_npz_file_prefix = f"{last_data_split_type}_{last_source_name}"
        last_stat_source_prefix = f"{last_source_name}_{last_data_split_type}"
        
        if current_processing_buf["bitboards"]: # If there's anything left in the buffer
            logger.info(f"Flushing remaining {len(current_processing_buf['bitboards'])} positions for {last_npz_file_prefix}...")
            saved_final_count = _flush_chunk_from_buffer(
                f"{last_data_split_type}_{last_source_name}", # Construct the key for the counter
                chunk_size, 
                current_processing_buf, 
                last_output_path_for_npz, 
                chunk_counters_map, 
                last_npz_file_prefix, 
                disable_verification
            )
            if saved_final_count > 0:
                worker_total_stats[f"{last_stat_source_prefix}_chunks"] += 1
                if progress_queue: 
                    try: progress_queue.put({'type': 'chunk', 'source_split_key': last_stat_source_prefix}) 
                    except: pass

    logger.info(f"Worker finished. Final stats for this worker: {dict(worker_total_stats)}")
    return worker_total_stats # Return the dict of all counts from this worker


def worker_wrapper(args_tuple):
    """Wrapper function to unpack arguments for the worker."""
    # files_for_worker_arg is a list of (file_path, source_name, data_split_type) tuples
    files_for_worker_arg, job_seed_arg, chunk_size_arg, base_output_dir_str_arg, \
    worker_buffer_multiplier_arg, disable_verification_arg, progress_queue_arg, use_intermediate_dir_arg = args_tuple
    
    return _worker(
        files_for_worker_arg, 
        job_seed_arg, 
        chunk_size_arg, 
        base_output_dir_str_arg, 
        worker_buffer_multiplier_arg,
        disable_verification_arg, 
        progress_queue_arg,
        use_intermediate_dir_arg
    )

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
                                         shuffle_batch_size: int, benchmark=False, verify_shuffle=False, max_delay: float = 5.0):
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
            pool.submit(_simple_merge_shuffle_split_worker, batch, rng_seed + i, verify_shuffle, max_delay)
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


def _simple_merge_shuffle_split_worker(file_batch: list[Path], rng_seed: int, verify_shuffle: bool = False, max_delay: float = 5.0):
    """Worker function: merge a batch of files, shuffle, split back to original files."""
    if len(file_batch) < 2:
        return "Not enough files in batch to shuffle", 0

    # Add random delay to stagger memory usage peaks across workers
    if max_delay > 0:
        import random as worker_random
        delay = worker_random.uniform(0, max_delay)
        time.sleep(delay)
        logging.debug(f"Shuffle worker delayed {delay:.2f}s to stagger memory usage")

    batch_names = [f.name for f in file_batch]
    try:
        start_time = time.time()

        # Load all files in the batch
        all_data_in_batch: list[dict[str, np.ndarray]] = []
        valid_files_in_batch: list[Path] = [] # Keep track of files that loaded successfully
        
        for file_path in file_batch:
            try:
                # Ensure file actually exists before trying to load, as it might have been deleted by another worker
                if not file_path.exists():
                    logging.warning(f"File {file_path} not found during shuffle load, possibly deleted by another process. Skipping.")
                    continue
                data = dict(np.load(file_path, allow_pickle=False))
                all_data_in_batch.append(data)
                valid_files_in_batch.append(file_path)
            except (zipfile.BadZipFile, ValueError, IOError) as e_load: # Common errors for corrupted/unreadable NPZ
                logging.error(f"Corrupted or unreadable NPZ file in shuffle worker: {file_path}. Error: {e_load}. Attempting to delete.")
                try:
                    file_path.unlink(missing_ok=True)
                    logging.info(f"Deleted corrupted file: {file_path}")
                except OSError as delete_e:
                    logging.error(f"Failed to delete corrupted file {file_path}: {delete_e}")
            except Exception as e_other: # Catch any other unexpected errors during load
                logging.error(f"Unexpected error loading NPZ file {file_path} in shuffle worker: {e_other}. Attempting to delete.")
                try:
                    file_path.unlink(missing_ok=True)
                    logging.info(f"Deleted unreadable file: {file_path}")
                except OSError as delete_e:
                    logging.error(f"Failed to delete unreadable file {file_path}: {delete_e}")

        if not all_data_in_batch: # Handles case where all files in batch were corrupt
            return f"Batch {batch_names} has no valid files left after load errors (all_data_in_batch is empty).", 0
        if len(all_data_in_batch) < 2: # If not enough valid files left to merge
            # This can happen if some files were corrupted and deleted
            logging.info(f"Skipping shuffle for batch {batch_names}: only {len(all_data_in_batch)} valid file(s) remaining, need at least 2.")
            # Return 0 positions as no shuffle operation was performed on this batch
            # The files that were valid remain as they are.
            # Note: The original implementation did not explicitly return total positions from skipped batches due to <2 files.
            # We maintain this, assuming those positions are not "processed" by this shuffle op.
            return f"Not enough valid files (found {len(all_data_in_batch)}) in batch {batch_names} to shuffle.", 0

        # Verify keys match across all *valid* files in the batch
        first_file_keys = set(all_data_in_batch[0].keys())
        for i, data_dict in enumerate(all_data_in_batch[1:], 1):
            if set(data_dict.keys()) != first_file_keys:
                # This indicates a more fundamental issue than just a corrupted file, possibly mixed data types.
                # For now, we will log an error and might have to skip this batch or handle it more gracefully.
                # If valid_files_in_batch[0] or valid_files_in_batch[i] is used, ensure mapping from data_dict back to file path if needed for error.
                logging.error(f"Key mismatch in batch. File corresponding to first data keys vs file for data at index {i}. Bailing on this batch.")
                # This is a critical error for merging, so we can't proceed with this batch.
                return f"Key mismatch in batch {batch_names}. Cannot merge.", 0

        # Calculate total positions from successfully loaded data
        total_positions = sum(len(data_dict["bitboards"]) for data_dict in all_data_in_batch if "bitboards" in data_dict and data_dict["bitboards"].ndim > 0) # ensure bitboards exist and are not empty
        if total_positions == 0:
            return f"Batch {batch_names} (after loading {len(all_data_in_batch)} valid files) contains no positions.", 0

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

        # Explicitly delete the list of individual file data to free memory sooner
        del all_data_in_batch
        # At this point, valid_files_in_batch (list of Path objects) is still needed for saving.

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

        # Split the shuffled data back into parts, matching the number of successfully loaded input files
        num_valid_files_in_batch = len(valid_files_in_batch)
        
        if num_valid_files_in_batch == 0: # Should have been caught earlier, but as a safeguard
            logging.warning(f"No valid files remained in batch {batch_names} to save shuffled data to. Total positions: {total_positions}")
            return f"Batch {batch_names} ended up with no valid files to save to.", 0

        # Calculate roughly equal split points for positions
        base_split_size = total_positions // num_valid_files_in_batch
        remainder = total_positions % num_valid_files_in_batch
        split_sizes = [base_split_size + 1] * remainder + [base_split_size] * (num_valid_files_in_batch - remainder)
        
        current_pos_idx = 0
        for i in range(num_valid_files_in_batch):
            start_idx = current_pos_idx
            end_idx = current_pos_idx + split_sizes[i]
            data_to_save = {key: merged_arrays[key][start_idx:end_idx] for key in merged_arrays}
            np.savez_compressed(valid_files_in_batch[i], **data_to_save) # Save to the corresponding valid file path
            current_pos_idx = end_idx

        elapsed = time.time() - start_time
        result_msg = f"Shuffled batch of {num_valid_files_in_batch} files ({batch_names}) with {total_positions} total positions in {elapsed:.2f}s"
        return result_msg, total_positions

    except Exception as e:
        error_msg = f"ERROR shuffling batch {batch_names}: {e}"
        logging.error(error_msg)
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return error_msg, 0
    finally:
        # Encourage garbage collection at the end of the worker's operation for this batch
        gc.collect()

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
        
        self.estimated_peak_buffer_ram_mb = 0
        if self.worker_buffer_multiplier_arg > 0:
            effective_buffer_sz = self.chunk_size_arg * self.worker_buffer_multiplier_arg
            self.estimated_peak_buffer_ram_mb = (APPROX_BYTES_PER_POSITION * effective_buffer_sz * 2) / (1024 * 1024)
        
        # Stats keys will be like: lichess_train_positions, lc0_test_chunks, etc.
        self.stats = defaultdict(int) # Use defaultdict for easier updates
        self.last_stats = defaultdict(int)
        
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
                    # update['source_split_key'] is e.g. "lichess_train"
                    self.stats[f"{update['source_split_key']}_positions"] += update['count']
                elif update['type'] == 'chunk':
                    self.stats[f"{update['source_split_key']}_chunks"] += 1
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
            current_stats = self.stats.copy() # defaultdict.copy() is fine
            
            time_delta = current_time - self.last_update_time
            
            # Calculate total train and test positions/chunks from current_stats
            total_train_pos = current_stats['lichess_train_positions'] + current_stats['lc0_train_positions']
            total_test_pos = current_stats['lichess_test_positions'] + current_stats['lc0_test_positions']
            total_train_chunks = current_stats['lichess_train_chunks'] + current_stats['lc0_train_chunks']
            total_test_chunks = current_stats['lichess_test_chunks'] + current_stats['lc0_test_chunks']
            
            combined_current_positions = total_train_pos + total_test_pos

            # Calculate last totals similarly
            last_total_train_pos = self.last_stats['lichess_train_positions'] + self.last_stats['lc0_train_positions']
            last_total_test_pos = self.last_stats['lichess_test_positions'] + self.last_stats['lc0_test_positions']
            combined_last_positions = last_total_train_pos + last_total_test_pos
            
            if time_delta > 0 and combined_current_positions > combined_last_positions:
                recent_rate = (combined_current_positions - combined_last_positions) / time_delta
                overall_rate = combined_current_positions / (current_time - self.start_time) if (current_time - self.start_time) > 0 else 0
                efficiency = overall_rate / self.total_workers if self.total_workers > 0 else 0
                
                worker_progress = (current_stats['completed_workers'] / self.total_workers * 100) if self.total_workers > 0 else 0
                
                file_progress_str = ""
                if current_stats['total_files'] > 0:
                    file_pct = (current_stats['files_processed'] / current_stats['total_files'] * 100)
                    file_progress_str = f"Files: {current_stats['files_processed']}/{current_stats['total_files']} ({file_pct:.1f}%) | "
                
                mem_estimate_str = f"Est. peak buffer RAM/worker: {self.estimated_peak_buffer_ram_mb:.1f} MB | " if self.estimated_peak_buffer_ram_mb > 0 else ""

                logging.info(
                    f"PROGRESS: {current_stats['completed_workers']}/{self.total_workers} workers ({worker_progress:.1f}%) | "
                    f"{file_progress_str}"
                    f"Positions: Train={total_train_pos:,} (L:{current_stats['lichess_train_positions']:,}, LC0:{current_stats['lc0_train_positions']:,}) | "
                    f"Test={total_test_pos:,} (L:{current_stats['lichess_test_positions']:,}, LC0:{current_stats['lc0_test_positions']:,}) | "
                    f"Rate: {recent_rate:.0f}/s recent, {overall_rate:.0f}/s avg | "
                    f"{mem_estimate_str}"
                    f"Efficiency: {efficiency:.0f} pos/s/worker | "
                    f"Chunks: Train (L:{current_stats['lichess_train_chunks']:,}, LC0:{current_stats['lc0_train_chunks']:,}) | "
                    f"Test (L:{current_stats['lichess_test_chunks']:,}, LC0:{current_stats['lc0_test_chunks']:,})"
                )
            
            self.last_update_time = current_time
            self.last_stats = current_stats # Store the copy of current_stats
    
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
    ap.add_argument("--raw-dir",  required=True, 
                    help="Directory with pre-sharded PGNs. Expected structure: raw-dir/{train|test}/{lichess|lc0}/*.pgn")
    ap.add_argument("--out-dir",  required=True, help="Output dir for tensors. Will mirror train/test structure.")
    ap.add_argument("--chunk-size",   type=int, default=DEFAULT_CHUNK_SIZE,
                    help="Positions per .npz chunk")
    ap.add_argument("--jobs",         type=int, default=mp.cpu_count(),
                    help="Parallel worker processes")
    ap.add_argument("--mix-passes",   type=int, default=DEFAULT_MIX_PASSES,
                    help="Shuffle passes (merge‑shuffle‑split)")
    ap.add_argument("--mix-freq",    type=int, default=DEFAULT_MIX_FREQ,
                    help="Run shuffler every N new chunks (per source type and data split)")
    ap.add_argument("--seed",        type=int, default=2024,
                    help="Global RNG seed for general purposes (worker file shuffling, etc.)")
    ap.add_argument("--log-level",   type=str, default=DEFAULT_LOG_LVL,
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging detail")
    
    # Test set arguments are REMOVED as splitting is now done by pgn_splitter.py

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
    
    ap.add_argument("--shuffle-only", action="store_true",
                    help="If set, skips PGN processing and only performs shuffling of existing .npz files in the output directory.")
    
    # Memory optimization arguments
    ap.add_argument("--use-intermediate-dir", action="store_true",
                    help="Write unshuffled files to intermediate directory first, then copy to final location after first shuffle. "
                         "This doubles disk usage but provides recovery capability and can help with memory optimization.")
    ap.add_argument("--shuffle-worker-delay-max", type=float, default=5.0,
                    help="Maximum random delay (in seconds) before shuffle workers start processing. "
                         "This staggers memory usage peaks. (Default: %(default)s)")
    ap.add_argument("--resume-from-intermediate", action="store_true",
                    help="Skip PGN processing and resume from existing intermediate directory. "
                         "Useful for recovery when processing succeeded but shuffle failed.")
    
    return ap.parse_args()


def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(processName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def perform_shuffle_only(args, out_dir: Path, shuffle_workers_count: int):
    """Performs only the shuffling and renaming steps on existing .npz files."""
    logging.info("Starting shuffle-only operation.")
    if args.mix_passes <= 0:
        logging.info("mix_passes is 0 or less, no shuffling will be performed. Files will only be collected and renamed.")

    for data_split_type in ["train", "test"]:
        for src in ("lichess", "lc0"):
            source_dir_for_split = out_dir / data_split_type / src

            if not source_dir_for_split.exists() or not source_dir_for_split.is_dir():
                logging.info(f"Directory not found: {source_dir_for_split}, skipping operations for {data_split_type}/{src}.")
                continue

            all_npz_files_in_dir = list(source_dir_for_split.glob("*.npz"))
            num_files_found = len(all_npz_files_in_dir)
            logging.info(f"Found {num_files_found} .npz files in {source_dir_for_split} for {data_split_type}/{src}.")

            if num_files_found > 1 and args.mix_passes > 0:
                logging.info(f"Proceeding with {args.mix_passes} shuffle pass(es) for {data_split_type}/{src} with {num_files_found} files.")
                for pass_num in range(args.mix_passes):
                    logging.info(f"Shuffle Pass {pass_num + 1}/{args.mix_passes} for {data_split_type}/{src}")
                    _simple_merge_shuffle_split_parallel(
                        source_dir_for_split,
                        args.seed + pass_num,
                        shuffle_workers_count,
                        args.shuffle_batch_size,
                        args.benchmark,
                        args.verify_shuffle,
                        args.shuffle_worker_delay_max
                    )
                logging.info(f"Completed shuffle passes for {data_split_type}/{src}.")
            elif num_files_found <= 1 and args.mix_passes > 0:
                logging.info(f"Only {num_files_found} file(s) found. No shuffle pass needed for {data_split_type}/{src}.")
            elif args.mix_passes <= 0:
                 logging.info(f"mix_passes <= 0. Skipping shuffle for {data_split_type}/{src}.")


            # Rename all *.npz files in the directory to the standard format,
            # regardless of whether shuffling occurred (for consistency).
            # Re-glob to ensure we have the current state of files.
            final_files_to_rename = sorted(list(source_dir_for_split.glob("*.npz")))
            if final_files_to_rename:
                logging.info(f"Renaming {len(final_files_to_rename)} files in {source_dir_for_split} to final format {src}_NNNNNNN.npz.")
                for i, old_file_path in enumerate(final_files_to_rename):
                    new_file_name = source_dir_for_split / f"{src}_{i:07d}.npz"
                    if old_file_path == new_file_name:
                        logging.debug(f"File {old_file_path} is already correctly named. Skipping rename.")
                        continue
                    try:
                        old_file_path.rename(new_file_name)
                    except Exception as e:
                        logging.error(f"Error renaming {old_file_path} to {new_file_name}: {e}")
            elif num_files_found > 0: # Files were there but now glob is empty - should not happen
                 logging.warning(f"No .npz files found for renaming in {source_dir_for_split} after operations, though files were initially present. This is unexpected.")
            else: # No files initially, none to rename
                logging.info(f"No .npz files to rename in {source_dir_for_split}.")

    logging.info("Shuffle-only operation completed.")


def generate_metadata_for_shuffle_only(args, out_dir: Path):
    """Generates metadata.json when running in --shuffle-only mode."""
    logging.info("Generating metadata for shuffle-only run...")
    metadata_path = out_dir / "metadata.json"
    final_counts = defaultdict(int)

    logging.info("Recalculating position and chunk counts from final .npz files...")
    for data_split_type in ["train", "test"]:
        for src_type in ["lichess", "lc0"]:
            src_dir = out_dir / data_split_type / src_type
            if src_dir.exists() and src_dir.is_dir():
                num_chunks_for_src_split = 0
                num_positions_for_src_split = 0
                # Files should now be named {src_type}_NNNNNNN.npz
                renamed_files = list(src_dir.glob(f"{src_type}_*.npz"))
                for f_path in renamed_files:
                    try:
                        # Ensure file actually exists before trying to load
                        if not f_path.exists():
                            logging.warning(f"File {f_path} not found during metadata scan, possibly deleted. Skipping.")
                            continue
                        with np.load(f_path) as data:
                            if "bitboards" in data and hasattr(data["bitboards"], 'shape') and len(data["bitboards"].shape) > 0:
                                num_positions_for_src_split += data["bitboards"].shape[0]
                            else:
                                logging.debug(f"Key 'bitboards' not found or shape unexpected in {f_path}. Position count for this file might be 0.")
                        num_chunks_for_src_split += 1 # Increment chunk count only if load succeeds
                    except (zipfile.BadZipFile, ValueError, IOError) as e_load:
                        logging.warning(f"Corrupted or unreadable NPZ file {f_path} encountered during metadata generation. Error: {e_load}. Attempting to delete.")
                        try:
                            f_path.unlink(missing_ok=True)
                            logging.info(f"Deleted corrupted file for metadata: {f_path}")
                        except OSError as delete_e:
                            logging.error(f"Failed to delete corrupted file {f_path} during metadata scan: {delete_e}")
                        # File is skipped for counting, num_chunks_for_src_split is NOT incremented for this file.
                    except Exception as e:
                        logging.warning(f"Could not read or parse {f_path} for metadata counts due to unexpected error: {e}. File will be skipped.")
                
                final_counts[f"{src_type}_{data_split_type}_positions"] = num_positions_for_src_split
                final_counts[f"{src_type}_{data_split_type}_chunks"] = num_chunks_for_src_split
                logging.info(f"Counted for {data_split_type}/{src_type}: {num_positions_for_src_split:,} positions, {num_chunks_for_src_split:,} chunks.")
            else:
                logging.info(f"Directory {src_dir} not found for metadata counts, assuming 0 positions/chunks for this type.")

    metadata = {
        "chunk_size"             : args.chunk_size,
        "total_lichess_train_positions": final_counts.get("lichess_train_positions", 0),
        "total_lc0_train_positions"    : final_counts.get("lc0_train_positions", 0),
        "total_lichess_test_positions": final_counts.get("lichess_test_positions", 0),
        "total_lc0_test_positions"    : final_counts.get("lc0_test_positions", 0),
        "lichess_train_chunks"         : final_counts.get("lichess_train_chunks", 0),
        "lc0_train_chunks"             : final_counts.get("lc0_train_chunks", 0),
        "lichess_test_chunks"          : final_counts.get("lichess_test_chunks", 0),
        "lc0_test_chunks"              : final_counts.get("lc0_test_chunks", 0),
        "mix_passes"             : args.mix_passes,
        "mix_freq"               : args.mix_freq, # Retain for context, though not used for triggering in this mode
        "worker_buffer_multiplier": args.worker_buffer_multiplier, # Retain for context
        "final_shuffle_skipped"  : False, # Shuffle was the main operation, so not skipped
        "preprocessing_date"     : time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_operation_type"    : "shuffle_only"
    }
    
    if metadata_path.exists():
        logging.info(f"Existing metadata file found at {metadata_path}. It will be overwritten.")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info("Done writing metadata for shuffle-only run. Manifest: %s", json.dumps(metadata, indent=2))


def _copy_intermediate_to_final(intermediate_dir: Path, final_dir: Path, data_split_type: str, src: str):
    """Copy files from intermediate directory to final directory structure."""
    intermediate_source_dir = intermediate_dir / data_split_type / src
    final_source_dir = final_dir / data_split_type / src
    
    if not intermediate_source_dir.exists():
        logging.info(f"No intermediate directory found at {intermediate_source_dir}, skipping copy for {data_split_type}/{src}.")
        return
    
    # Create final directory structure
    final_source_dir.mkdir(parents=True, exist_ok=True)
    
    # Pattern for worker-generated files in intermediate directory
    intermediate_pattern = f"{data_split_type}_{src}_w*_c*.npz"
    intermediate_files = list(intermediate_source_dir.glob(intermediate_pattern))
    
    if intermediate_files:
        logging.info(f"Copying {len(intermediate_files)} files from {intermediate_source_dir} to {final_source_dir}")
        for intermediate_file in intermediate_files:
            final_file = final_source_dir / intermediate_file.name
            try:
                shutil.copy2(intermediate_file, final_file)
            except Exception as e:
                logging.error(f"Error copying {intermediate_file} to {final_file}: {e}")
                raise
        logging.info(f"Successfully copied {len(intermediate_files)} files for {data_split_type}/{src}")
    else:
        logging.info(f"No files found matching pattern {intermediate_pattern} in {intermediate_source_dir}")


def _cleanup_intermediate_directory(intermediate_dir: Path):
    """Remove the intermediate directory after successful processing."""
    if intermediate_dir.exists():
        try:
            shutil.rmtree(intermediate_dir)
            logging.info(f"Cleaned up intermediate directory: {intermediate_dir}")
        except Exception as e:
            logging.warning(f"Failed to clean up intermediate directory {intermediate_dir}: {e}")


def main():
    script_global_start_time = time.time() # For total script execution time
    args = parse_args()
    setup_logging(args.log_level)

    # Cluster optimizations
    if args.large_chunks:
        chunk_size = max(args.chunk_size, 65536)
        logging.info(f"Large chunks mode: Using chunk size {chunk_size}")
    else:
        chunk_size = args.chunk_size
    
    if args.shuffle_workers is None:
        shuffle_workers = min(8, max(1, args.jobs // 4))
    else:
        shuffle_workers = args.shuffle_workers
    
    logging.info(f"Config: jobs={args.jobs}, shuffle_workers={shuffle_workers}, chunk_size={chunk_size}")
    if args.use_intermediate_dir:
        logging.info("Using intermediate directory workflow.")
    if args.shuffle_worker_delay_max > 0:
        logging.info(f"Shuffle workers will use random delays up to {args.shuffle_worker_delay_max:.1f}s.")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir_path = out_dir / "intermediate"

    if args.shuffle_only:
        logging.info("Running in --shuffle-only mode. PGN processing will be skipped.")
        if not out_dir.exists():
            logging.error(f"Output directory {out_dir} does not exist. Cannot perform shuffle-only operation.")
            sys.exit(1)
        perform_shuffle_only(args, out_dir, shuffle_workers) # Pass shuffle_workers
        generate_metadata_for_shuffle_only(args, out_dir)
        logging.info("Shuffle-only mode finished.")
        sys.exit(0)

    # Initialize variables that will be used later, regardless of resume state
    total_stats = defaultdict(int)
    all_pgn_files_count = 0
    total_processed_positions_all_workers = 0
    pgn_processing_elapsed_time = 0.0 # Specifically for PGN processing part
    
    # This start_time is for the PGN processing block if it runs, or for the resume block
    block_start_time = time.time() 

    if args.resume_from_intermediate:
        logging.info(f"Resuming from intermediate directory: {intermediate_dir_path}. PGN processing skipped.")
        if not intermediate_dir_path.exists() or not intermediate_dir_path.is_dir():
            logging.error(f"Intermediate directory {intermediate_dir_path} not found or not a dir. Cannot resume.")
            sys.exit(1)
        if not list(intermediate_dir_path.rglob("*.npz")):
            logging.error(f"No .npz files in intermediate directory {intermediate_dir_path} to resume from.")
            sys.exit(1)
        # In resume mode, worker-generated files are expected in intermediate_dir_path.
        # They will be moved to out_dir/train|test/... before shuffling.
        # total_processed_positions_all_workers remains 0 for this script execution's PGN processing phase.

    else: # Normal PGN Processing
        logging.info(f"Starting PGN processing. Raw PGN directory: {raw_dir.resolve()}")
        
        # --- Collect PGN Shard Files ---
        collected_pgn_files: list[tuple[str, str, str]] = []
        exts_tuple = (".pgn", ".pgn.gz", ".pgn.bz2")
        for data_split_type in ["train", "test"]:
            split_dir = raw_dir / data_split_type
            if split_dir.exists() and split_dir.is_dir():
                logging.info(f"Scanning {split_dir} for PGN files...")
                found_in_split = 0
                for item_path in split_dir.rglob("*"):
                    if item_path.is_file() and \
                       (item_path.suffix.lower() in exts_tuple or \
                        any(str(item_path).lower().endswith(x) for x in exts_tuple)):
                        file_name_lower = item_path.name.lower()
                        inferred_source = "lichess" if "lichess" in file_name_lower else "lc0" if "lc0" in file_name_lower else None
                        if inferred_source:
                            collected_pgn_files.append((str(item_path), inferred_source, data_split_type))
                            found_in_split += 1
                        # else: logging.debug(f"Skipping file with no clear source: {item_path.name}")
                logging.info(f"Found {found_in_split} PGN files in {split_dir}.")
            # else: logging.warning(f"Directory {split_dir} not found.")

        if not collected_pgn_files:
            logging.error(f"No PGN files found in subdirectories of {raw_dir}. Exiting.")
            sys.exit(1)
        all_pgn_files_count = len(collected_pgn_files)
        random.Random(args.seed).shuffle(collected_pgn_files)
        logging.info(f"Collected and globally shuffled {all_pgn_files_count} PGN files.")

        # --- Distribute PGN Files to Worker Buckets ---
        final_buckets: list[list[tuple[str, str, str]]] = [[] for _ in range(args.jobs)]
        if args.jobs > 0:
            for idx, file_tuple in enumerate(collected_pgn_files):
                final_buckets[idx % args.jobs].append(file_tuple)
        elif collected_pgn_files: # jobs is 0 or less
            logging.warning("Number of jobs is 0 or less; PGN processing workers will not run.")
            # For consistency, if jobs=0, all files could go to a conceptual first bucket if needed later,
            # but ProcessPoolExecutor won't be used.
            # final_buckets[0].extend(collected_pgn_files)
        
        # --- Log Worker Assignment Summary --- 
        # ... (Logging as previously implemented) ...
        if args.jobs > 0:
            assigned_count_check = 0
            for i_bucket, bucket_c in enumerate(final_buckets):
                if bucket_c:
                    logging.info(f"  Worker {i_bucket} assigned {len(bucket_c)} files.")
                    assigned_count_check += len(bucket_c)
            if assigned_count_check != all_pgn_files_count:
                 logging.warning(f"Worker assignment count ({assigned_count_check}) mismatches total files ({all_pgn_files_count}).")

        # --- PGN Processing with ProcessPoolExecutor ---
        if args.jobs > 0 and any(final_buckets):
            manager = mp.Manager()
            progress_queue = manager.Queue()
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
                # Worker output dir is out_dir for use_intermediate_dir=True, else specific train/test subdirs
                # The _worker function itself handles placing files into intermediate_dir_path/train|test or out_dir/train|test
                futures = [
                    pool.submit(worker_wrapper, (
                        bucket, args.seed + i, chunk_size, 
                        str(out_dir), # Base for worker, _worker prepends "intermediate/" if use_intermediate_dir
                        args.worker_buffer_multiplier, args.disable_verification, 
                        progress_queue, args.use_intermediate_dir
                    )) for i, bucket in enumerate(final_buckets) if bucket
                ]
                logging.info(f"Submitted {len(futures)} PGN worker tasks.")
                pgn_progress_monitor = ProgressMonitor(len(futures), chunk_size_arg=chunk_size, worker_buffer_multiplier_arg=args.worker_buffer_multiplier, progress_queue=progress_queue)
                pgn_progress_monitor.start_monitoring()
                for future in as_completed(futures):
                    try:
                        worker_res = future.result()
                        for k, v in worker_res.items(): total_stats[k] += v # Aggregate stats from workers
                        if pgn_progress_monitor: pgn_progress_monitor.update_worker_completion(worker_res)
                    except Exception as e_pgn_worker:
                        logging.error(f"PGN Worker task failed: {e_pgn_worker}")
                        if pgn_progress_monitor: pgn_progress_monitor.update_worker_completion({})
                if pgn_progress_monitor:
                    pgn_progress_monitor.stop_monitoring()
                    # final_monitor_stats could be updated here if needed for metadata
                    # final_monitor_stats.update(pgn_progress_monitor.get_final_stats())
            logging.info("Finished PGN processing worker pool.")
        else:
            logging.info("Skipping PGN processing worker pool (jobs=0 or no files to process).")

        pgn_processing_elapsed_time = time.time() - block_start_time
        total_processed_positions_all_workers = sum(v for k, v in total_stats.items() if k.endswith("_positions"))
        logging.info(f"PGN processing phase completed in {pgn_processing_elapsed_time:.2f}s. Total positions from workers: {total_processed_positions_all_workers:,}")
        if args.benchmark and pgn_processing_elapsed_time > 0.01:
            rate = total_processed_positions_all_workers / pgn_processing_elapsed_time
            logging.info(f"  PGN Processing Benchmark: {rate:.0f} pos/s")
        if all_pgn_files_count > 0 and total_processed_positions_all_workers > 0:
             logging.info(f"  Avg positions/PGN file: {total_processed_positions_all_workers / all_pgn_files_count:.0f}")

    # --- Intermediate Directory File Movement (if applicable) ---
    # This step ensures files for shuffling are in out_dir/train|test/lichess|lc0/
    # If resuming: copy from intermediate_dir_path to out_dir subdirs.
    # If use_intermediate_dir (and not resuming): copy from intermediate_dir_path to out_dir subdirs.
    # If neither: files are already in out_dir subdirs (written directly by workers).

    if args.resume_from_intermediate or (args.use_intermediate_dir and not args.resume_from_intermediate):
        if not intermediate_dir_path.exists() or not intermediate_dir_path.is_dir():
            logging.error(f"Intermediate directory {intermediate_dir_path} expected but not found. Critical error. Exiting.")
            sys.exit(1)
        
        logging.info(f"Copying files from intermediate directory {intermediate_dir_path} to final output subdirectories for shuffling.")
        copied_files_count = 0
        for data_s_type in ["train", "test"]:
            for src_type in ("lichess", "lc0"):
                source_subdir = intermediate_dir_path / data_s_type / src_type
                dest_subdir = out_dir / data_s_type / src_type
                dest_subdir.mkdir(parents=True, exist_ok=True)

                if source_subdir.exists():
                    # Worker files are named: {data_split_type}_{src}_w{PID}_c{ID}.npz
                    worker_file_pattern = f"{data_s_type}_{src_type}_w*_c*.npz"
                    for f_to_copy in source_subdir.glob(worker_file_pattern):
                        try:
                            shutil.copy2(str(f_to_copy), str(dest_subdir / f_to_copy.name))
                            copied_files_count += 1
                        except Exception as e_cp_inter:
                            logging.error(f"Error copying {f_to_copy} to {dest_subdir}: {e_cp_inter}")
        if copied_files_count > 0:
            logging.info(f"Successfully copied {copied_files_count} files from {intermediate_dir_path}.")
        else:
            logging.warning(f"No worker-generated files found in {intermediate_dir_path} to copy. This might be okay if workers failed or if it was already cleaned.")
    
    # --- Final Shuffling and Renaming --- 
    # At this point, all files to be shuffled (worker-named) are in out_dir/train|test/lichess|lc0/
    if args.skip_final_shuffle_pass:
        # ... (renaming logic as before, operating on files in out_dir/train|test/...)
        logging.info("Skipping final disk-based shuffle pass as requested. Renaming worker files.")
        for data_split_type in ["train", "test"]:
            for src in ("lichess", "lc0"):
                # ... (renaming logic for worker files to final {src}_{idx}.npz format, as before)
                source_dir_for_rename = out_dir / data_split_type / src
                if not source_dir_for_rename.exists(): continue
                worker_file_pattern = f"{data_split_type}_{src}_w*_c*.npz"
                files_to_rename = sorted(list(source_dir_for_rename.glob(worker_file_pattern)))
                if files_to_rename:
                    logging.info(f"Renaming {len(files_to_rename)} files in {source_dir_for_rename} to final format.")
                    for i, old_f_path in enumerate(files_to_rename):
                        new_f_name = source_dir_for_rename / f"{src}_{i:07d}.npz"
                        try: old_f_path.rename(new_f_name)
                        except Exception as e_rename: logging.error(f"Error renaming {old_f_path} to {new_f_name}: {e_rename}")
    else: # Perform shuffle passes
        # ... (shuffle pass logic as before, operating on files in out_dir/train|test/...)
        logging.info("Proceeding with final disk-based shuffle pass(es).")
        for data_split_type in ["train", "test"]:
            for src in ("lichess", "lc0"):
                source_dir_for_shuffle = out_dir / data_split_type / src
                if not source_dir_for_shuffle.exists(): continue

                # Files for shuffling still have worker-generated names at this stage
                # Pattern: {data_split_type}_{src}_w{PID}_c{ID}.npz
                initial_file_pattern_for_shuffle = f"{data_split_type}_{src}_w*_c*.npz"
                files_in_dir_for_shuffling = list(source_dir_for_shuffle.glob(initial_file_pattern_for_shuffle))

                if len(files_in_dir_for_shuffling) > 1 and args.mix_passes > 0:
                    logging.info(f"Final mixing for {data_split_type}/{src}: {len(files_in_dir_for_shuffling)} files, {args.mix_passes} passes.")
                    for pass_n in range(args.mix_passes):
                        logging.info(f"Shuffle Pass {pass_n + 1}/{args.mix_passes} for {data_split_type}/{src}")
                        _simple_merge_shuffle_split_parallel(
                            source_dir_for_shuffle, args.seed + pass_n, shuffle_workers, 
                            args.shuffle_batch_size, args.benchmark, args.verify_shuffle, 
                            args.shuffle_worker_delay_max
                        )
                    # After shuffling, rename the (still worker-named but content-shuffled) files
                    # Re-glob with the same pattern as they retain original names after shuffle_split_worker
                    final_shuffled_files_to_rename = sorted(list(source_dir_for_shuffle.glob(initial_file_pattern_for_shuffle)))
                    if final_shuffled_files_to_rename:
                        logging.info(f"Renaming {len(final_shuffled_files_to_rename)} shuffled files in {source_dir_for_shuffle} to final format.")
                        for i, old_f_path in enumerate(final_shuffled_files_to_rename):
                            new_f_name = source_dir_for_shuffle / f"{src}_{i:07d}.npz"
                            try: old_f_path.rename(new_f_name)
                            except Exception as e_rename_shuf: logging.error(f"Error renaming {old_f_path} to {new_f_name}: {e_rename_shuf}")
                elif len(files_in_dir_for_shuffling) == 1:
                    logging.info(f"Only 1 file for {data_split_type}/{src}. Renaming for consistency.")
                    old_f_path = files_in_dir_for_shuffling[0]
                    new_f_name = source_dir_for_shuffle / f"{src}_0000000.npz"
                    if old_f_path != new_f_name: 
                        try: old_f_path.rename(new_f_name)
                        except Exception as e_rename_single: logging.error(f"Error renaming {old_f_path} to {new_f_name}: {e_rename_single}")
                else:
                    logging.info(f"No files or not enough files for shuffling {data_split_type}/{src}.")

    # --- Cleanup Intermediate Directory ---
    # Not automatically cleaning up intermediate_dir_path to preserve it for potential restarts 
    # when args.use_intermediate_dir or args.resume_from_intermediate was true.
    # Cleanup can be done manually or via a separate option if needed later.
    # if (args.use_intermediate_dir or args.resume_from_intermediate) and intermediate_dir_path.exists():
    #     logging.info(f"Cleaning up intermediate directory: {intermediate_dir_path}")
    #     _cleanup_intermediate_directory(intermediate_dir_path)

    # --- Metadata Generation ---
    logging.info("Generating final metadata by scanning output files in final directories...")
    final_counts_for_metadata = defaultdict(int)
    for data_s_type in ["train", "test"]:
        for src_type in ["lichess", "lc0"]:
            final_src_dir = out_dir / data_s_type / src_type
            if final_src_dir.exists() and final_src_dir.is_dir():
                renamed_final_files = list(final_src_dir.glob(f"{src_type}_*.npz"))
                num_chunks = 0
                num_positions = 0
                for f_path_meta in renamed_final_files:
                    try:
                        if not f_path_meta.exists(): continue
                        with np.load(f_path_meta) as data:
                            if "bitboards" in data and hasattr(data["bitboards"], 'shape') and data["bitboards"].ndim > 0:
                                num_positions += data["bitboards"].shape[0]
                        num_chunks += 1
                    except Exception as e_meta_load:
                        logging.warning(f"Could not read {f_path_meta} for metadata: {e_meta_load}")
                
                final_counts_for_metadata[f"{src_type}_{data_s_type}_positions"] = num_positions
                final_counts_for_metadata[f"{src_type}_{data_s_type}_chunks"] = num_chunks
                logging.info(f"Metadata count for {data_s_type}/{src_type}: {num_positions:,} positions, {num_chunks:,} chunks.")

    metadata = {
        "chunk_size": args.chunk_size,
        "total_lichess_train_positions": final_counts_for_metadata.get("lichess_train_positions", 0),
        "total_lc0_train_positions": final_counts_for_metadata.get("lc0_train_positions", 0),
        "total_lichess_test_positions": final_counts_for_metadata.get("lichess_test_positions", 0),
        "total_lc0_test_positions": final_counts_for_metadata.get("lc0_test_positions", 0),
        "lichess_train_chunks": final_counts_for_metadata.get("lichess_train_chunks", 0),
        "lc0_train_chunks": final_counts_for_metadata.get("lc0_train_chunks", 0),
        "lichess_test_chunks": final_counts_for_metadata.get("lichess_test_chunks", 0),
        "lc0_test_chunks": final_counts_for_metadata.get("lc0_test_chunks", 0),
        "mix_passes": args.mix_passes if not args.skip_final_shuffle_pass else 0,
        "mix_freq": args.mix_freq,
        "worker_buffer_multiplier": args.worker_buffer_multiplier,
        "final_shuffle_skipped": args.skip_final_shuffle_pass,
        "resumed_from_intermediate": args.resume_from_intermediate,
        "used_intermediate_directory_for_workers": args.use_intermediate_dir and not args.resume_from_intermediate,
        "pgn_processing_time_seconds": round(pgn_processing_elapsed_time, 1) if not args.resume_from_intermediate else 0.0,
        "total_raw_pgn_files_processed": all_pgn_files_count if not args.resume_from_intermediate else 0,
        "preprocessing_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Preprocessing finished. Manifest: %s", json.dumps(metadata, indent=2))
    final_script_time = time.time() - script_global_start_time 
    logging.info(f"Total script execution time: {final_script_time:.2f} seconds.")


if __name__ == "__main__":
    script_global_start_time = time.time()
    main()