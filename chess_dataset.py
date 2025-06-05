"""
chess_dataset.py  (v3)
---------------------
High‑throughput **memory‑mapped** loader for Chess‑ViT *without* contrastive learning.
Simplified version without flipped bitboard generation.

Typical usage
-------------
```python
from chess_dataset import (
    ChunkMmapDataset, fast_chess_collate_fn, move_batch_to_device)
from torch.utils.data import DataLoader

train_loader = DataLoader(
    ChunkMmapDataset("/data/chess_chunks", batch_size=512, seed=42),
    batch_size=None,
    num_workers=6,
    pin_memory=True,
    collate_fn=fast_chess_collate_fn,
    persistent_workers=True,
    prefetch_factor=2,
)

for cpu_batch in train_loader:
    batch = move_batch_to_device(cpu_batch, torch.device("cuda"))
    loss = model(batch)
    ...
```
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Any
import json
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import gc

# ---------------------------------------------------------------------------
# Dataset: memory‑mapped chunks → fixed‑size batches (CPU tensors)
# ---------------------------------------------------------------------------
class ChunkMmapDataset(IterableDataset):
    """Iterates over 4 096‑position ``.npz`` *chunks* with zero‑copy slicing."""

    @staticmethod
    def _get_file_positions(file_path, root_dir):
        """Helper to get position count from a single file."""
        try:
            num_positions = 0
            with np.load(file_path, mmap_mode="r", allow_pickle=False) as npz:
                if npz.files:
                    num_positions = npz[npz.files[0]].shape[0]
            return {
                'path': file_path.relative_to(root_dir).as_posix(),
                'mtime': file_path.stat().st_mtime,
                'positions': num_positions,
                'file_obj': file_path
            }
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return None

    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        *,
        file_glob: str = "*.npz",
        shuffle_files: bool = True,
        infinite: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(root_dir)

        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.infinite = infinite
        self._rng = random.Random(seed)

        # Cache Setup
        glob_hash = hashlib.md5(file_glob.encode('utf-8')).hexdigest()[:12]
        self.cache_file_path = self.root_dir / f".mmap_dataset_cache_{glob_hash}.json"

        # Load or Build File List and Position Count
        current_files_on_disk: List[Path] = sorted(self.root_dir.glob(file_glob))
        if not current_files_on_disk:
            raise FileNotFoundError(f"No files matching {file_glob} under {root_dir}")

        cached_data = self._load_cache(file_glob)

        if cached_data:
            # Validate cache
            cached_file_info_map = {Path(item['path']): item for item in cached_data['files']}
            
            current_file_paths_set = {f.relative_to(self.root_dir).as_posix() for f in current_files_on_disk}
            cached_file_paths_set = {Path(item['path']).as_posix() for item in cached_data['files']}

            if current_file_paths_set == cached_file_paths_set:
                is_stale = False
                validated_file_list: List[Path] = []
                temp_total_positions = 0
                
                for f_path_obj in current_files_on_disk:
                    relative_f_path_str = f_path_obj.relative_to(self.root_dir).as_posix()
                    cached_entry = cached_file_info_map.get(Path(relative_f_path_str))

                    if cached_entry and f_path_obj.stat().st_mtime == cached_entry['mtime']:
                        validated_file_list.append(f_path_obj)
                        temp_total_positions += cached_entry['positions']
                    else:
                        is_stale = True
                        print(f"INFO: ChunkMmapDataset - Cache is stale. File changed or missing in cache: {relative_f_path_str}")
                        break
                
                if not is_stale:
                    self.files = validated_file_list
                    self.total_positions_in_dataset = temp_total_positions
                    print(f"INFO: ChunkMmapDataset - Loaded metadata for {len(self.files)} files and {self.total_positions_in_dataset} positions from cache: {self.cache_file_path}")
                    return
            else:
                print("INFO: ChunkMmapDataset - Cache is stale due to change in file set.")
        
        print("INFO: ChunkMmapDataset - Cache not used or stale. Rebuilding file list and position count.")
        self.files = []
        self.total_positions_in_dataset = 0
        file_metadata_for_cache: List[Dict[str, Any]] = []

        print(f"INFO: ChunkMmapDataset - Starting parallel calculation of total positions from {len(current_files_on_disk)} files...")

        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            future_to_file = {
                executor.submit(self._get_file_positions, f, self.root_dir): f 
                for f in current_files_on_disk
            }
            
            completed = 0
            for future in as_completed(future_to_file):
                result = future.result()
                if result:
                    self.files.append(result['file_obj'])
                    self.total_positions_in_dataset += result['positions']
                    file_metadata_for_cache.append({
                        'path': result['path'],
                        'mtime': result['mtime'],
                        'positions': result['positions']
                    })
                
                completed += 1
                if completed % 100 == 0 or completed == len(current_files_on_disk):
                    print(f"INFO: ChunkMmapDataset - Processed {completed}/{len(current_files_on_disk)} files. "
                          f"Current total: {self.total_positions_in_dataset}", flush=True)
        
        self._save_cache(file_glob, file_metadata_for_cache, self.total_positions_in_dataset)
        print(f"INFO: ChunkMmapDataset - Finished calculation. Total positions in dataset: {self.total_positions_in_dataset}")
        
        if not self.files:
             raise FileNotFoundError(f"No processable files matching {file_glob} under {self.root_dir} after attempting to count positions.")
        if self.total_positions_in_dataset == 0 and self.files:
            raise ValueError(f"Dataset contains files but total positions count is zero after processing. Check .npz files in {self.root_dir}")

    def _load_cache(self, current_file_glob: str) -> Any:
        if self.cache_file_path.exists():
            try:
                with open(self.cache_file_path, 'r') as f:
                    data = json.load(f)
                if data.get('file_glob') == current_file_glob:
                    return data
                else:
                    print(f"INFO: ChunkMmapDataset - Cache file found but for a different glob pattern. Ignoring cache.")
            except json.JSONDecodeError:
                print(f"Warning: ChunkMmapDataset - Error decoding cache file {self.cache_file_path}. Ignoring cache.")
            except Exception as e:
                print(f"Warning: ChunkMmapDataset - Error loading cache file {self.cache_file_path}: {e}. Ignoring cache.")
        return None

    def _save_cache(self, file_glob: str, file_info: List[Dict[str, Any]], total_positions: int):
        try:
            with open(self.cache_file_path, 'w') as f:
                json.dump({
                    'file_glob': file_glob,
                    'total_positions_in_dataset': total_positions,
                    'files': file_info
                }, f, indent=4)
            print(f"INFO: ChunkMmapDataset - Saved metadata cache to {self.cache_file_path}")
        except Exception as e:
            print(f"Warning: ChunkMmapDataset - Error saving cache file {self.cache_file_path}: {e}")
            
    def _iter_files(self) -> List[Path]:
        lst = list(self.files)
        if self.shuffle_files:
            self._rng.shuffle(lst)
        return lst

    def _yield_batches_from(self, path: Path):
        with np.load(path, mmap_mode="r", allow_pickle=False) as npz:
            keys = npz.files
            n = npz[keys[0]].shape[0]
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                if end > n:
                    break  # keep static shapes
                batch_data = {k: npz[k][start:end] for k in keys}
                batch_data['source_file_basename'] = path.name
                batch_data['original_indices_in_file'] = np.arange(start, end, dtype=np.int64)
                yield batch_data

    def __iter__(self):
        worker = get_worker_info()
        files = self._iter_files()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
            self._rng = random.Random(self._rng.randint(0, 2**32 - 1))

        while True:
            for f in files:
                for batch in self._yield_batches_from(f):
                    yield batch
            if not self.infinite:
                break
            if self.shuffle_files:
                self._rng.shuffle(files)


# ---------------------------------------------------------------------------
# Collate – NumPy ➜ **CPU** torch.Tensors (no device copies here!)
# ---------------------------------------------------------------------------

def fast_chess_collate_fn(sample_list):
    """Flexible collate function for chess data."""
    if isinstance(sample_list, dict):
        np_dict: Dict[str, np.ndarray] = sample_list
    elif len(sample_list) == 1:
        np_dict = sample_list[0]
    else:
        keys = sample_list[0].keys()
        np_dict = {k: np.concatenate([d[k] for d in sample_list if k not in ('source_file_basename', 'original_indices_in_file')]) for k in keys if k not in ('source_file_basename', 'original_indices_in_file')}
        if 'source_file_basename' in keys:
            np_dict['source_file_basename'] = [d['source_file_basename'] for d in sample_list]
        if 'original_indices_in_file' in keys:
            np_dict['original_indices_in_file'] = np.concatenate([d['original_indices_in_file'] for d in sample_list], axis=0)

    out: Dict[str, torch.Tensor] = {}
    
    # Check if we're in a worker process
    worker_info = get_worker_info()
    in_worker_process = worker_info is not None
    
    for key, arr in np_dict.items():
        if key == 'source_file_basename':
            out[key] = arr
            continue
        
        if arr.dtype.type is np.str_:
            out[key] = arr  # keep FEN strings as NumPy
            continue

        # Type-safety before torch wraps the array
        if arr.dtype == np.uint16:
            arr = arr.astype(np.int16, copy=False)
        elif arr.dtype == np.uint32:
            arr = arr.astype(np.int32, copy=False)
        elif arr.dtype == np.uint64:
            arr = arr.astype(np.int64, copy=False)

        t = torch.from_numpy(arr)
        
        # dtype refinement
        if key in {"bitboards", "policy", "policy_target"}:
            t = t.float()
        elif key == "legal_mask":
            t = t.bool()
        elif key == "value_target":
            t = t.long()
        elif key == "material_category":
            t = t.long()
        elif key == "ply_target":
            t = t.short()

        # Only pin memory if not in worker process and tensor is not already pinned
        if not in_worker_process and not t.is_pinned():
            try:
                t = t.pin_memory()
            except RuntimeError as e:
                # Handle CUDA errors gracefully (e.g., CUDA not available or busy)
                if "CUDA" in str(e):
                    # Skip pinning if CUDA is not available or busy
                    pass
                else:
                    # Re-raise if it's a different kind of RuntimeError
                    raise

        out[key] = t

    del np_dict
    return out


# ---------------------------------------------------------------------------
# Helper – move batch to device (simplified without flipped variants)
# ---------------------------------------------------------------------------

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device, *, non_blocking: bool = True):
    """Move all tensor values to device (simplified without flipped bitboards)."""
    out: Dict[str, torch.Tensor] = {}
    
    # Move all tensors to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:  # FEN strings, source_file_basename etc.
            out[k] = v
    
    return out
