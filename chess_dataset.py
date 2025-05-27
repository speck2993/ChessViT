"""
chess_dataset.py  (v2)
---------------------
High‑throughput **memory‑mapped** loader for Chess‑ViT *without* implicit
GPU copies – fully compatible with `DataLoader(pin_memory=True)`.

Key behavioural change (v2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The collate function now **keeps tensors on CPU** so that PyTorch's
  built‑in pin‑memory step can run.  GPU transfer (and flips) are deferred
  to the training step via a small helper `move_batch_to_device`.

Typical usage
-------------
```python
from chess_dataset import (
    MMapChunkDataset, chess_collate_fn, move_batch_to_device)
from torch.utils.data import DataLoader

train_loader = DataLoader(
    MMapChunkDataset("/data/chess_chunks", batch_size=512, seed=42),
    batch_size=None,            # dataset already chunks
    num_workers=6,
    pin_memory=True,            # now safe
    collate_fn=chess_collate_fn,
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
import gc # Import gc

# ---------------------------------------------------------------------------
# Utility — tensor flips (works on CPU *or* GPU because it's just torch.flip)
# ---------------------------------------------------------------------------

def flip_bitboards_torch(bitboards: torch.Tensor, mode: str = "v") -> torch.Tensor:
    """Return a view of *bitboards* flipped along the chosen axis."""
    if mode == "v":
        return bitboards.flip(-2)
    if mode == "h":
        return bitboards.flip(-1)
    if mode == "hv":
        return bitboards.flip((-2, -1))
    raise ValueError("mode must be 'v', 'h', or 'hv'")


# ---------------------------------------------------------------------------
# Dataset: memory‑mapped chunks → fixed‑size batches (CPU tensors)
# ---------------------------------------------------------------------------
class ChunkMmapDataset(IterableDataset):
    """Iterates over 4 096‑position ``.npz`` *chunks* with zero‑copy slicing."""

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

        # --- Cache Setup ---
        glob_hash = hashlib.md5(file_glob.encode('utf-8')).hexdigest()[:12]
        self.cache_file_path = self.root_dir / f".mmap_dataset_cache_{glob_hash}.json"

        # --- Load or Build File List and Position Count ---
        current_files_on_disk: List[Path] = sorted(self.root_dir.glob(file_glob))
        if not current_files_on_disk:
            raise FileNotFoundError(f"No files matching {file_glob} under {root_dir}")

        cached_data = self._load_cache(file_glob)

        if cached_data:
            # Validate cache
            cached_file_info_map = {Path(item['path']): item for item in cached_data['files']}
            
            # Check if the set of file paths is the same
            current_file_paths_set = {f.relative_to(self.root_dir).as_posix() for f in current_files_on_disk}
            cached_file_paths_set = {Path(item['path']).as_posix() for item in cached_data['files']}

            if current_file_paths_set == cached_file_paths_set:
                is_stale = False
                validated_file_list: List[Path] = []
                temp_total_positions = 0
                
                for f_path_obj in current_files_on_disk: # Iterate in sorted order of current files
                    relative_f_path_str = f_path_obj.relative_to(self.root_dir).as_posix()
                    # Find corresponding entry in cache (cache paths are already relative)
                    # The keys in cached_file_info_map are Path objects of relative paths.
                    cached_entry = cached_file_info_map.get(Path(relative_f_path_str))

                    if cached_entry and f_path_obj.stat().st_mtime == cached_entry['mtime']:
                        validated_file_list.append(f_path_obj)
                        temp_total_positions += cached_entry['positions']
                    else:
                        is_stale = True
                        print(f"INFO: ChunkMmapDataset - Cache is stale. File changed or missing in cache: {relative_f_path_str}")
                        break
                
                if not is_stale:
                    self.files = validated_file_list # Use the validated live Path objects
                    self.total_positions_in_dataset = temp_total_positions
                    print(f"INFO: ChunkMmapDataset - Loaded metadata for {len(self.files)} files and {self.total_positions_in_dataset} positions from cache: {self.cache_file_path}")
                    return # Successfully loaded from cache
            else:
                print("INFO: ChunkMmapDataset - Cache is stale due to change in file set.")
        
        print("INFO: ChunkMmapDataset - Cache not used or stale. Rebuilding file list and position count.")
        self.files = []
        self.total_positions_in_dataset = 0
        file_metadata_for_cache: List[Dict[str, Any]] = []

        # Parallel file loading
        print(f"INFO: ChunkMmapDataset - Starting parallel calculation of total positions from {len(current_files_on_disk)} files...")

        # Use ThreadPoolExecutor for parallel I/O
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._get_file_positions, f, self.root_dir): f 
                for f in current_files_on_disk
            }
            
            # Process completed tasks
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
        
        if not self.files: # This should be caught by the initial check on current_files_on_disk
             raise FileNotFoundError(f"No processable files matching {file_glob} under {self.root_dir} after attempting to count positions.")
        if self.total_positions_in_dataset == 0 and self.files:
            raise ValueError(f"Dataset contains files but total positions count is zero after processing. Check .npz files in {self.root_dir}")

    # --------------------------- cache helpers ----------------------------
    def _load_cache(self, current_file_glob: str) -> Any:
        if self.cache_file_path.exists():
            try:
                with open(self.cache_file_path, 'r') as f:
                    data = json.load(f)
                if data.get('file_glob') == current_file_glob:
                    return data
                else:
                    print(f"INFO: ChunkMmapDataset - Cache file found but for a different glob pattern ('{data.get('file_glob')}' vs '{current_file_glob}'). Ignoring cache.")
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
            
    # --------------------------- helpers ----------------------------------
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
                # Yield existing keys plus the new identifiers
                batch_data = {k: npz[k][start:end] for k in keys}
                batch_data['source_file_basename'] = path.name
                batch_data['original_indices_in_file'] = np.arange(start, end, dtype=np.int64) # Ensure consistent dtype
                yield batch_data

    # ---------------------- IterableDataset API ---------------------------
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
    """Flexible collate.

    * If the *DataLoader* was constructed with ``batch_size=None`` (the
      recommended, fastest path) then ``sample_list`` has length **1** and
      already holds a *batched* dict from :class:`MMapChunkDataset`.

    * If the user passed a numeric ``batch_size`` to the DataLoader (e.g.
      during quick benchmarking), ``sample_list`` is a *list of dicts* and
      we concatenate along axis‑0.
    """
    # Fast path: dataset already batched
    # ------------------------------------------------------------
    # 1) Dataset already produced a batch (sample_list is a dict)
    # ------------------------------------------------------------
    if isinstance(sample_list, dict):
        np_dict: Dict[str, np.ndarray] = sample_list
    # ------------------------------------------------------------
    # 2) DataLoader gave us a list of dicts (it was given a real
    #    batch_size > 1).  Merge along axis-0.
    # ------------------------------------------------------------
    elif len(sample_list) == 1:
        np_dict = sample_list[0]
    else:  # User let DataLoader do the batching – merge manually
        keys = sample_list[0].keys()
        # Handle regular keys first
        np_dict = {k: np.concatenate([d[k] for d in sample_list if k not in ('source_file_basename', 'original_indices_in_file')]) for k in keys if k not in ('source_file_basename', 'original_indices_in_file')}
        # Special handling for new keys if DataLoader did batching (less common for this dataset struct)
        if 'source_file_basename' in keys:
            # This would become a list of basenames if batched by DataLoader. For simplicity, assume not the primary path.
            np_dict['source_file_basename'] = [d['source_file_basename'] for d in sample_list]
        if 'original_indices_in_file' in keys:
            np_dict['original_indices_in_file'] = np.concatenate([d['original_indices_in_file'] for d in sample_list], axis=0)

    out: Dict[str, torch.Tensor] = {}
    for key, arr in np_dict.items():
        # Handle non-numpy array types first
        if key == 'source_file_basename': # Keep as is (str or list of str)
            out[key] = arr
            continue
        
        # Then handle FEN strings (which are numpy arrays of strings)
        if arr.dtype.type is np.str_:
            out[key] = arr  # keep FEN strings on CPU as NumPy
            continue
        # if key == 'source_file_basename': # Keep as is (str or list of str)
        #     out[key] = arr
        #     continue

        # ---------- type-safety before torch wraps the array ----------
        # PyTorch supports uint8 but *no* other unsigned ints.
        if arr.dtype == np.uint16:
            # np → signed-int16 view (no copy if allowed by alignment)
            arr = arr.astype(np.int16, copy=False)
        elif arr.dtype == np.uint32:
            arr = arr.astype(np.int32, copy=False)
        elif arr.dtype == np.uint64:
            arr = arr.astype(np.int64, copy=False)

        t = torch.from_numpy(arr)
        # dtype refinement ---------------------------
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
        # else keep original (int8 flags)
        # Ensure tensors are allocated in pinned (page-locked) memory so that the
        # subsequent H→D copy (triggered by DataLoader with ``pin_memory=True``)
        # becomes purely asynchronous and does **not** allocate a second buffer.
        # If the tensor is already pinned this is a no-op; otherwise `.pin_memory()`
        # returns a pinned clone.
        if not t.is_pinned():
            t = t.pin_memory()

        out[key] = t

    # Explicitly delete the dictionary holding NumPy array views after conversion to tensors.
    # The PyTorch tensors in 'out' may share memory with these views if torch.from_numpy() allowed.
    # This helps signal that these specific Python references are no longer needed by this function.
    del np_dict
    # gc.collect() # Generally not recommended here as it might be too frequent and slow down data loading.

    return out


# ---------------------------------------------------------------------------
# Helper – move batch to device *and* add flipped variants lazily
# ---------------------------------------------------------------------------

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device, *, non_blocking: bool = True):
    """Move all tensor values to device and create flipped bitboards using views where possible."""
    out: Dict[str, torch.Tensor] = {}
    
    # Move all tensors to device first
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            target_dtype = None
            # Down-cast certain large float tensors (primarily bitboards) to fp16 when on CUDA to save memory/bandwidth
            if device.type == "cuda" and v.dtype in {torch.float32, torch.float64} and k == "bitboards":
                target_dtype = torch.float16
            out[k] = v.to(device, dtype=target_dtype, non_blocking=non_blocking) if target_dtype is not None else v.to(device, non_blocking=non_blocking)
        else:  # FEN strings, source_file_basename etc.
            out[k] = v
    
    # Create flipped variants on GPU (these are views when possible)
    bb = out.get("bitboards")
    if bb is not None and bb.device.type == "cuda":
        # Use flip which can sometimes create views instead of copies
        out["bitboards_v"] = torch.flip(bb, dims=[-2])
        out["bitboards_h"] = torch.flip(bb, dims=[-1])
        out["bitboards_hv"] = torch.flip(bb, dims=[-2, -1])
    
    return out
