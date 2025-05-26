import os
import glob
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import Counter
import sys # Import sys for progress bar
import gc # Import gc for garbage collection
import concurrent.futures # Import for ThreadPoolExecutor

# --- Configuration Constants ---
NUM_FILES_TO_MOVE_DEFAULT = 3
# NUM_TRAIN_SAMPLE_FILES_DEFAULT = 8 # Old
# NUM_POSITIONS_TO_SAMPLE_PER_TRAIN_FILE_DEFAULT = 200 # Old
NUM_TRAIN_SRS_POSITIONS_DEFAULT = 500 # New: Total number of SRS positions
NUM_POLICY_PLANES = 73
WDL_LABELS = ['Loss', 'Draw', 'Win']
DEFAULT_SRS_WORKERS = min(4, os.cpu_count() // 2 if os.cpu_count() else 1) # Default SRS workers
if DEFAULT_SRS_WORKERS == 0: DEFAULT_SRS_WORKERS = 1

# --- Helper Functions ---

def ensure_dir_exists(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_npz_files(directory):
    return list(Path(directory).glob('*.npz'))

def move_files_to_test_set(source_dir_str, test_dir_str, num_files_to_move):
    source_dir = Path(source_dir_str)
    test_dir = Path(test_dir_str)

    ensure_dir_exists(test_dir)
    
    available_files = get_npz_files(source_dir)
    if not available_files:
        print(f"No .npz files found in {source_dir}. Skipping file moving for this source.")
        return []

    num_to_select = min(len(available_files), num_files_to_move)
    if num_to_select < num_files_to_move:
        print(f"Warning: Only {num_to_select} files available in {source_dir}, requested {num_files_to_move}. Moving all available.")
    
    files_to_move_paths = random.sample(available_files, num_to_select)
    moved_files_in_test = []

    print(f"Moving {num_to_select} files from {source_dir} to {test_dir}...")
    for file_path in files_to_move_paths:
        dest_path = test_dir / file_path.name
        try:
            shutil.move(str(file_path), str(dest_path))
            moved_files_in_test.append(dest_path)
        except Exception as e:
            print(f"  Error moving {file_path.name}: {e}")
    print(f"Moved {len(moved_files_in_test)} files to {test_dir}.")
    return moved_files_in_test

def load_data_from_files(file_list, max_positions=None):
    all_value_targets = []
    all_policy_targets = []
    positions_loaded = 0
    for npz_file in file_list:
        if max_positions is not None and positions_loaded >= max_positions:
            break
        try:
            with np.load(npz_file, allow_pickle=False) as data:
                if 'value_target' not in data or 'policy_target' not in data:
                    print(f"Warning: {npz_file} missing 'value_target' or 'policy_target'. Skipping.")
                    continue
                value_targets_in_file = data['value_target']
                policy_targets_in_file = data['policy_target']
                current_batch_size = value_targets_in_file.shape[0]
                if max_positions is not None:
                    remaining_to_load = max_positions - positions_loaded
                    if current_batch_size > remaining_to_load:
                        indices = np.random.choice(current_batch_size, remaining_to_load, replace=False)
                        value_targets_in_file = value_targets_in_file[indices]
                        policy_targets_in_file = policy_targets_in_file[indices]
                        current_batch_size = remaining_to_load
                all_value_targets.append(value_targets_in_file)
                all_policy_targets.append(policy_targets_in_file)
                positions_loaded += current_batch_size
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue
    if not all_value_targets or not all_policy_targets:
        return None, None
    return np.concatenate(all_value_targets), np.concatenate(all_policy_targets)

def get_distributions(value_targets, policy_targets):
    if value_targets is None or policy_targets is None:
        return None, None
    wdl_counts = Counter(value_targets)
    total_wdl = sum(wdl_counts.values())
    wdl_dist = {label: wdl_counts.get(i, 0) / total_wdl if total_wdl > 0 else 0 
                for i, label in enumerate(WDL_LABELS)}
    if policy_targets.ndim == 4:
        plane_activations = np.sum(policy_targets, axis=(2, 3))
    elif policy_targets.ndim == 2 and policy_targets.shape[1] == NUM_POLICY_PLANES:
        plane_activations = policy_targets
    else:
        print(f"Warning: Unexpected policy_target shape {policy_targets.shape}. Cannot compute plane distribution.")
        return wdl_dist, None
    total_plane_usage = np.sum(plane_activations, axis=0)
    sum_of_all_plane_usage = np.sum(total_plane_usage)
    policy_plane_dist = total_plane_usage / sum_of_all_plane_usage if sum_of_all_plane_usage > 0 else np.zeros_like(total_plane_usage)
    return wdl_dist, policy_plane_dist

def _sample_one_srs_position(train_file_list):
    """Worker function to sample a single position for SRS. Includes memory cleanup."""
    selected_file_path = random.choice(train_file_list)
    
    value_sample_copy = None
    policy_sample_copy = None
    data_arrays = {}

    try:
        with np.load(selected_file_path, allow_pickle=False) as data_loaded:
            if 'value_target' not in data_loaded or 'policy_target' not in data_loaded:
                return None, None # Signal an issue with this specific sample attempt
            
            data_arrays['value_target'] = data_loaded['value_target']
            data_arrays['policy_target'] = data_loaded['policy_target']

        num_positions_in_file = data_arrays['value_target'].shape[0]
        if num_positions_in_file == 0:
            return None, None
        
        selected_idx_in_file = random.randint(0, num_positions_in_file - 1)
        
        value_sample_copy = data_arrays['value_target'][selected_idx_in_file].copy()
        policy_sample_copy = data_arrays['policy_target'][selected_idx_in_file].copy()
        
        return value_sample_copy, policy_sample_copy

    except Exception as e:
        # Print error on a new line so it doesn't mess up the progress bar (if used by caller)
        # sys.stdout.write('\r' + ' ' * 80 + '\r') 
        # print(f"\nError in _sample_one_srs_position for {selected_file_path}: {e}")
        return None, None # Signal error for this sample
    finally:
        if 'value_target' in data_arrays:
            del data_arrays['value_target']
        if 'policy_target' in data_arrays:
            del data_arrays['policy_target']
        # Copies are returned, original samples in worker are not needed beyond this point
        gc.collect()

def get_train_srs_distributions(train_data_dir_str, num_srs_positions, num_workers):
    """Samples individual positions memorylessly from the training set using ThreadPoolExecutor."""
    train_dir = Path(train_data_dir_str)
    available_files = get_npz_files(train_dir)

    if not available_files:
        print(f"No .npz files found in {train_dir} for SRS. Skipping train sample analysis.")
        return None, None

    all_value_samples = []
    all_policy_samples = []

    print(f"Sampling {num_srs_positions} SRS positions from {train_dir} using {num_workers} workers...")
    progress_bar_length = 40
    completed_tasks = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_sample_one_srs_position, available_files) for _ in range(num_srs_positions)]
        
        for future in concurrent.futures.as_completed(futures):
            value_sample, policy_sample = future.result()
            if value_sample is not None and policy_sample is not None:
                all_value_samples.append(value_sample)
                all_policy_samples.append(policy_sample)
            
            completed_tasks += 1
            percent_done = completed_tasks / num_srs_positions
            filled_length = int(progress_bar_length * percent_done)
            bar = '=' * filled_length + '-' * (progress_bar_length - filled_length)
            sys.stdout.write(f'\r  SRS Sampling [{bar}] {percent_done:.1%} ({completed_tasks}/{num_srs_positions}) positions...')
            sys.stdout.flush()

    sys.stdout.write('\r' + ' ' * (progress_bar_length + 50) + '\r')
    sys.stdout.flush()
    print(f"Finished SRS sampling.")

    if not all_value_samples or not all_policy_samples:
        print("No SRS samples collected (or too many errors). Cannot compute distributions.")
        return None, None
    
    sampled_values = np.array(all_value_samples)
    try:
        sampled_policies = np.stack(all_policy_samples)
    except Exception as e:
        print(f"Error stacking policy samples for SRS: {e}. Shapes might be inconsistent.")
        return get_distributions(sampled_values, None)

    print(f"Collected {len(sampled_values)} valid SRS samples.")
    return get_distributions(sampled_values, sampled_policies)

def plot_comparison(dist1, dist2, title1, title2, overall_title, xticklabels, output_filename):
    if dist1 is None or dist2 is None:
        print(f"Skipping plot {overall_title} due to missing data.")
        return
    n_groups = len(xticklabels)
    fig, ax = plt.subplots(figsize=(max(10, n_groups * 0.5), 6))
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    if isinstance(dist1, dict):
        vals1 = [dist1.get(label, 0) for label in xticklabels] if xticklabels != WDL_LABELS else [dist1.get(wdl_label,0) for wdl_label in WDL_LABELS]
    else:
        vals1 = dist1
    if isinstance(dist2, dict):
        vals2 = [dist2.get(label, 0) for label in xticklabels] if xticklabels != WDL_LABELS else [dist2.get(wdl_label,0) for wdl_label in WDL_LABELS]
    else:
        vals2 = dist2
    rects1 = ax.bar(index, vals1, bar_width, alpha=opacity, label=title1)
    rects2 = ax.bar(index + bar_width, vals2, bar_width, alpha=opacity, label=title2)
    ax.set_xlabel('Category')
    ax.set_ylabel('Proportion')
    ax.set_title(overall_title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--')
    fig.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"Saved plot: {output_filename}")

# --- Main Script Logic ---
def main(args):
    base_data_dir = Path(args.base_data_dir)
    base_test_dir = Path(args.base_test_dir)
    output_plot_dir = Path(args.output_plot_dir)
    ensure_dir_exists(output_plot_dir)

    sources = ["lichess", "lc0"]
    results = {}

    for source in sources:
        print(f"\nProcessing source: {source.upper()}")
        source_data_path = base_data_dir / source
        source_test_path = base_test_dir / source

        # 1. Create test set files
        moved_files = move_files_to_test_set(str(source_data_path), str(source_test_path), args.num_files_to_move)
        if not moved_files:
            print(f"No files moved to test set for {source}. Analysis might be limited.")

        # 2. Analyze distributions for the new test set
        print(f"Analyzing test set for {source}...")
        test_values, test_policies = load_data_from_files(moved_files)
        wdl_dist_test, policy_dist_test = get_distributions(test_values, test_policies)
        results[source] = {
            'wdl_test': wdl_dist_test,
            'policy_test': policy_dist_test
        }
        if wdl_dist_test:
            print(f"  Test WDL ({source}): { {k: f'{v:.3f}' for k,v in wdl_dist_test.items()} }")
        if policy_dist_test is not None:
            print(f"  Test Policy Plane Sums first 5 ({source}): {[f'{x:.3f}' for x in policy_dist_test[:5]]}")

        # 3. Analyze distributions for a sample from the remaining training set (SRS)
        print(f"Analyzing training set SRS for {source}...")
        wdl_dist_train_srs, policy_dist_train_srs = get_train_srs_distributions(
            str(source_data_path),
            args.num_train_srs_positions,
            args.srs_workers # Pass the number of workers
        )
        results[source]['wdl_train_srs'] = wdl_dist_train_srs
        results[source]['policy_train_srs'] = policy_dist_train_srs
        if wdl_dist_train_srs:
            print(f"  Train SRS WDL ({source}): { {k: f'{v:.3f}' for k,v in wdl_dist_train_srs.items()} }")
        if policy_dist_train_srs is not None:
            print(f"  Train SRS Policy Plane Sums first 5 ({source}): {[f'{x:.3f}' for x in policy_dist_train_srs[:5]]}")

        # 4. Plot comparisons
        if results[source]['wdl_test'] and results[source]['wdl_train_srs']:
            plot_comparison(
                results[source]['wdl_test'], results[source]['wdl_train_srs'],
                f'{source.capitalize()} Test Set', f'{source.capitalize()} Train SRS',
                f'WDL Distribution Comparison - {source.capitalize()}',
                WDL_LABELS,
                output_plot_dir / f'{source}_wdl_comparison.png'
            )
        
        if results[source]['policy_test'] is not None and results[source]['policy_train_srs'] is not None:
            policy_plane_labels = [f'P{i}' for i in range(NUM_POLICY_PLANES)]
            plot_comparison(
                results[source]['policy_test'], results[source]['policy_train_srs'],
                f'{source.capitalize()} Test Set', f'{source.capitalize()} Train SRS',
                f'Policy Plane Usage Comparison - {source.capitalize()}',
                policy_plane_labels,
                output_plot_dir / f'{source}_policy_plane_comparison.png'
            )
    
    print("\nScript finished. Plots saved in:", output_plot_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a test split from data and analyze distributions.")
    parser.add_argument('--base-data-dir', type=str, default='data', help='Base directory containing source data (e.g., data/lichess, data/lc0)')
    parser.add_argument('--base-test-dir', type=str, default='test', help='Base directory to create test splits (e.g., test/lichess, test/lc0)')
    parser.add_argument('--output-plot-dir', type=str, default='test_set_analysis_plots', help='Directory to save comparison plots.')
    parser.add_argument('--num-files-to-move', type=int, default=NUM_FILES_TO_MOVE_DEFAULT, help='Number of .npz files to move to the test set from each source.')
    # parser.add_argument('--num-train-sample-files', type=int, default=NUM_TRAIN_SAMPLE_FILES_DEFAULT, help='Number of files to sample from the training set for comparison.') # Old
    # parser.add_argument('--num-positions-per-train-file', type=int, default=NUM_POSITIONS_TO_SAMPLE_PER_TRAIN_FILE_DEFAULT, help='Number of positions to sample from each training file for comparison.') # Old
    parser.add_argument('--num-train-srs-positions', type=int, default=NUM_TRAIN_SRS_POSITIONS_DEFAULT, help='Total number of Simple Random Sample positions from the training set for comparison.')
    parser.add_argument('--srs-workers', type=int, default=DEFAULT_SRS_WORKERS, help='Number of worker threads for SRS sampling.')
    
    args = parser.parse_args()
    main(args) 