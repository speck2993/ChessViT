import os
import sys
import tempfile
import shutil
import numpy as np
import yaml
import subprocess
from pathlib import Path

def create_dummy_npz_file(filepath: Path, batch_size: int, num_policy_planes: int, is_lichess: bool):
    """Creates a dummy NPZ file with pre-batched data."""
    
    # Keys and dtypes based on DTYPE_MAP in preprocess_tensors.py
    data = {
        'bitboards': np.random.rand(batch_size, 14, 8, 8).astype(np.float16),
        'policy_target': np.zeros((batch_size, num_policy_planes, 8, 8), dtype=np.float16),
        'legal_mask': np.random.choice([True, False], size=(batch_size, num_policy_planes, 8, 8), p=[0.1, 0.9]).astype(bool),
        'value_target': np.random.randint(0, 3, size=(batch_size,), dtype=np.int8), # WDL: 0=L, 1=D, 2=W
        'material_category': np.random.randint(0, 20, size=(batch_size,), dtype=np.uint8),
        'ply_target': np.random.randint(1, 100, size=(batch_size,), dtype=np.uint16),
        'is_960_game': np.random.choice([False, True], size=(batch_size,)).astype(bool),
        'is_lichess_game': np.full((batch_size,), is_lichess, dtype=bool),
        'side_to_move': np.random.choice([False, True], size=(batch_size,)).astype(bool), # False=Black, True=White
        'rep_count_ge1': np.random.choice([False, True], size=(batch_size,), p=[0.8, 0.2]).astype(bool),
        'rep_count_ge2': np.random.choice([False, True], size=(batch_size,), p=[0.9, 0.1]).astype(bool),
        'material_raw': np.random.randint(-9, 10, size=(batch_size,), dtype=np.int8),
        'is_lc0_game': np.full((batch_size,), not is_lichess, dtype=bool),
        'total_plys_in_game': np.random.randint(20, 200, size=(batch_size,), dtype=np.int16),
    }

    # Make at least one policy target valid per sample
    for i in range(batch_size):
        p_idx = np.random.randint(0, num_policy_planes)
        r_idx = np.random.randint(0, 8)
        f_idx = np.random.randint(0, 8)
        data['policy_target'][i, p_idx, r_idx, f_idx] = 1.0
        data['legal_mask'][i, p_idx, r_idx, f_idx] = True

    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, **data)
    print(f"Created dummy NPZ: {filepath}")

def create_dummy_dist_init(filepath: Path):
    """Creates a dummy distance matrix file."""
    bias = np.zeros((65, 65), dtype=np.float32)
    for i in range(1, 65):
        for j in range(1, 65):
            if i == j:
                bias[i, j] = 0.0
            else:
                bias[i, j] = -1.0 / (1.0 + abs(i - j) * 0.1) 
    np.save(filepath, bias)
    print(f"Created dummy dist_init: {filepath}")

def create_dummy_config(filepath: Path, data_dir: Path, dist_init_path: Path, output_dir: Path, batch_size: int):
    """Creates a minimal dummy YAML config file."""
    
    test_device = 'cpu'
    test_precision = 'fp32'
    try:
        import torch
        if torch.cuda.is_available():
            test_device = 'cuda'
            test_precision = 'fp16' # Test fp16 path if CUDA is available
            print("INFO (test_training_pipeline): CUDA available, dummy config will use 'cuda' and 'fp16'.")
        else:
            print("INFO (test_training_pipeline): CUDA not available, dummy config will use 'cpu' and 'fp32'.")
    except ImportError:
        print("WARNING (test_training_pipeline): PyTorch not found, defaulting device to 'cpu', precision 'fp32'.")

    config = {
        'model': {
            'dim': 32, 'depth': 2, 'early_depth': 1, 'heads': 2,
            'distance_matrix_path': str(dist_init_path.resolve()),
            'freeze_distance_iters': 5, # Low value for test
            'pool_every_k_blocks': None, 'cls_dropout_rate': 0.0,
            'cls_pool_alpha_init': 1.0, 'cls_pool_alpha_requires_grad': False,
            'policy_head_conv_dim': 32, 'policy_head_mlp_hidden_dim': 32, # This is used by ViTChess policy_mlp_hidden_dim and policy_head_mlp_hidden_dim
            'num_policy_planes': 73, 'num_value_outputs': 3, 'num_material_categories': 20,
            'num_moves_left_outputs': 1, 'policy_cls_projection_dim': 32,
            'value_head_mlp_hidden_dim': 32, # Added for the new MLP value head
            'drop_path': 0.0, 'dim_head': None,
        },
        'loss_weights': { 
            'policy': 1.0, 'value': 1.0, 'moves_left': 0.1,
            'auxiliary_value': 0.1, 'material': 0.1, 'contrastive': 0.0, # Disable contrastive
            'nt_xent_temperature': 0.1,
        },
        'optimiser': {
            'type': 'adamw', 'lr': 1e-5, 'weight_decay': 1e-4,
            'betas': [0.9, 0.95], 'sched': 'cosine', 'warmup_steps': 1,
        },
        'dataset': {
            'data_dir': str(data_dir.resolve()), 'batch_size': batch_size,
            'grad_accum': 1, 'num_workers': 0, 'flips': True, 
            'type': 'tensor', 'tensor_glob_pattern': '**/*.npz',
            'test_data_dir': str(data_dir.resolve()), # Use same data for val
        },
        'runtime': {
            'device': test_device, 'precision': test_precision, 'grad_accum': 1,
            'max_steps': 3, 'log_every': 1, 'ckpt_every': 10, 'val_every': 2,
            'checkpoint_format': 'safetensors', 'gradient_clip_norm': 1.0,
            'compile_model': False, # Disable torch.compile for basic test
        },
        'logging': {
            'output_dir': str(output_dir.resolve()),
            'tensorboard': False, 'matplotlib': False, 'wandb': False,
        },
        'rolling_metrics': {'window_size': 5},
        'seed': 42
    }
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Created dummy config: {filepath}")

def main():
    # Ensure train.py is findable if test script is not in root
    current_script_dir = Path(__file__).parent
    train_script_path = (current_script_dir / "train.py").resolve()
    if not train_script_path.exists():
        # Fallback if __file__ is not reliable (e.g. interactive session)
        train_script_path = Path("train.py").resolve() 
    if not train_script_path.exists():
        print(f"ERROR: train.py not found at {train_script_path} or ./train.py")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"Using temporary directory: {temp_dir}")

        dummy_data_root = temp_dir / "dummy_tensor_data"
        dummy_output_dir = temp_dir / "dummy_training_output"
        dummy_dist_path = temp_dir / "dummy_dist_init.npy"
        dummy_config_path = temp_dir / "dummy_config.yaml"

        batch_size_for_test = 2
        num_policy_planes_for_test = 73 

        create_dummy_npz_file(dummy_data_root / "lichess" / "dummy_lichess_chunk_00.npz", batch_size_for_test, num_policy_planes_for_test, is_lichess=True)
        create_dummy_npz_file(dummy_data_root / "lichess" / "dummy_lichess_chunk_01.npz", batch_size_for_test, num_policy_planes_for_test, is_lichess=True)
        create_dummy_npz_file(dummy_data_root / "lc0" / "dummy_lc0_chunk_00.npz", batch_size_for_test, num_policy_planes_for_test, is_lichess=False)
        create_dummy_npz_file(dummy_data_root / "lc0" / "dummy_lc0_chunk_01.npz", batch_size_for_test, num_policy_planes_for_test, is_lichess=False)

        create_dummy_dist_init(dummy_dist_path)
        create_dummy_config(dummy_config_path, dummy_data_root, dummy_dist_path, dummy_output_dir, batch_size_for_test)
            
        command = [
            sys.executable, str(train_script_path),
            '--config', str(dummy_config_path)
        ]

        print(f"\nRunning training script: {' '.join(command)}")
        try:
            # Ensure the CWD for subprocess is the project root if train.py expects that
            # For simplicity, assume train.py can be run from anywhere if config paths are absolute
            process = subprocess.run(command, capture_output=True, text=True, timeout=180) # 3 min timeout

            print("\n--- train.py STDOUT ---")
            print(process.stdout)
            print("--- END train.py STDOUT ---")

            if process.stderr:
                print("\n--- train.py STDERR ---")
                print(process.stderr)
                print("--- END train.py STDERR ---")

            if process.returncode == 0:
                print("\n✓ Training script ran successfully for a few steps.")
                # Check for key log messages
                step_1_found = "[Step 1/" in process.stdout or "Step [1/" in process.stdout # Adjusted for potential log format
                step_3_found = "[Step 3/" in process.stdout or "Step [3/" in process.stdout
                val_start_found = "--- Starting Validation at Step 2 ---" in process.stdout
                val_end_found = "Finished evaluation on" in process.stdout or "Validation complete for step 2" in process.stdout

                if step_1_found and step_3_found:
                    print("✓ Step log messages found.")
                else:
                    print(f"✗ WARNING: Did not find all expected step log messages (1 and 3). Step1: {step_1_found}, Step3: {step_3_found}")
                
                if val_start_found and val_end_found:
                    print("✓ Validation logs found.")
                else:
                    print(f"✗ WARNING: Did not find all expected validation log messages. Start: {val_start_found}, End: {val_end_found}")
                
                if not (step_1_found and step_3_found and val_start_found and val_end_found):
                     print("NOTE: Check STDOUT above for details if warnings occurred.")

            else:
                print(f"\n✗ Training script failed with return code {process.returncode}.")
                sys.exit(1)

        except subprocess.TimeoutExpired:
            print("\n✗ Training script timed out.")
            if process and hasattr(process, 'stdout') and process.stdout:
                 print("\n--- train.py STDOUT (on timeout) ---")
                 print(process.stdout)
            if process and hasattr(process, 'stderr') and process.stderr:
                 print("\n--- train.py STDERR (on timeout) ---")
                 print(process.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ An error occurred while running train.py: {e}")
            sys.exit(1)

    print("\nTest script finished.")

if __name__ == "__main__":
    main() 