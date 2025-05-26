#!/usr/bin/env python3
"""
train.py

Training script for Chess-ViT with mixed precision, gradient clipping,
and rolling-average metrics. Uses StreamingChessDataset for data.
"""
import os
import glob
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import deque, defaultdict
from safetensors.torch import save_file
from losses import ply_loss_fn, custom_policy_loss_fn, nt_xent_loss_fn, new_policy_loss_fn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
from typing import Optional, Dict, List, Any

from chess_vit import ViTChess, load_model_from_checkpoint
from chess_dataset import ChunkMmapDataset, fast_chess_collate_fn, move_batch_to_device

def load_config(path: str) -> dict:
    # Ensure UTF-8 encoding when reading config to avoid locale decoding errors
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(
    dataset_type: str,
    data_dir: str, 
    batch_size: int, 
    num_workers: int, 
    flips: bool, # For StreamingChessDataset (if re-added), ignored by TensorChessDataset
    seed: Optional[int] = None,
    tensor_glob_pattern: str = "*.npz",
    shuffle_files: bool = True, # New param, True for training, False for validation
    infinite: bool = True # New param, True for training, False for validation
    ):
    # Note: StreamingChessDataset and the original chess_collate_fn are no longer imported from chess_dataset.py
    # If dataset_type == 'streaming', this section will likely cause a NameError unless those are defined/imported elsewhere.
    if dataset_type == 'streaming':
        # Glob all files under data_dir (expected to be PGNs)
        files = glob.glob(os.path.join(data_dir, '*'))
        if not files:
            raise ValueError(f"No PGN files found in directory: {data_dir}")
        # dataset = StreamingChessDataset( ... ) - This will fail if StreamingChessDataset is not defined.
        # For this example, we assume the user will handle the 'streaming' case,
        # or it's no longer used with the new chess_dataset.py
        # The original code for StreamingChessDataset is preserved below for structure,
        # but it will not work without StreamingChessDataset and its collate function.
        # --- Original StreamingChessDataset instantiation ---
        # dataset = StreamingChessDataset(
        #     pgn_paths=files,
        #     num_workers=num_workers,
        #     chunk_size=batch_size,
        #     flips=flips,
        #     seed=seed
        # )
        # dl_num_workers = 0
        # current_collate_fn = old_chess_collate_fn # This was the original name
        # current_batch_size = batch_size
        # --- End Original ---
        # For now, let's make it explicit that this path is problematic:
        raise NotImplementedError("StreamingChessDataset is not supported with the new chess_dataset.py structure unless defined elsewhere.")
    elif dataset_type == 'tensor':
        # Ensure data_dir is a string for Path operations in ChunkMmapDataset
        # This can happen if it's derived from os.path.join which might return a Path-like object
        # on some systems if the base was a Path. Explicitly casting.
        str_data_dir = str(data_dir)

        dataset = ChunkMmapDataset(
            root_dir=str_data_dir,
            batch_size=batch_size,
            shuffle_files=shuffle_files,
            infinite=infinite,
            seed=seed,
            file_glob=tensor_glob_pattern
        )
        dl_num_workers = num_workers
        current_collate_fn = fast_chess_collate_fn
        current_batch_size = None # ChunkMmapDataset yields pre-batched data
    elif dataset_type == 'proportional':
        # NOTE: ChunkMmapDataset doesn't inherently support proportional sampling.
        # This will now treat all .npz files under data_dir equally.
        print("WARNING: 'proportional' dataset_type is using ChunkMmapDataset, which does not support proportional sampling from subdirectories. All .npz files will be treated equally.")
        str_data_dir = str(data_dir)
        dataset = ChunkMmapDataset(
            root_dir=str_data_dir,
            batch_size=batch_size,
            shuffle_files=shuffle_files,
            infinite=infinite,
            seed=seed,
            file_glob=tensor_glob_pattern
        )
        dl_num_workers = num_workers
        current_collate_fn = fast_chess_collate_fn
        current_batch_size = None # ChunkMmapDataset yields pre-batched data
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Choose 'streaming', 'tensor', or 'proportional'.")
    
    loader = DataLoader(
        dataset,
        batch_size=current_batch_size,
        num_workers=dl_num_workers,
        pin_memory=True,
        collate_fn=current_collate_fn,
        persistent_workers=True if dl_num_workers > 0 else False,
        prefetch_factor=6 if dl_num_workers > 0 else None,
        multiprocessing_context='spawn' if dl_num_workers > 0 else None
    )
    return loader


def save_checkpoint(step: int, model: torch.nn.Module, optimizer: optim.Optimizer,
                    scheduler: optim.lr_scheduler._LRScheduler, scaler: GradScaler,
                    cfg: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_prefix = os.path.join(output_dir, f"ckpt_{step:08d}")
    # Save model weights in safetensors
    model_path = ckpt_prefix + ".safetensors"
    save_file(model.state_dict(), model_path)
    # Save optimizer, scheduler, scaler, step, and config
    extras = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'step': step,
        'config': cfg
    }
    torch.save(extras, ckpt_prefix + ".pt")


def plot_all_losses(loss_history_dict: Dict[str, Dict[str, Any]], 
                    val_steps_history: List[int], 
                    output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for loss_name, history_data in loss_history_dict.items():
        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        # Plot training loss
        if 'train' in history_data and 'steps_train' in history_data and history_data['train']:
            ax.plot(history_data['steps_train'], history_data['train'], label=f'Train {loss_name.replace("_", " ").title()}', marker='.')
        
        # Plot validation losses
        for key, values in history_data.items():
            if key.startswith('val_') and values: # e.g., val_lc0, val_lichess
                val_source_name = key.split('val_')[-1]
                if val_steps_history and len(values) == len(val_steps_history):
                     ax.plot(val_steps_history, values, label=f'Val {val_source_name.upper()} {loss_name.replace("_", " ").title()}', marker='x')
                elif values: # Fallback if lengths don't match (shouldn't happen with correct logic)
                    print(f"Warning: Mismatch in length for {loss_name} - {key} or val_steps_history. Plotting with limited x-axis.")
                    ax.plot(values, label=f'Val {val_source_name.upper()} {loss_name.replace("_", " ").title()} (steps issue)', marker='x')


        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title(f'{loss_name.replace("_", " ").title()} Loss History')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plot_filename = os.path.join(output_dir, f"{loss_name}_history.png")
        try:
            plt.savefig(plot_filename)
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close()

def evaluate_model(model: ViTChess, device: torch.device,
                   data_source_dir: str, batch_size: int, num_workers: int,
                   tensor_glob_pattern: str, loss_weights_cfg: Dict,
                   nt_xent_temperature: float, seed: Optional[int] = None,
                   # Add is_lichess_key for flexibility if the key name changes in dataset
                   is_lichess_key: str = 'is_lichess_game') -> Dict[str, float]:
    """Evaluates the model on a given dataset and returns average losses."""
    print(f"Evaluating on: {data_source_dir}")
    model.eval() # Set model to evaluation mode

    # Create a new DataLoader for the validation data source
    # shuffle_files=False and infinite=False for consistent evaluation
    val_loader = make_dataset(
        dataset_type='tensor',
        data_dir=data_source_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        flips=False, # Flips usually not applied during validation unless specifically intended
        seed=seed,
        tensor_glob_pattern=tensor_glob_pattern,
        shuffle_files=False,
        infinite=False
    )

    total_positions = 0
    accumulated_losses = defaultdict(float)
    
    # Define all loss names we expect to calculate (consistent with training)
    # This ensures all keys are present in the output, even if a component is zero
    all_loss_keys = ['policy', 'value', 'moves_left', 'auxiliary_value', 'material', 
                     'contrastive', 'contrastive_v', 'contrastive_h', 'contrastive_hv', 
                     'contrastive_source', 'total', 'compare_lc0']
    for key in all_loss_keys:
        accumulated_losses[key] = 0.0

    with torch.no_grad():
        for i, cpu_batch in enumerate(val_loader):
            batch = move_batch_to_device(cpu_batch, device)
            current_batch_size = batch['bitboards'].size(0) # Actual batch size
            total_positions += current_batch_size

            # Ensure all flag tensors are present, defaulting to zeros if missing
            current_B = batch['bitboards'].size(0)
            default_flags_kwargs = {}
            expected_flags = ['is960', 'stm', 'rep1', 'rep2']
            for flag_key in expected_flags:
                if batch.get(flag_key) is None:
                    # print(f"Warning: flag {flag_key} not found in batch. Defaulting to zeros.")
                    default_flags_kwargs[flag_key] = torch.zeros(current_B, dtype=torch.float, device=device)
                else:
                    default_flags_kwargs[flag_key] = batch.get(flag_key)
            
            # Handle is_lichess_game separately due to different key name
            if batch.get('is_lichess_game') is None:
                # print(f"Warning: flag is_lichess_game not found in batch. Defaulting to zeros.")
                default_flags_kwargs['is_lichess'] = torch.zeros(current_B, dtype=torch.float, device=device)
            else:
                default_flags_kwargs['is_lichess'] = batch.get('is_lichess_game')

            outputs = model(
                batch['bitboards'],
                is960=default_flags_kwargs['is960'],
                stm=default_flags_kwargs['stm'],
                rep1=default_flags_kwargs['rep1'],
                rep2=default_flags_kwargs['rep2'],
                is_lichess=default_flags_kwargs['is_lichess'], 
                flipped_bitboards_v=batch.get('bitboards_v'),
                flipped_bitboards_h=batch.get('bitboards_h'),
                flipped_bitboards_hv=batch.get('bitboards_hv')
            )
            
            logits = outputs['policy']
            policy_target_from_batch = batch['policy_target']
            legal_mask_from_batch = batch.get('legal_mask')
            target_for_loss = torch.full_like(policy_target_from_batch, -1.0, device=logits.device)
            if legal_mask_from_batch is not None:
                legal_mask_on_device = legal_mask_from_batch.to(device=logits.device, dtype=torch.bool)
                policy_target_on_device = policy_target_from_batch.to(device=logits.device)
                target_for_loss[legal_mask_on_device] = policy_target_on_device[legal_mask_on_device]
            else:
                target_for_loss = policy_target_from_batch.to(device=logits.device)
            
            loss_policy = new_policy_loss_fn(target_for_loss, logits)
            loss_value = F.cross_entropy(outputs['final_value'], batch['value_target'])
            loss_moves = ply_loss_fn(outputs['final_moves_left'], batch['ply_target'])
            loss_aux = F.cross_entropy(outputs['early_value'], batch['value_target'])
            loss_material = F.cross_entropy(outputs['early_material'], batch['material_category'])

            # Contrastive loss during evaluation (typically zero if flipped_bitboards are not passed)
            # For simplicity, we'll assume it's zero or calculated if the model's forward pass can handle it.
            # Here, we are NOT passing flipped_bitboards, so 'early_cls' will be present, but contrastive components might be missing
            # from 'outputs' if the model's forward pass expects flipped inputs to generate them.
            # We will rely on the training loop's contrastive loss calculation logic for the keys.
            # loss_contrast = torch.tensor(0.0, device=device) # Default if no components
            # Updated contrastive loss calculation for evaluation (similar to training)
            
            loss_contrast_flip_v = torch.tensor(0.0, device=device)
            loss_contrast_flip_h = torch.tensor(0.0, device=device)
            loss_contrast_flip_hv = torch.tensor(0.0, device=device)
            loss_contrast_source = torch.tensor(0.0, device=device)

            individual_nt_xent_losses = {}

            cls_orig_eval = outputs.get("early_cls")
            if cls_orig_eval is not None and cls_orig_eval.shape[0] > 0:
                if 'contrastive_cls_v_flip' in outputs and outputs['contrastive_cls_v_flip'] is not None and \
                   outputs['contrastive_cls_v_flip'].shape[0] == cls_orig_eval.shape[0]:
                    loss_contrast_flip_v = nt_xent_loss_fn(cls_orig_eval, outputs['contrastive_cls_v_flip'], temperature=nt_xent_temperature)
                    individual_nt_xent_losses['contrastive_v'] = loss_contrast_flip_v.item()
                
                if 'contrastive_cls_h_flip' in outputs and outputs['contrastive_cls_h_flip'] is not None and \
                   outputs['contrastive_cls_h_flip'].shape[0] == cls_orig_eval.shape[0]:
                    loss_contrast_flip_h = nt_xent_loss_fn(cls_orig_eval, outputs['contrastive_cls_h_flip'], temperature=nt_xent_temperature)
                    individual_nt_xent_losses['contrastive_h'] = loss_contrast_flip_h.item()
                if 'contrastive_cls_hv_flip' in outputs and outputs['contrastive_cls_hv_flip'] is not None and \
                   outputs['contrastive_cls_hv_flip'].shape[0] == cls_orig_eval.shape[0]:
                    loss_contrast_flip_hv = nt_xent_loss_fn(cls_orig_eval, outputs['contrastive_cls_hv_flip'], temperature=nt_xent_temperature)
                    individual_nt_xent_losses['contrastive_hv'] = loss_contrast_flip_hv.item()
                # Cross-source NT-Xent loss - Modified for true source flag toggle
                if cls_orig_eval.shape[0] > 0:
                    # Get original bitboards and flags from the batch, ensure they are on the correct device
                    # These are already prepared in default_flags_kwargs and batch
                    
                    # Prepare inputs for get_early_block_cls_features
                    bb_for_toggle = batch['bitboards'] # Already on device
                    is960_for_toggle = default_flags_kwargs['is960']
                    stm_for_toggle = default_flags_kwargs['stm']
                    rep1_for_toggle = default_flags_kwargs['rep1']
                    rep2_for_toggle = default_flags_kwargs['rep2']
                    
                    is_lichess_actual = default_flags_kwargs['is_lichess']
                    # Create toggled is_lichess flags (0.0 -> 1.0, 1.0 -> 0.0)
                    # Ensure it's a float tensor for multiplication with bias later
                    is_lichess_toggled = 1.0 - is_lichess_actual.float() 

                    # Get embeddings with the source flag toggled
                    # The model.get_early_block_cls_features will handle internal .float() for flags
                    cls_toggled = model.get_early_block_cls_features(
                        bb_for_toggle,
                        is960_for_toggle,
                        stm_for_toggle,
                        rep1_for_toggle,
                        rep2_for_toggle,
                        is_lichess_toggled 
                    )

                    if cls_toggled.shape[0] == cls_orig_eval.shape[0]:
                        loss_contrast_source = nt_xent_loss_fn(cls_orig_eval, cls_toggled, temperature=nt_xent_temperature)
                        individual_nt_xent_losses['contrastive_source'] = loss_contrast_source.item()
                    else:
                        # This case should ideally not happen if batch sizes are consistent
                        print(f"Warning: Mismatch in shapes for source contrastive loss. cls_orig: {cls_orig_eval.shape}, cls_toggled: {cls_toggled.shape}")
                        individual_nt_xent_losses['contrastive_source'] = 0.0
                else:
                    individual_nt_xent_losses['contrastive_source'] = 0.0 # Batch size 0 or cls_orig is empty

            # Weighted average of contrastive losses for evaluation
            # For evaluation, this is more for consistent logging. The actual values depend on whether flipped boards were passed.
            # If not passed (typical for eval), flip losses will be based on potentially non-existent keys or be zero.
            # The training part has the actual logic for constructing these.
            # Here, we just sum up what was computed.
            # The weighting logic from training (1 for flips, 3 for source) is applied.
            num_flip_components = 0
            sum_flip_losses = torch.tensor(0.0, device=device)
            if loss_contrast_flip_v.item() > 0: sum_flip_losses+=loss_contrast_flip_v; num_flip_components+=1
            if loss_contrast_flip_h.item() > 0: sum_flip_losses+=loss_contrast_flip_h; num_flip_components+=1
            if loss_contrast_flip_hv.item() > 0: sum_flip_losses+=loss_contrast_flip_hv; num_flip_components+=1
            
            avg_flip_loss = sum_flip_losses / num_flip_components if num_flip_components > 0 else torch.tensor(0.0, device=device)

            # In evaluation, loss_contrast_source would typically be 0 unless eval data is specifically set up for it.
            # The overall contrastive loss is the weighted average.
            # Weights: flips=1 each, source=3. Total weight = num_flip_components + (3 if source_loss > 0 else 0)
            total_contrastive_weight = num_flip_components
            weighted_sum_contrastive = avg_flip_loss * num_flip_components # Sum of flip losses
            
            # For evaluation, we assume loss_contrast_source is not actively computed unless eval setup is very specific.
            # So, effectively, the contrastive loss reported in eval will be the average of available flip losses.
            loss_contrast = avg_flip_loss # Default to average of flip losses for eval

            lw = loss_weights_cfg
            total_loss = (
                lw['policy'] * loss_policy +
                lw['value'] * loss_value +
                lw['moves_left'] * loss_moves +
                lw['auxiliary_value'] * loss_aux +
                lw['material'] * loss_material +
                lw['contrastive'] * loss_contrast
            )
            compare_lc0 = loss_policy + 1.6 * loss_value + 0.5 * loss_moves

            accumulated_losses['policy'] += loss_policy.item() * current_batch_size
            accumulated_losses['value'] += loss_value.item() * current_batch_size
            accumulated_losses['moves_left'] += loss_moves.item() * current_batch_size
            accumulated_losses['auxiliary_value'] += loss_aux.item() * current_batch_size
            accumulated_losses['material'] += loss_material.item() * current_batch_size
            accumulated_losses['contrastive'] += loss_contrast.item() * current_batch_size # This is the overall weighted average
            accumulated_losses['contrastive_v'] += loss_contrast_flip_v.item() * current_batch_size
            accumulated_losses['contrastive_h'] += loss_contrast_flip_h.item() * current_batch_size
            accumulated_losses['contrastive_hv'] += loss_contrast_flip_hv.item() * current_batch_size
            # Add source contrastive loss to accumulated_losses (will be 0 if not computed)
            accumulated_losses['contrastive_source'] = accumulated_losses.get('contrastive_source', 0.0) + loss_contrast_source.item() * current_batch_size
            accumulated_losses['total'] += total_loss.item() * current_batch_size
            accumulated_losses['compare_lc0'] += compare_lc0.item() * current_batch_size
            
            if (i + 1) % 100 == 0 : # Log progress within evaluation
                 print(f"  Evaluated {total_positions} positions from {data_source_dir}...")


    avg_losses = {name: (loss_sum / total_positions if total_positions > 0 else 0)
                  for name, loss_sum in accumulated_losses.items()}
    
    # model.train() # Caller should handle restoring model state
    print(f"Finished evaluation on {data_source_dir}. Total positions: {total_positions}")
    return avg_losses

def main(config_path: str):
    # Load config and cast critical fields to proper types
    cfg = load_config(config_path)
    # Optimizer config types
    opt_cfg = cfg['optimiser']
    opt_cfg['lr'] = float(opt_cfg['lr'])
    opt_cfg['weight_decay'] = float(opt_cfg['weight_decay'])
    opt_cfg['betas'] = tuple(map(float, opt_cfg['betas']))
    opt_cfg['warmup_steps'] = int(opt_cfg['warmup_steps'])
    # Dataset config types
    ds_cfg = cfg['dataset']
    ds_cfg['batch_size'] = int(ds_cfg['batch_size'])
    ds_cfg['num_workers'] = int(ds_cfg['num_workers'])
    ds_cfg['flips'] = bool(ds_cfg['flips'])
    ds_cfg['type'] = ds_cfg.get('type', 'streaming') # 'streaming' or 'tensor', default to 'streaming'
    ds_cfg['tensor_glob_pattern'] = ds_cfg.get('tensor_glob_pattern', '**/*.npz') # For tensor dataset
    # Runtime config types
    rt = cfg['runtime']
    rt['grad_accum'] = int(rt['grad_accum'])
    rt['max_steps'] = int(rt['max_steps'])
    rt['log_every'] = int(rt['log_every'])
    rt['ckpt_every'] = int(rt['ckpt_every'])
    rt['gradient_clip_norm'] = float(rt['gradient_clip_norm'])
    rt['val_every'] = int(rt.get('val_every', 20000)) # New
    rt['test_data_dir'] = str(rt.get('test_data_dir', 'test')) # New
    # Model config types
    model_cfg_types = cfg['model']
    model_cfg_types['dim'] = int(model_cfg_types['dim'])
    model_cfg_types['depth'] = int(model_cfg_types['depth'])
    model_cfg_types['early_depth'] = int(model_cfg_types['early_depth'])
    model_cfg_types['heads'] = int(model_cfg_types['heads'])
    model_cfg_types['freeze_distance_iters'] = int(model_cfg_types['freeze_distance_iters'])
    # Handle Optional[int] for pool_every_k_blocks
    pool_every_k_config_val = model_cfg_types.get('pool_every_k_blocks')
    if pool_every_k_config_val is not None:
        model_cfg_types['pool_every_k_blocks'] = int(pool_every_k_config_val)
    else:
        model_cfg_types['pool_every_k_blocks'] = None # Ensure it's None if not present or explicitly None
    model_cfg_types['cls_dropout_rate'] = float(model_cfg_types.get('cls_dropout_rate', 0.0))
    model_cfg_types['cls_pool_alpha_init'] = float(model_cfg_types.get('cls_pool_alpha_init', 1.0))
    model_cfg_types['cls_pool_alpha_requires_grad'] = bool(model_cfg_types.get('cls_pool_alpha_requires_grad', True))
    model_cfg_types['policy_head_conv_dim'] = int(model_cfg_types.get('policy_head_conv_dim', 128))
    model_cfg_types['policy_head_mlp_hidden_dim'] = int(model_cfg_types.get('policy_head_mlp_hidden_dim', 256))
    model_cfg_types['drop_path'] = float(model_cfg_types.get('drop_path', 0.1))
    # Rolling metrics config types
    rm = cfg.get('rolling_metrics', {})
    if 'window_size' in rm:
        rm['window_size'] = int(rm['window_size'])
    
    # Setup device
    if cfg['runtime']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        print("INFO: Training on CUDA.")
    else:
        device = torch.device("cpu")
        if cfg['runtime']['device'] == 'cuda':
            print("WARNING: CUDA specified in config but not available. Training on CPU.")
        else:
            print("INFO: Training on CPU.")

    # Seed
    seed_everything(cfg.get('seed', 42))

    # Load distance matrix
    dist_path = cfg['model']['distance_matrix_path']
    # distance_matrix is (65,65). MultiHeadSelfAttention in Block will expand it per head.
    distance_matrix_numpy = np.load(dist_path)
    distance_matrix = torch.from_numpy(distance_matrix_numpy).float() # Keep on CPU, model init moves to device

    # Build model
    model_cfg = dict(
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        early_depth=cfg['model']['early_depth'],
        heads=cfg['model']['heads'],
        distance_matrix=distance_matrix,
        freeze_distance=True,
        pool_every_k_blocks=cfg['model']['pool_every_k_blocks'],
        cls_dropout_rate=cfg['model']['cls_dropout_rate'],
        cls_pool_alpha_init=cfg['model']['cls_pool_alpha_init'],
        cls_pool_alpha_requires_grad=cfg['model']['cls_pool_alpha_requires_grad'],
        policy_head_conv_dim=cfg['model']['policy_head_conv_dim'],
        policy_head_mlp_hidden_dim=cfg['model']['policy_head_mlp_hidden_dim'],
        num_policy_planes=cfg['model'].get('num_policy_planes', 73),
        drop_path=cfg['model']['drop_path']
    )
    model = ViTChess(**model_cfg)
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception as e:
        print(f"torch.compile skipped ({e})")
    model.to(device)

    # Optimizer and scheduler
    optimizer_args = dict(
        lr=opt_cfg['lr'],
        betas=opt_cfg['betas'],
        weight_decay=opt_cfg['weight_decay']
    )
    if device.type == 'cuda': # Only use fused if on CUDA
        optimizer_args['fused'] = True
        print("INFO: Using fused AdamW optimizer for CUDA.")
    else:
        print("INFO: Using non-fused AdamW optimizer (device is not CUDA).")

    optimizer = optim.AdamW(
        model.parameters(),
        **optimizer_args
    )
    # Cosine with warmup: placeholder scheduler
    total_steps = rt['max_steps']
    warmup_steps = opt_cfg['warmup_steps']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # DataLoader
    loader = make_dataset(
        dataset_type=ds_cfg['type'],
        data_dir=ds_cfg['data_dir'], # Changed from raw_data_dir
        batch_size=ds_cfg['batch_size'],
        num_workers=ds_cfg['num_workers'],
        flips=ds_cfg['flips'], # Passed along, used by streaming, ignored by tensor dataset directly
        seed=cfg.get('seed', 42),
        tensor_glob_pattern=ds_cfg['tensor_glob_pattern'],
        shuffle_files=True, # For training
        infinite=True     # For training
    )
    data_iter = iter(loader)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=cfg['logging']['output_dir']) if cfg['logging']['tensorboard'] else None

    # Rolling metrics
    window = rm['window_size']
    metric_names = ['policy', 'value', 'moves_left', 'auxiliary_value', 'material', 
                    'contrastive', 'contrastive_v', 'contrastive_h', 'contrastive_hv', 
                    'contrastive_source', 'total', 'compare_lc0'] # Added 'contrastive_source'
    metrics = {name: deque(maxlen=window) for name in metric_names}

    # History of rolling metrics
    history = {name: [] for name in metric_names}
    log_steps: list = []

    # Throughput tracking
    last_log_time = time.time()
    last_log_step = 0
    # Add a list to store data loading times
    data_loading_times = deque(maxlen=window if 'window_size' in rm else 100) # reuse rolling window size

    # New loss history structure
    loss_history: Dict[str, Dict[str, Any]] = {name: {'train': [], 'steps_train': []} for name in metric_names}
    val_steps_history: List[int] = [] # Shared steps for all validation sources

    # Training loop
    step = 0
    grad_accum = rt['grad_accum']
    log_every = rt['log_every']
    ckpt_every = rt['ckpt_every']
    val_every = rt['val_every']
    test_data_dir_from_cfg = ds_cfg.get('test_data_dir', 'test') # Changed from rt to ds_cfg
    freeze_iters = model_cfg_types['freeze_distance_iters']

    try:
        while step < rt['max_steps']:
            model.train() # Ensure model is in training mode for the training step
            if step == freeze_iters:
                model.freeze_distance_bias(False)
            if step % grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            t_data_start = time.time() # Start timing
            cpu_batch = next(data_iter)
            t_data_end = time.time() # End timing
            data_loading_times.append(t_data_end - t_data_start) # Store duration
            
            # Move batch to target device and add flipped variants
            batch = move_batch_to_device(cpu_batch, device) 
            # Data is now on the correct device

            # Ensure all flag tensors are present, defaulting to zeros if missing
            current_B = batch['bitboards'].size(0)
            default_flags_kwargs = {}
            expected_flags = ['is960', 'stm', 'rep1', 'rep2']
            for flag_key in expected_flags:
                if batch.get(flag_key) is None:
                    # print(f"Warning: flag {flag_key} not found in batch. Defaulting to zeros.")
                    default_flags_kwargs[flag_key] = torch.zeros(current_B, dtype=torch.float, device=device)
                else:
                    default_flags_kwargs[flag_key] = batch.get(flag_key)
            
            # Handle is_lichess_game separately due to different key name
            if batch.get('is_lichess_game') is None:
                # print(f"Warning: flag is_lichess_game not found in batch. Defaulting to zeros.")
                default_flags_kwargs['is_lichess'] = torch.zeros(current_B, dtype=torch.float, device=device)
            else:
                default_flags_kwargs['is_lichess'] = batch.get('is_lichess_game')

            # Forward + loss
            with autocast():
                outputs = model(
                    batch['bitboards'],
                    is960=default_flags_kwargs['is960'],
                    stm=default_flags_kwargs['stm'],
                    rep1=default_flags_kwargs['rep1'],
                    rep2=default_flags_kwargs['rep2'],
                    is_lichess=default_flags_kwargs['is_lichess'], 
                    flipped_bitboards_v=batch.get('bitboards_v'),
                    flipped_bitboards_h=batch.get('bitboards_h'),
                    flipped_bitboards_hv=batch.get('bitboards_hv')
                )
                # Compute component losses using robust masking and scaling
                logits = outputs['policy']

                # Prepare target for new_policy_loss_fn
                policy_target_from_batch = batch['policy_target'] 
                legal_mask_from_batch = batch.get('legal_mask')
                
                target_for_loss = torch.full_like(policy_target_from_batch, -1.0, device=logits.device)
                if legal_mask_from_batch is not None:
                    legal_mask_on_device = legal_mask_from_batch.to(device=logits.device, dtype=torch.bool)
                    # Ensure policy_target_from_batch is also on the same device before indexing
                    policy_target_on_device = policy_target_from_batch.to(device=logits.device)
                    target_for_loss[legal_mask_on_device] = policy_target_on_device[legal_mask_on_device]
                else:
                    print("Warning: legal_mask is None in training loop. `new_policy_loss_fn` might not behave as expected if targets don't have -1 for illegal moves.")
                    # Fallback: use original target, but this might not be what new_policy_loss_fn expects
                    target_for_loss = policy_target_from_batch.to(device=logits.device) 
                
                # --- Original Loss Calculation (Batch Averaged) ---
                loss_policy = new_policy_loss_fn(target_for_loss, logits) 
                loss_value = F.cross_entropy(outputs['final_value'], batch['value_target'])
                loss_moves = ply_loss_fn(outputs['final_moves_left'], batch['ply_target'])
                loss_aux = F.cross_entropy(outputs['early_value'], batch['value_target'])
                loss_material = F.cross_entropy(outputs['early_material'], batch['material_category'])
                
                # Contrastive loss calculation (remains batch averaged)
                cls_orig = outputs["early_cls"]
                loss_contrast_components = []
                individual_nt_xent_losses = {}
                nt_temperature = cfg.get('loss_weights', {}).get('nt_xent_temperature', 0.1)
                
                loss_contrast_flip_v = torch.tensor(0.0, device=device)
                loss_contrast_flip_h = torch.tensor(0.0, device=device)
                loss_contrast_flip_hv = torch.tensor(0.0, device=device)
                loss_contrast_source = torch.tensor(0.0, device=device)

                if 'contrastive_cls_v_flip' in outputs and outputs['contrastive_cls_v_flip'] is not None and \
                   cls_orig.shape[0] > 0 and outputs['contrastive_cls_v_flip'].shape[0] == cls_orig.shape[0]:
                    loss_contrast_flip_v = nt_xent_loss_fn(cls_orig, outputs['contrastive_cls_v_flip'], temperature=nt_temperature)
                    individual_nt_xent_losses['contrastive_v'] = loss_contrast_flip_v.item()
                
                if 'contrastive_cls_h_flip' in outputs and outputs['contrastive_cls_h_flip'] is not None and \
                   cls_orig.shape[0] > 0 and outputs['contrastive_cls_h_flip'].shape[0] == cls_orig.shape[0]:
                    loss_contrast_flip_h = nt_xent_loss_fn(cls_orig, outputs['contrastive_cls_h_flip'], temperature=nt_temperature)
                    individual_nt_xent_losses['contrastive_h'] = loss_contrast_flip_h.item()

                if 'contrastive_cls_hv_flip' in outputs and outputs['contrastive_cls_hv_flip'] is not None and \
                   cls_orig.shape[0] > 0 and outputs['contrastive_cls_hv_flip'].shape[0] == cls_orig.shape[0]:
                    loss_contrast_flip_hv = nt_xent_loss_fn(cls_orig, outputs['contrastive_cls_hv_flip'], temperature=nt_temperature)
                    individual_nt_xent_losses['contrastive_hv'] = loss_contrast_flip_hv.item()

                # Cross-source NT-Xent loss - Modified for true source flag toggle
                if cls_orig.shape[0] > 0:
                    # Get original bitboards and flags from the batch, ensure they are on the correct device
                    # These are already prepared in default_flags_kwargs and batch
                    
                    # Prepare inputs for get_early_block_cls_features
                    bb_for_toggle = batch['bitboards'] # Already on device
                    is960_for_toggle = default_flags_kwargs['is960']
                    stm_for_toggle = default_flags_kwargs['stm']
                    rep1_for_toggle = default_flags_kwargs['rep1']
                    rep2_for_toggle = default_flags_kwargs['rep2']
                    
                    is_lichess_actual = default_flags_kwargs['is_lichess']
                    # Create toggled is_lichess flags (0.0 -> 1.0, 1.0 -> 0.0)
                    # Ensure it's a float tensor for multiplication with bias later
                    is_lichess_toggled = 1.0 - is_lichess_actual.float() 

                    # Get embeddings with the source flag toggled
                    # The model.get_early_block_cls_features will handle internal .float() for flags
                    cls_toggled = model.get_early_block_cls_features(
                        bb_for_toggle,
                        is960_for_toggle,
                        stm_for_toggle,
                        rep1_for_toggle,
                        rep2_for_toggle,
                        is_lichess_toggled 
                    )

                    if cls_toggled.shape[0] == cls_orig.shape[0]:
                        loss_contrast_source = nt_xent_loss_fn(cls_orig, cls_toggled, temperature=nt_temperature)
                        individual_nt_xent_losses['contrastive_source'] = loss_contrast_source.item()
                    else:
                        # This case should ideally not happen if batch sizes are consistent
                        print(f"Warning: Mismatch in shapes for source contrastive loss. cls_orig: {cls_orig.shape}, cls_toggled: {cls_toggled.shape}")
                        individual_nt_xent_losses['contrastive_source'] = 0.0
                else:
                    individual_nt_xent_losses['contrastive_source'] = 0.0 # Batch size 0 or cls_orig is empty

                # Calculate weighted average for the 'contrastive' loss component
                # Weights: 1 for each flip (v,h,hv), 3 for source
                weighted_sum_contrast_losses = torch.tensor(0.0, device=device)
                total_weight_contrast = 0.0
                
                if loss_contrast_flip_v.item() > 1e-9: # Check if loss is non-trivial
                    weighted_sum_contrast_losses += 1.0 * loss_contrast_flip_v
                    total_weight_contrast += 1.0
                if loss_contrast_flip_h.item() > 1e-9:
                    weighted_sum_contrast_losses += 1.0 * loss_contrast_flip_h
                    total_weight_contrast += 1.0
                if loss_contrast_flip_hv.item() > 1e-9:
                    weighted_sum_contrast_losses += 1.0 * loss_contrast_flip_hv
                    total_weight_contrast += 1.0
                if loss_contrast_source.item() > 1e-9:
                    weighted_sum_contrast_losses += 3.0 * loss_contrast_source
                    total_weight_contrast += 3.0
                
                final_averaged_contrast_loss = weighted_sum_contrast_losses / total_weight_contrast if total_weight_contrast > 0 else torch.tensor(0.0, device=device)
                # --- End Original Loss Calculation ---

                # Weighted total (using direct batch-averaged losses)
                lw = cfg['loss_weights']
                total_loss = (
                    lw['policy'] * loss_policy
                    + lw['value'] * loss_value
                    + lw['moves_left'] * loss_moves
                    + lw['auxiliary_value'] * loss_aux
                    + lw['material'] * loss_material
                    + lw['contrastive'] * final_averaged_contrast_loss 
                )
                # Comparison metric (using direct batch-averaged losses)
                compare_lc0 = (
                    loss_policy
                    + 1.6 * loss_value
                    + 0.5 * loss_moves
                )

            # Backward
            scaler.scale(total_loss).backward()

            # Step optimizer
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), rt['gradient_clip_norm'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # Record metrics (use direct batch-averaged loss items)
            metrics['policy'].append(loss_policy.item())
            metrics['value'].append(loss_value.item())
            metrics['moves_left'].append(loss_moves.item())
            metrics['auxiliary_value'].append(loss_aux.item())
            metrics['material'].append(loss_material.item())
            metrics['contrastive'].append(final_averaged_contrast_loss.item())
            # The individual_nt_xent_losses part for metrics was already correct and remains
            metrics.setdefault('contrastive_v', deque(maxlen=window)).append(individual_nt_xent_losses.get('contrastive_v', 0.0))
            metrics.setdefault('contrastive_h', deque(maxlen=window)).append(individual_nt_xent_losses.get('contrastive_h', 0.0))
            metrics.setdefault('contrastive_hv', deque(maxlen=window)).append(individual_nt_xent_losses.get('contrastive_hv', 0.0))
            metrics.setdefault('contrastive_source', deque(maxlen=window)).append(individual_nt_xent_losses.get('contrastive_source', 0.0))
            metrics['total'].append(total_loss.item())
            metrics['compare_lc0'].append(compare_lc0.item())

            # Logging
            if (step + 1) % log_every == 0:
                mean_metrics = {name: np.mean(metrics[name]) for name in metric_names}
                mean_data_loading_time = np.mean(data_loading_times)
                alpha_log_str = ""

                if writer: # TensorBoard specific logging
                    for name, val in mean_metrics.items():
                        loss_history[name]['train'].append(val)
                        loss_history[name]['steps_train'].append(step + 1)
                        writer.add_scalar(f'train/{name}', val, step + 1)
                    
                    if hasattr(model, 'alphas') and model.alphas:
                        alpha_values_tb = [] # Separate for tb if needed, though string is fine
                        for i, alpha_param in enumerate(model.alphas):
                            writer.add_scalar(f'alpha_pool_{i}', alpha_param.item(), step + 1)
                            alpha_values_tb.append(f'{alpha_param.item():.3f}')
                        # alpha_log_str could be formed here if needed only for tb log line

                # Console logging (always happens if log_every condition met)
                if hasattr(model, 'alphas') and model.alphas: # Recalculate alpha_log_str for console if needed
                    alpha_values_console = [f'{alpha_param.item():.3f}' for alpha_param in model.alphas]
                    alpha_log_str = f", alphas: [{', '.join(alpha_values_console)}]"
                
                current_time = time.time()
                delta_steps = (step + 1) - last_log_step
                positions = delta_steps * ds_cfg['batch_size']
                elapsed = current_time - last_log_time
                throughput = positions / elapsed if elapsed > 0 else float('inf')
                
                print(f"[Step {step+1}/{rt['max_steps']}] total_loss: {mean_metrics['total']:.4f}, lc0_loss: {mean_metrics['compare_lc0']:.4f}, policy: {mean_metrics['policy']:.4f}, value: {mean_metrics['value']:.4f}, moves_left: {mean_metrics['moves_left']:.4f}, auxiliary_value: {mean_metrics['auxiliary_value']:.4f}, material: {mean_metrics['material']:.4f}, contrastive_avg: {mean_metrics.get('contrastive', 0.0):.4f} (v:{mean_metrics.get('contrastive_v',0.0):.2f}, h:{mean_metrics.get('contrastive_h',0.0):.2f}, hv:{mean_metrics.get('contrastive_hv',0.0):.2f}, src:{mean_metrics.get('contrastive_source',0.0):.2f}), {throughput:.1f} positions/sec, data_load_time: {mean_data_loading_time:.4f}s{alpha_log_str}", flush=True)
                
                last_log_time = current_time
                last_log_step = step + 1

                # Plot matplotlib PNG (conditional on config and writer for consistency, or make independent)
                # If matplotlib is meant to be independent of tensorboard, this also needs adjustment.
                # For now, keeping it conditional on tensorboard for simplicity of this change.
                if writer and cfg['logging']['matplotlib']:
                    plot_all_losses(loss_history, val_steps_history, cfg['logging']['output_dir'])

            # Validation step
            if (step + 1) % val_every == 0:
                print(f"\n--- Starting Validation at Step {step+1} ---")
                original_model_training_state = model.training
                model.eval() # Set to eval mode

                test_base_dir = test_data_dir_from_cfg # Use variable derived from ds_cfg
                if not os.path.isdir(test_base_dir):
                    print(f"Warning: Test data directory {test_base_dir} not found. Skipping validation.")
                else:
                    val_steps_history.append(step + 1) # Record this validation step
                    for source_name_in_dir in os.listdir(test_base_dir):
                        source_path_to_check = os.path.join(test_base_dir, source_name_in_dir)
                        if os.path.isdir(source_path_to_check): # Only process if it's a directory
                            avg_val_losses = evaluate_model(
                                model, device, source_path_to_check, # Pass the actual directory
                                ds_cfg['batch_size'], ds_cfg['num_workers'],
                                ds_cfg['tensor_glob_pattern'], cfg['loss_weights'],
                                cfg.get('loss_weights', {}).get('nt_xent_temperature', 0.1),
                                seed=cfg.get('seed', 42),
                                is_lichess_key='is_lichess_game'
                            )
                            print(f"Validation Results for {source_name_in_dir.upper()} (Step {step+1}):")
                            log_val_message_parts = []
                            for loss_n, loss_v in avg_val_losses.items():
                                loss_history[loss_n][f'val_{source_name_in_dir}'] = loss_history[loss_n].get(f'val_{source_name_in_dir}', []) + [loss_v]
                                if writer:
                                    writer.add_scalar(f'val/{source_name_in_dir}/{loss_n}', loss_v, step + 1)
                                if loss_n in ['total', 'policy', 'value', 'compare_lc0']: # Print key validation losses
                                    log_val_message_parts.append(f"{loss_n}: {loss_v:.4f}")
                            print("  " + ", ".join(log_val_message_parts))
                
                if original_model_training_state:
                    model.train() # Restore training mode
                print(f"--- Finished Validation at Step {step+1} ---\n")

            # Checkpoint
            if (step + 1) % ckpt_every == 0:
                save_checkpoint(step + 1, model, optimizer, scheduler, scaler, cfg, cfg['logging']['output_dir'])

            step += 1

    except KeyboardInterrupt:
        print("Interrupted. Saving final checkpoint...")
        save_checkpoint(step, model, optimizer, scheduler, scaler, cfg, cfg['logging']['output_dir'])
    finally:
        if writer:
            # Final plot generation
            if cfg['logging']['matplotlib'] and loss_history:
                 print("Generating final plots...")
                 plot_all_losses(loss_history, val_steps_history, cfg['logging']['output_dir'])
            writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Chess-ViT")
    parser.add_argument('--config', '-c', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)