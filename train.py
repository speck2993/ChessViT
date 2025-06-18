#!/usr/bin/env python3
"""
train.py

Training script for Chess-ViT with mixed precision, gradient clipping,
and rolling-average metrics. Simplified for single-source data without contrastive learning.
"""
import os
import glob
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import deque, defaultdict
from safetensors.torch import save_file
from tqdm import tqdm
from losses import (
    ply_loss_fn as _ply_loss_fn,
    new_policy_loss_fn as _new_policy_loss_fn,
)

def label_smoothed_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, 
                                 smoothing: float = 0.02) -> torch.Tensor:
    """
    Apply label smoothing to cross entropy loss.
    For 3-class value targets: [1, 0, 0] -> [0.96, 0.02, 0.02]
    """
    num_classes = logits.size(-1)
    confidence = 1.0 - smoothing
    
    # Create one-hot targets
    one_hot = F.one_hot(targets, num_classes=num_classes).float()
    
    # Apply smoothing: confidence for correct class, smoothing/(num_classes-1) for others
    smooth_targets = one_hot * confidence + (1 - one_hot) * (smoothing / (num_classes - 1))
    
    # Compute cross entropy with soft targets
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
    
    return loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
from typing import Optional, Dict, List, Any
import torch.nn as nn
import logging
from datetime import datetime

from chess_vit import ViTChess, load_model_from_checkpoint
from chess_dataset import ChunkMmapDataset, fast_chess_collate_fn, move_batch_to_device
from email_notifier import TrainingEmailNotifier, setup_email_logging

def load_config(path: str) -> dict:
    # Ensure UTF-8 encoding when reading config to avoid locale decoding errors
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # --- Type casting for critical config values ---
    # Model
    cfg['model']['dim'] = int(cfg['model']['dim'])
    cfg['model']['depth'] = int(cfg['model']['depth'])
    cfg['model']['early_depth'] = int(cfg['model']['early_depth'])
    cfg['model']['heads'] = int(cfg['model']['heads'])

    if 'pool_every_k_blocks' in cfg['model'] and cfg['model']['pool_every_k_blocks'] is not None:
        cfg['model']['pool_every_k_blocks'] = int(cfg['model']['pool_every_k_blocks'])
    if 'cls_dropout_rate' in cfg['model'] and cfg['model']['cls_dropout_rate'] is not None:
        cfg['model']['cls_dropout_rate'] = float(cfg['model']['cls_dropout_rate'])
    if 'policy_head_conv_dim' in cfg['model'] and cfg['model']['policy_head_conv_dim'] is not None:
        cfg['model']['policy_head_conv_dim'] = int(cfg['model']['policy_head_conv_dim'])
    if 'policy_head_mlp_hidden_dim' in cfg['model'] and cfg['model']['policy_head_mlp_hidden_dim'] is not None:
        cfg['model']['policy_head_mlp_hidden_dim'] = int(cfg['model']['policy_head_mlp_hidden_dim'])
    if 'value_cross_attn_summary_dim' in cfg['model'] and cfg['model']['value_cross_attn_summary_dim'] is not None:
        cfg['model']['value_cross_attn_summary_dim'] = int(cfg['model']['value_cross_attn_summary_dim'])
    if 'moves_left_cross_attn_summary_dim' in cfg['model'] and cfg['model']['moves_left_cross_attn_summary_dim'] is not None:
        cfg['model']['moves_left_cross_attn_summary_dim'] = int(cfg['model']['moves_left_cross_attn_summary_dim'])
    if 'adaptive_pool_temperature' in cfg['model'] and cfg['model']['adaptive_pool_temperature'] is not None:
        cfg['model']['adaptive_pool_temperature'] = float(cfg['model']['adaptive_pool_temperature'])

    # SmolGen parameters
    if 'smolgen_start_layer' in cfg['model']:
        cfg['model']['smolgen_start_layer'] = int(cfg['model']['smolgen_start_layer'])
    if 'smolgen_latent_dim' in cfg['model']:
        cfg['model']['smolgen_latent_dim'] = int(cfg['model']['smolgen_latent_dim'])
    if 'smolgen_dropout' in cfg['model']:
        cfg['model']['smolgen_dropout'] = float(cfg['model']['smolgen_dropout'])

    # Value head parameters
    if 'value_spatial_compress_dim' in cfg['model']:
        cfg['model']['value_spatial_compress_dim'] = int(cfg['model']['value_spatial_compress_dim'])
    if 'value_head_mlp_dims' in cfg['model'] and cfg['model']['value_head_mlp_dims'] is not None:
        cfg['model']['value_head_mlp_dims'] = [int(d) for d in cfg['model']['value_head_mlp_dims']]
    if 'moves_left_spatial_compress_dim' in cfg['model']:
        cfg['model']['moves_left_spatial_compress_dim'] = int(cfg['model']['moves_left_spatial_compress_dim'])
    if 'moves_left_head_mlp_dims' in cfg['model'] and cfg['model']['moves_left_head_mlp_dims'] is not None:
        cfg['model']['moves_left_head_mlp_dims'] = [int(d) for d in cfg['model']['moves_left_head_mlp_dims']]

    # Hybrid patch embedding parameters
    if 'patch_resblock_hidden' in cfg['model']:
        cfg['model']['patch_resblock_hidden'] = int(cfg['model']['patch_resblock_hidden'])
    if 'patch_global_proj_dim' in cfg['model']:
        cfg['model']['patch_global_proj_dim'] = int(cfg['model']['patch_global_proj_dim'])
    if 'patch_embed_dropout' in cfg['model']:
        cfg['model']['patch_embed_dropout'] = float(cfg['model']['patch_embed_dropout'])

    # Optimiser
    cfg['optimiser']['lr'] = float(cfg['optimiser']['lr'])
    cfg['optimiser']['weight_decay'] = float(cfg['optimiser']['weight_decay'])
    if 'warmup_steps' in cfg['optimiser'] and cfg['optimiser']['warmup_steps'] is not None:
        cfg['optimiser']['warmup_steps'] = int(cfg['optimiser']['warmup_steps'])

    # Dataset
    cfg['dataset']['batch_size'] = int(cfg['dataset']['batch_size'])
    if 'grad_accum' in cfg['dataset'] and cfg['dataset']['grad_accum'] is not None:
        cfg['dataset']['grad_accum'] = int(cfg['dataset']['grad_accum'])
    else:
        cfg['dataset']['grad_accum'] = 1
        print("Warning: 'grad_accum' not found in config under 'dataset'. Defaulting to 1.")

    cfg['dataset']['num_workers'] = int(cfg['dataset']['num_workers'])

    # Runtime
    cfg['runtime']['max_steps'] = int(cfg['runtime']['max_steps'])
    cfg['runtime']['log_every'] = int(cfg['runtime']['log_every'])
    cfg['runtime']['ckpt_every'] = int(cfg['runtime']['ckpt_every'])
    if 'val_every' in cfg['runtime'] and cfg['runtime']['val_every'] is not None:
        cfg['runtime']['val_every'] = int(cfg['runtime']['val_every'])
    if 'gradient_clip_norm' in cfg['runtime'] and cfg['runtime']['gradient_clip_norm'] is not None:
        cfg['runtime']['gradient_clip_norm'] = float(cfg['runtime']['gradient_clip_norm'])
    if 'gradient_clip_value' in cfg['runtime'] and cfg['runtime']['gradient_clip_value'] is not None:
        cfg['runtime']['gradient_clip_value'] = float(cfg['runtime']['gradient_clip_value'])
        
    # Loss weights
    if 'loss_weights' not in cfg:
        cfg['loss_weights'] = {}
    for k in cfg['loss_weights']:
        cfg['loss_weights'][k] = float(cfg['loss_weights'][k])
    
    # Loss weight schedules (dynamic weight adjustment)
    if 'loss_weight_schedules' in cfg:
        for loss_name, schedule_config in cfg['loss_weight_schedules'].items():
            if 'start_weight' in schedule_config:
                schedule_config['start_weight'] = float(schedule_config['start_weight'])
            if 'end_weight' in schedule_config:
                schedule_config['end_weight'] = float(schedule_config['end_weight'])
            if 'decay_rate' in schedule_config:
                schedule_config['decay_rate'] = float(schedule_config['decay_rate'])
            if 'start_step' in schedule_config:
                schedule_config['start_step'] = int(schedule_config['start_step'])
            if 'duration_steps' in schedule_config:
                schedule_config['duration_steps'] = int(schedule_config['duration_steps'])

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
    seed: Optional[int] = None,
    tensor_glob_pattern: str = "*.npz",
    shuffle_files: bool = True,
    infinite: bool = True,
    pad_partial_batches: bool = True
    ):
    
    if dataset_type == 'tensor':
        str_data_dir = str(data_dir)
        dataset = ChunkMmapDataset(
            root_dir=str_data_dir,
            batch_size=batch_size,
            shuffle_files=shuffle_files,
            infinite=infinite,
            seed=seed,
            file_glob=tensor_glob_pattern,
            pad_partial_batches=pad_partial_batches
        )
        dl_num_workers = num_workers
        current_collate_fn = fast_chess_collate_fn
        current_batch_size = None
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Only 'tensor' supported.")
    
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
    """Save checkpoint with aggressive memory management."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt_prefix = os.path.join(output_dir, f"ckpt_{step:08d}")
    
    # Save model first - handle compiled models properly
    model_path = ckpt_prefix + ".safetensors"
    try:
        # Get the state dict - this will use our custom state_dict method
        state_dict = model.state_dict()
        save_file(state_dict, model_path)
    except RuntimeError as e:
        if "share memory" in str(e):
            print(f"Warning: Shared memory error during save. Attempting to use uncompiled model state dict...")
            # If the model is compiled, try to get the original module's state dict
            if hasattr(model, '_orig_mod'):
                state_dict = model._orig_mod.state_dict()
            else:
                # Fallback: copy the state dict to break references
                state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            save_file(state_dict, model_path)
            print(f"Successfully saved model using fallback method")
        else:
            raise
    
    # Clear cache after model save
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Save optimizer and extras
    extras = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'step': step,
        'config': cfg
    }
    torch.save(extras, ckpt_prefix + ".pt")
    
    # Aggressive cleanup
    del extras
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"Checkpoint saved: {ckpt_prefix}")

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
        if 'val' in history_data and history_data['val'] and val_steps_history and len(history_data['val']) == len(val_steps_history):
            ax.plot(val_steps_history, history_data['val'], label=f'Val {loss_name.replace("_", " ").title()}', marker='x')

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
                   seed: Optional[int] = None,
                   test_batch_size: Optional[int] = None) -> Dict[str, float]:
    """Evaluates model with configurable test batch size."""
    print(f"Evaluating on: {data_source_dir}")
    
    # Use larger batch size for testing if specified
    eval_batch_size = test_batch_size or batch_size
    print(f"Using test batch size: {eval_batch_size}")
    
    model.eval()
    total_loss = 0.0
    total_samples = 0
    summed_losses: Dict[str, float] = defaultdict(float)

    val_loader = make_dataset(
        dataset_type='tensor',
        data_dir=data_source_dir,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed,
        tensor_glob_pattern=tensor_glob_pattern,
        shuffle_files=False,
        infinite=False,
        pad_partial_batches=True
    )

    # Calculate total batches for tqdm progress bar using the actual evaluation batch size
    total_batches = 0
    if hasattr(val_loader.dataset, 'total_positions_in_dataset') and val_loader.dataset.total_positions_in_dataset > 0:
        total_batches = (val_loader.dataset.total_positions_in_dataset + eval_batch_size - 1) // eval_batch_size
    else:
        # Fallback if total_positions_in_dataset is not available or 0, progress bar will not show total
        print("Warning: Could not determine total number of batches for validation progress bar.")

    with torch.no_grad():
        for i, cpu_batch in enumerate(tqdm(val_loader, total=total_batches, desc=f"Evaluating {data_source_dir}", unit="batch")):
            batch = move_batch_to_device(cpu_batch, device)
            current_batch_size = batch['bitboards'].size(0)
            total_samples += current_batch_size

            # Ensure all flag tensors are present, defaulting to zeros if missing
            current_B = batch['bitboards'].size(0)
            default_flags_kwargs = {}
            expected_flags = ['is960', 'stm', 'rep1', 'rep2']
            for flag_key in expected_flags:
                if batch.get(flag_key) is None:
                    default_flags_kwargs[flag_key] = torch.zeros(current_B, dtype=torch.float, device=device)
                else:
                    default_flags_kwargs[flag_key] = batch.get(flag_key)

            outputs = model(
                batch['bitboards'],
                is960=default_flags_kwargs['is960'],
                stm=default_flags_kwargs['stm'],
                rep1=default_flags_kwargs['rep1'],
                rep2=default_flags_kwargs['rep2']
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
            
            loss_policy = _new_policy_loss_fn(target_for_loss, logits)
            loss_value = label_smoothed_cross_entropy(outputs['final_value'], batch['value_target'], smoothing=0.04)
            loss_moves = _ply_loss_fn(outputs['final_moves_left'], batch['ply_target'])
            loss_aux = label_smoothed_cross_entropy(outputs['early_value'], batch['value_target'], smoothing=0.04)
            loss_material = F.cross_entropy(outputs['early_material'], batch['material_category'])
            loss_cls_sparsity = torch.tensor(0.0, device=device)
            if 'final_cls_features' in outputs:
                loss_cls_sparsity = outputs['final_cls_features'].abs().mean()

            lw = loss_weights_cfg
            total_loss = (
                lw['policy'] * loss_policy +
                lw['value'] * loss_value +
                lw['moves_left'] * loss_moves +
                lw['auxiliary_value'] * loss_aux +
                lw['material'] * loss_material
            )
            compare_lc0 = loss_policy + 1.6 * loss_value + 0.5 * loss_moves

            summed_losses['policy'] += loss_policy.item() * current_batch_size
            summed_losses['value'] += loss_value.item() * current_batch_size
            summed_losses['moves_left'] += loss_moves.item() * current_batch_size
            summed_losses['auxiliary_value'] += loss_aux.item() * current_batch_size
            summed_losses['material'] += loss_material.item() * current_batch_size
            summed_losses['total'] += total_loss.item() * current_batch_size
            summed_losses['compare_lc0'] += compare_lc0.item() * current_batch_size

    avg_losses = {name: (loss_sum / total_samples if total_samples > 0 else 0)
                  for name, loss_sum in summed_losses.items()}
    
    print(f"Finished evaluation on {data_source_dir}. Total positions: {total_samples}")
    return avg_losses

def log_nan_batch(batch: Dict[str, torch.Tensor], step: int, output_dir: str):
    """Log information about a batch that produced NaN loss."""
    log_file = os.path.join(output_dir, "nan_positions.log")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(log_file, 'a') as f:
            f.write(f"\n--- NaN Loss at Step {step} ---\n")
            f.write(f"Batch size: {batch['bitboards'].size(0)}\n")
            
            # Log source files if available
            if 'source_file_basename' in batch:
                f.write(f"Source file: {batch['source_file_basename']}\n")
            
            # Log indices if available
            if 'original_indices_in_file' in batch:
                indices = batch['original_indices_in_file']
                if hasattr(indices, 'cpu'):
                    indices = indices.cpu().numpy()
                f.write(f"Indices in file: {indices}\n")
            
            # Log FEN strings if available
            if 'fen' in batch:
                fens = batch['fen']
                if isinstance(fens, np.ndarray):
                    for i, fen in enumerate(fens):
                        f.write(f"Position {i}: {fen}\n")
                        if i >= 5:  # Limit to first 5 positions
                            f.write("... (truncated)\n")
                            break
            
            # Check for obvious anomalies in the data
            bitboards = batch['bitboards']
            if hasattr(bitboards, 'isnan') and bitboards.isnan().any():
                f.write(f"Found NaN values in bitboards\n")
            if hasattr(bitboards, 'isinf') and bitboards.isinf().any():
                f.write(f"Found Inf values in bitboards\n")
            
            f.write("=" * 50 + "\n")
    except Exception as e:
        print(f"Error logging NaN batch: {e}")

def compute_dynamic_loss_weights(cfg: dict, current_step: int, max_steps: int) -> Dict[str, float]:
    """
    Compute dynamic loss weights based on training progress.
    
    Args:
        cfg: Configuration dictionary containing loss_weights and optional loss_weight_schedules
        current_step: Current training step
        max_steps: Total number of training steps
        
    Returns:
        Dictionary of loss weight names to current weight values
    """
    # Start with static weights as defaults
    weights = dict(cfg['loss_weights'])
    
    # Check if dynamic scheduling is configured
    if 'loss_weight_schedules' not in cfg:
        return weights
    
    schedules = cfg['loss_weight_schedules']
    
    for loss_name, schedule_config in schedules.items():
        if loss_name not in weights:
            continue
            
        start_weight = schedule_config.get('start_weight', weights[loss_name])
        end_weight = schedule_config.get('end_weight', weights[loss_name])
        schedule_type = schedule_config.get('schedule_type', 'linear')
        
        # New timing parameters
        start_step = schedule_config.get('start_step', 0)
        duration_steps = schedule_config.get('duration_steps', max_steps)
        
        # Calculate end step
        end_step = start_step + duration_steps
        
        # Determine current weight based on timing
        if current_step <= start_step:
            # Before transition starts
            weights[loss_name] = start_weight
        elif current_step >= end_step:
            # After transition ends
            weights[loss_name] = end_weight
        else:
            # During transition - compute progress within the transition period
            transition_progress = (current_step - start_step) / duration_steps
            
            # Compute interpolation factor based on schedule type
            if schedule_type == 'linear':
                factor = transition_progress
            elif schedule_type == 'cosine':
                factor = 0.5 * (1 + np.cos(np.pi * (1 - transition_progress)))  # Cosine decay
            elif schedule_type == 'exponential':
                decay_rate = schedule_config.get('decay_rate', 0.1)
                factor = np.exp(-decay_rate * transition_progress * 10)  # Exponential decay
            else:
                factor = transition_progress  # Default to linear
            
            # Interpolate between start and end weights
            if schedule_type in ['linear', 'exponential']:
                # For linear and exponential: interpolate from start to end
                weights[loss_name] = start_weight + factor * (end_weight - start_weight)
            elif schedule_type == 'cosine':
                # For cosine: smooth transition from start to end
                weights[loss_name] = end_weight + factor * (start_weight - end_weight)
    
    return weights

def main(config_path: str, email_address: Optional[str] = None):
    cfg = load_config(config_path)
    
    # Setup email notifications if email provided
    email_notifier = None
    training_start_time = datetime.now()
    if email_address:
        setup_email_logging()
        email_notifier = TrainingEmailNotifier(
            recipient_email=email_address,
            send_time_hour=22,  # 10:00 PM
            send_time_minute=0
        )
        if email_notifier.is_credentials_available():
            logging.info(f"Email notifications enabled for {email_address}")
            # Send a test email to confirm everything is working
            email_notifier.send_start_notification(
                config_path=config_path, 
                max_steps=cfg['runtime']['max_steps']
            )
        else:
            logging.warning("Email address provided but credentials not available. Set TRAINING_EMAIL_SENDER and TRAINING_EMAIL_PASSWORD environment variables.")
            email_notifier = None
    
    # Simplified type casting since we removed contrastive components
    opt_cfg = cfg['optimiser']
    opt_cfg['lr'] = float(opt_cfg['lr'])
    opt_cfg['weight_decay'] = float(opt_cfg['weight_decay'])
    opt_cfg['betas'] = tuple(map(float, opt_cfg['betas']))
    opt_cfg['warmup_steps'] = int(opt_cfg['warmup_steps'])
    
    ds_cfg = cfg['dataset']
    ds_cfg['batch_size'] = int(ds_cfg['batch_size'])
    ds_cfg['num_workers'] = int(ds_cfg['num_workers'])
    ds_cfg['type'] = ds_cfg.get('type', 'tensor')
    ds_cfg['tensor_glob_pattern'] = ds_cfg.get('tensor_glob_pattern', '*.npz')
    
    rt = cfg['runtime']
    rt['max_steps'] = int(rt['max_steps'])
    rt['log_every'] = int(rt['log_every'])
    rt['ckpt_every'] = int(rt['ckpt_every'])
    rt['gradient_clip_norm'] = float(rt['gradient_clip_norm'])
    rt['gradient_clip_value'] = float(rt.get('gradient_clip_value', 5.0))  # Add value clipping
    rt['val_every'] = int(rt.get('val_every', 20000))

    model_cfg_types = cfg['model']
    model_cfg_types['dim'] = int(model_cfg_types['dim'])
    model_cfg_types['depth'] = int(model_cfg_types['depth'])
    model_cfg_types['early_depth'] = int(model_cfg_types['early_depth'])
    model_cfg_types['heads'] = int(model_cfg_types['heads'])

    pool_every_k_config_val = model_cfg_types.get('pool_every_k_blocks')
    if pool_every_k_config_val is not None:
        model_cfg_types['pool_every_k_blocks'] = int(pool_every_k_config_val)
    else:
        model_cfg_types['pool_every_k_blocks'] = None
    model_cfg_types['cls_dropout_rate'] = float(model_cfg_types.get('cls_dropout_rate', 0.0))
    model_cfg_types['cls_pool_alpha_init'] = float(model_cfg_types.get('cls_pool_alpha_init', 1.0))
    model_cfg_types['cls_pool_alpha_requires_grad'] = bool(model_cfg_types.get('cls_pool_alpha_requires_grad', True))
    model_cfg_types['policy_head_conv_dim'] = int(model_cfg_types.get('policy_head_conv_dim', 128))
    model_cfg_types['policy_head_mlp_hidden_dim'] = int(model_cfg_types.get('policy_head_mlp_hidden_dim', 256))
    model_cfg_types['drop_path'] = float(model_cfg_types.get('drop_path', 0.1))
    model_cfg_types['value_head_dropout_rate'] = float(model_cfg_types.get('value_head_dropout_rate', 0.0))
    model_cfg_types['dim_head'] = int(model_cfg_types.get('dim_head', 64))

    # Setup device
    if cfg['runtime']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        print("INFO: Training on CUDA.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")  # Faster matmuls
    else:
        device = torch.device("cpu")
        if cfg['runtime']['device'] == 'cuda':
            print("WARNING: CUDA specified in config but not available. Training on CPU.")
        else:
            print("INFO: Training on CPU.")

    seed_everything(cfg.get('seed', 42))

    # Build model (simplified without contrastive components)
    logging.info("Creating model...")
    model = ViTChess(
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        early_depth=cfg['model']['early_depth'],
        heads=cfg['model']['heads'],
        drop_path=cfg['model'].get('drop_path', 0.1), 
        num_policy_planes=cfg['model'].get('num_policy_planes', 73),
        num_value_outputs=cfg['model'].get('num_value_outputs', 3),
        num_material_categories=cfg['model'].get('num_material_categories', 20),
        num_moves_left_outputs=cfg['model'].get('num_moves_left_outputs', 1),
        policy_cls_projection_dim=cfg['model'].get('policy_cls_projection_dim', 64),
        policy_mlp_hidden_dim=cfg['model'].get('policy_head_mlp_hidden_dim'),
        pool_every_k_blocks=cfg['model'].get('pool_every_k_blocks'),
        cls_pool_alpha_init=cfg['model'].get('cls_pool_alpha_init', 1.0),
        cls_pool_alpha_requires_grad=cfg['model'].get('cls_pool_alpha_requires_grad', True),
        cls_dropout_rate=cfg['model'].get('cls_dropout_rate', 0.0),
        policy_head_conv_dim=cfg['model'].get('policy_head_conv_dim', 128),
        policy_head_mlp_hidden_dim=cfg['model'].get('policy_head_mlp_hidden_dim', 256),
        value_head_dropout_rate=cfg['model'].get('value_head_dropout_rate', 0.0),
        dim_head=cfg['model'].get('dim_head', 64),
        adaptive_pool_temperature=cfg['model'].get('adaptive_pool_temperature'),
        # SmolGen parameters
        smolgen_start_layer=cfg['model'].get('smolgen_start_layer', 2),
        smolgen_latent_dim=cfg['model'].get('smolgen_latent_dim', 256),
        smolgen_dropout=cfg['model'].get('smolgen_dropout', 0.1),
        # Value head parameters
        value_spatial_compress_dim=cfg['model'].get('value_spatial_compress_dim', 16),
        value_head_mlp_dims=cfg['model'].get('value_head_mlp_dims', [256]),
        moves_left_spatial_compress_dim=cfg['model'].get('moves_left_spatial_compress_dim', 8),
        moves_left_head_mlp_dims=cfg['model'].get('moves_left_head_mlp_dims', [128]),
        # Hybrid patch embedding parameters
        patch_resblock_hidden=cfg['model'].get('patch_resblock_hidden', 32),
        patch_global_proj_dim=cfg['model'].get('patch_global_proj_dim', 16),
        patch_embed_dropout=cfg['model'].get('patch_embed_dropout', 0.1),
    )
        
    model.to(device)
    logging.info(f"Model placed on device: {device}")

    # Automatically handle torch.compile + gradient_checkpointing conflict
    use_compile = cfg['runtime'].get('compile_model', False)
    use_ckpt = cfg['runtime'].get('gradient_checkpointing', True)
    
    if use_compile and use_ckpt:
        print("INFO: torch.compile and gradient_checkpointing conflict detected.")
        print("      Prioritizing torch.compile for throughput (since you're not memory-constrained).")
        print("      Disabling gradient_checkpointing automatically.")
        use_ckpt = False
    
    # Enable gradient checkpointing if appropriate
    if use_ckpt:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")
    else:
        print("Gradient checkpointing disabled")

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: Total={total_params/1e6:.2f}M, Trainable={trainable_params/1e6:.2f}M")

    # Display key training configuration for verification
    print(f"Training configuration:")
    print(f"  Max steps: {rt['max_steps']:,}")
    print(f"  Batch size: {ds_cfg['batch_size']:,}")
    print(f"  Log every: {rt['log_every']:,} steps")
    print(f"  Checkpoint every: {rt['ckpt_every']:,} steps") 
    print(f"  Validate every: {rt['val_every']:,} steps")
    print(f"  Learning rate: {opt_cfg['lr']:.2e}")
    if 'loss_weight_schedules' in cfg:
        print(f"  Dynamic loss weights: {len(cfg['loss_weight_schedules'])} schedules configured")
    else:
        print(f"  Dynamic loss weights: None (using static weights)")

    # Improved mixed precision setup
    use_amp = cfg['runtime']['precision'] == 'fp16'
    
    # Detect best dtype for hardware
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
        print("Using bfloat16 for mixed precision")
    else:
        autocast_dtype = torch.float16
        print("Using float16 for mixed precision")
    
    scaler = GradScaler(enabled=use_amp)
    
    if use_amp:
        logging.info(f"Mixed precision training ENABLED with {autocast_dtype}.")
    else:
        logging.info("Mixed precision training DISABLED (using fp32).")

    # Try to compile the model (PyTorch 2.0+)
    if hasattr(torch, 'compile') and cfg['runtime'].get('compile_model', False):
        try:
            logging.info("Attempting to compile model with torch.compile()...")
            
            # Configure compiler for stability
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 64
            
            # Get compilation mode and CUDA graph settings
            compile_mode = cfg['runtime'].get('compile_mode', 'default')
            original_grad_accum = ds_cfg['grad_accum']
            enable_cudagraphs = cfg['runtime'].get('compile_with_cudagraphs', False)
            
            if enable_cudagraphs and torch.cuda.is_available():
                # CUDA graphs require grad_accum=1 for memory safety
                if original_grad_accum > 1:
                    logging.info(f"CUDA graphs enabled: adjusting grad_accum from {original_grad_accum} to 1")
                    logging.info("Learning rate will be scaled proportionally to maintain effective batch size")
                    ds_cfg['grad_accum'] = 1
                    # Scale learning rate to maintain effective batch size
                    opt_cfg['lr'] *= original_grad_accum
                    logging.info(f"Scaled learning rate to {opt_cfg['lr']:.2e}")
                
                # Enable CUDA graphs with specified mode
                if compile_mode == 'max-autotune':
                    logging.info("Using max-autotune mode for maximum performance (longer compilation time)")
                    model = torch.compile(model, mode="max-autotune")
                else:
                    model = torch.compile(model, mode="reduce-overhead")
                logging.info(f"Model compiled successfully with CUDA graphs enabled (mode: {compile_mode}).")
            else:
                # Disable CUDA graphs for safer async prefetch compatibility  
                if torch.cuda.is_available():
                    torch._inductor.config.triton.cudagraphs = False
                    logging.info("Disabled CUDA graphs for safer async prefetch compatibility.")
                
                # Use custom options for safety
                if compile_mode == 'max-autotune':
                    logging.info("Using max-autotune mode for maximum performance (longer compilation time)")
                    model = torch.compile(
                        model,
                        mode="max-autotune",
                        options={"triton.cudagraphs": False}  # Disable CUDA graphs for safety
                    )
                else:
                    model = torch.compile(
                        model,
                        options={"triton.cudagraphs": False}  # Disable CUDA graphs for safety
                    )
                logging.info(f"Model compiled successfully with CUDA graph safety measures (mode: {compile_mode}).")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}. Proceeding with uncompiled model.")
  
    # Optimizer and scheduler
    logging.info("Creating optimizer and scheduler...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt_cfg['lr'],
        weight_decay=opt_cfg['weight_decay'],
        betas=(opt_cfg['betas'][0], opt_cfg['betas'][1])
    )
    
    warmup_steps = opt_cfg['warmup_steps']
    total_steps = rt['max_steps']

    if opt_cfg['sched'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps)
        )
    elif opt_cfg['sched'] == 'constant':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=max(1, total_steps))
    else:
        raise ValueError(f"Unknown scheduler: {opt_cfg['sched']}")

    # DataLoader
    loader = make_dataset(
        dataset_type=ds_cfg['type'],
        data_dir=ds_cfg['data_dir'],
        batch_size=ds_cfg['batch_size'],
        num_workers=ds_cfg['num_workers'],
        seed=cfg.get('seed', 42),
        tensor_glob_pattern=ds_cfg['tensor_glob_pattern'],
        shuffle_files=True,
        infinite=True,
        pad_partial_batches=cfg.get('pad_partial_batches', True)
    )
    data_iter = iter(loader)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=cfg['logging']['output_dir']) if cfg['logging']['tensorboard'] else None

    # Rolling metrics (simplified without contrastive components)
    rm = cfg.get('rolling_metrics', {})
    window = rm.get('window_size', 1000)
    metric_names = ['policy', 'value', 'moves_left', 'auxiliary_value', 'material', 'total', 'compare_lc0']
    metrics = {name: deque(maxlen=window) for name in metric_names}

    # Throughput tracking
    last_log_time = time.time()
    last_log_step = 0
    data_loading_times = deque(maxlen=window)

    # Loss history structure (simplified)
    loss_history: Dict[str, Dict[str, Any]] = {name: {'train': [], 'steps_train': [], 'val': []} for name in metric_names}
    val_steps_history: List[int] = []

    # Training loop
    step = 0
    grad_accum = ds_cfg['grad_accum']
    log_every = rt['log_every']
    ckpt_every = rt['ckpt_every']
    val_every = rt['val_every']
    test_data_dir_from_cfg = ds_cfg['test_data_dir'] # Use dataset config for test_data_dir

    # Asynchronous GPU prefetch
    if device.type == "cuda":
        prefetch_stream = torch.cuda.Stream(device=device)
    else:
        prefetch_stream = None

    def _to_gpu_async(cpu_batch):
        if prefetch_stream is not None:
            with torch.cuda.stream(prefetch_stream):
                return move_batch_to_device(cpu_batch, device)
        return move_batch_to_device(cpu_batch, device)

    # Prime the pipeline
    first_cpu_batch = next(data_iter)
    next_gpu_batch = _to_gpu_async(first_cpu_batch)
    if prefetch_stream is not None:
        prefetch_stream.synchronize()

    try:
        while step < rt['max_steps']:
            model.train()
            if step % grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            # Mark CUDA graph step boundary for memory safety
            if torch.cuda.is_available() and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                # Only needed when CUDA graphs are disabled due to gradient accumulation
                if not cfg['runtime'].get('compile_with_cudagraphs', False) or ds_cfg['grad_accum'] > 1:
                    torch.compiler.cudagraph_mark_step_begin()

            # Get current prefetched batch & prefetch next
            if prefetch_stream is not None:
                torch.cuda.current_stream().wait_stream(prefetch_stream)

            batch = next_gpu_batch

            # Start prefetch for next iteration
            t_data_start = time.time()
            cpu_batch_next = next(data_iter)
            t_data_end = time.time()
            data_loading_times.append(t_data_end - t_data_start)

            next_gpu_batch = _to_gpu_async(cpu_batch_next)
            
            # Ensure all flag tensors are present
            current_B = batch['bitboards'].size(0)
            default_flags_kwargs = {}
            expected_flags = ['is960', 'stm', 'rep1', 'rep2']
            for flag_key in expected_flags:
                if batch.get(flag_key) is None:
                    default_flags_kwargs[flag_key] = torch.zeros(current_B, dtype=torch.float, device=device)
                else:
                    default_flags_kwargs[flag_key] = batch.get(flag_key)

            # Compute dynamic loss weights OUTSIDE autocast to prevent recompilation
            lw = compute_dynamic_loss_weights(cfg, step, rt['max_steps'])
            
            # Forward + loss
            with autocast(device_type='cuda', dtype=autocast_dtype):
                outputs = model(
                    batch['bitboards'],
                    is960=default_flags_kwargs['is960'],
                    stm=default_flags_kwargs['stm'],
                    rep1=default_flags_kwargs['rep1'],
                    rep2=default_flags_kwargs['rep2']
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
                    print("Warning: legal_mask is None in training loop.")
                    target_for_loss = policy_target_from_batch.to(device=logits.device) 
                
                # Compute losses
                loss_policy = _new_policy_loss_fn(target_for_loss, logits) 
                loss_value = label_smoothed_cross_entropy(outputs['final_value'], batch['value_target'], smoothing=0.04)
                loss_moves = _ply_loss_fn(outputs['final_moves_left'], batch['ply_target'])
                loss_aux = label_smoothed_cross_entropy(outputs['early_value'], batch['value_target'], smoothing=0.04)
                loss_material = F.cross_entropy(outputs['early_material'], batch['material_category'])
                loss_cls_sparsity = torch.tensor(0.0, device=device)
                if 'final_cls_features' in outputs:
                    loss_cls_sparsity = outputs['final_cls_features'].abs().mean()

                # Add SmolGen bias regularization for numerical stability
                bias_reg_loss = torch.tensor(0.0, device=device)
                if hasattr(model, 'shared_smolgen') and model.shared_smolgen is not None:
                    # L2 regularization on bias temperature to prevent extreme scaling
                    bias_reg_loss = 0.001 * (model.shared_smolgen.bias_temperature - 1.0) ** 2
                
                # Weighted total (use pre-computed weights)
                total_loss = (
                    lw['policy'] * loss_policy
                    + lw['value'] * loss_value
                    + lw['moves_left'] * loss_moves
                    + lw['auxiliary_value'] * loss_aux
                    + lw['material'] * loss_material
                    + bias_reg_loss
                )
                compare_lc0 = (
                    loss_policy
                    + 1.6 * loss_value
                    + 0.5 * loss_moves
                )

            # Check for NaN loss before backward pass
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"WARNING: NaN/Inf loss detected at step {step}. Skipping backward pass.")
                
                # Log SmolGen bias temperature for debugging
                if hasattr(model, 'shared_smolgen') and model.shared_smolgen is not None:
                    bias_temp = model.shared_smolgen.bias_temperature.item()
                    print(f"SmolGen bias temperature: {bias_temp:.6f}")
                    if abs(bias_temp) > 5.0:
                        print("WARNING: SmolGen bias temperature is very large!")
                
                log_nan_batch(batch, step, cfg['logging']['output_dir'])
                step += 1
                continue  # Skip this batch entirely

            # Backward
            scaler.scale(total_loss).backward()

            # Step optimizer with enhanced gradient clipping
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                
                # Apply both norm and value clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), rt['gradient_clip_norm'])
                torch.nn.utils.clip_grad_value_(model.parameters(), rt['gradient_clip_value'])
                
                # Additional safeguard: clamp SmolGen bias temperature to prevent runaway scaling
                if hasattr(model, 'shared_smolgen') and model.shared_smolgen is not None:
                    with torch.no_grad():
                        model.shared_smolgen.bias_temperature.clamp_(0.1, 10.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # Record metrics
            metrics['policy'].append(loss_policy.item())
            metrics['value'].append(loss_value.item())
            metrics['moves_left'].append(loss_moves.item())
            metrics['auxiliary_value'].append(loss_aux.item())
            metrics['material'].append(loss_material.item())
            metrics['total'].append(total_loss.item())
            metrics['compare_lc0'].append(compare_lc0.item())

            # Logging
            if (step + 1) % log_every == 0:
                try:
                    mean_metrics = {name: np.mean(metrics[name]) for name in metric_names}
                    mean_data_loading_time = np.mean(data_loading_times)
                    alpha_log_str = ""

                    if writer:
                        for name, val in mean_metrics.items():
                            loss_history[name]['train'].append(val)
                            loss_history[name]['steps_train'].append(step + 1)
                            writer.add_scalar(f'train/{name}', val, step + 1)
                        
                        # Log current dynamic loss weights
                        current_weights = compute_dynamic_loss_weights(cfg, step, rt['max_steps'])
                        for weight_name, weight_val in current_weights.items():
                            writer.add_scalar(f'loss_weights/{weight_name}', weight_val, step + 1)
                        
                        if hasattr(model, 'alphas') and model.alphas:
                            for i, alpha_param in enumerate(model.alphas):
                                writer.add_scalar(f'alpha_pool_{i}', alpha_param.item(), step + 1)
                        
                        # Log SmolGen bias temperature for monitoring
                        if hasattr(model, 'shared_smolgen') and model.shared_smolgen is not None:
                            bias_temp = model.shared_smolgen.bias_temperature.item()
                            writer.add_scalar('smolgen/bias_temperature', bias_temp, step + 1)

                    if hasattr(model, 'alphas') and model.alphas:
                        alpha_values_console = [f'{alpha_param.item():.3f}' for alpha_param in model.alphas]
                        alpha_log_str = f", alphas: [{', '.join(alpha_values_console)}]"
                    
                    # Add SmolGen bias temperature to console logging
                    bias_temp_str = ""
                    if hasattr(model, 'shared_smolgen') and model.shared_smolgen is not None:
                        bias_temp = model.shared_smolgen.bias_temperature.item()
                        bias_temp_str = f", bias_temp: {bias_temp:.3f}"
                    
                    current_time = time.time()
                    delta_steps = (step + 1) - last_log_step
                    positions = delta_steps * ds_cfg['batch_size']
                    elapsed = current_time - last_log_time
                    throughput = positions / elapsed if elapsed > 0 else float('inf')
                    
                    # Get email timing status if available
                    email_status_str = ""
                    if email_notifier:
                        try:
                            status = email_notifier.debug_email_status()
                            email_status_str = f"next_email: {status.get('next_email_update', 'N/A')}"
                        except Exception as e:
                            email_status_str = "next_email: error"
                            logging.warning(f"Could not get email debug status: {e}")

                    # Add dynamic weight info to console output
                    weight_info_str = ""
                    if 'loss_weight_schedules' in cfg:
                        current_weights = compute_dynamic_loss_weights(cfg, step, rt['max_steps'])
                        weight_parts = []
                        for loss_name in ['value', 'moves_left', 'material']:
                            if loss_name in cfg['loss_weight_schedules']:
                                # Add phase indicator 
                                schedule = cfg['loss_weight_schedules'][loss_name]
                                start_step = schedule.get('start_step', 0)
                                duration_steps = schedule.get('duration_steps', rt['max_steps'])
                                end_step = start_step + duration_steps
                                
                                if step <= start_step:
                                    phase = "pre"
                                elif step >= end_step:
                                    phase = "post"
                                else:
                                    phase = "active"
                                
                                weight_parts.append(f"{loss_name}={current_weights[loss_name]:.3f}({phase})")
                        if weight_parts:
                            weight_info_str = f"weights: {', '.join(weight_parts)}"

                    log_str_parts = [
                        f"[Step {step+1}/{rt['max_steps']}]",
                        f"total_loss: {mean_metrics['total']:.4f}",
                        f"lc0_loss: {mean_metrics['compare_lc0']:.4f}",
                        f"policy: {mean_metrics['policy']:.4f}",
                        f"value: {mean_metrics['value']:.4f}",
                        f"moves_left: {mean_metrics['moves_left']:.4f}",
                        f"aux_val: {mean_metrics['auxiliary_value']:.4f}",
                        f"material: {mean_metrics['material']:.4f}",
                        f"{throughput:.1f} pos/s",
                        f"data_load: {mean_data_loading_time:.4f}s"
                    ]
                    if alpha_log_str:
                        log_str_parts.append(alpha_log_str.strip(', '))
                    if bias_temp_str:
                        log_str_parts.append(bias_temp_str.strip(', '))
                    if weight_info_str:
                        log_str_parts.append(weight_info_str)
                    if email_status_str:
                        log_str_parts.append(email_status_str)

                    print(", ".join(log_str_parts), flush=True)
                    
                    last_log_time = current_time
                    last_log_step = step + 1

                    if writer and cfg['logging']['matplotlib']:
                        plot_all_losses(loss_history, val_steps_history, cfg['logging']['output_dir'])
                    
                    # Check if it's time to send daily email update
                    if email_notifier:
                        email_notifier.send_training_update(
                            step=step + 1,
                            max_steps=rt['max_steps'],
                            mean_metrics=mean_metrics,
                            throughput=throughput,
                            output_dir=cfg['logging']['output_dir'],
                            training_start_time=training_start_time
                        )

                    # Clear CUDA cache to manage memory growth
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"ERROR in logging section at step {step+1}: {e}")
                    import traceback
                    traceback.print_exc()

            # Validation step (simplified for single source)
            if (step + 1) % val_every == 0:
                print(f"\n--- Starting Validation at Step {step+1} ---")
                was_training = model.training
                model.eval()

                test_base_dir = test_data_dir_from_cfg
                if not os.path.isdir(test_base_dir):
                    print(f"Warning: Test data directory {test_base_dir} not found. Skipping validation.")
                else:
                    val_steps_history.append(step + 1)
                    # Use current dynamic loss weights for validation too
                    current_val_weights = compute_dynamic_loss_weights(cfg, step, rt['max_steps'])
                    avg_val_losses = evaluate_model(
                        model, device, test_base_dir,
                        ds_cfg['batch_size'], ds_cfg['num_workers'],
                        ds_cfg['tensor_glob_pattern'], current_val_weights,
                        seed=cfg.get('seed', 42),
                        test_batch_size=ds_cfg.get('test_batch_size')
                    )
                    print(f"Validation Results (Step {step+1}):")
                    log_val_message_parts = []
                    for loss_n, loss_v in avg_val_losses.items():
                        loss_history[loss_n]['val'].append(loss_v)
                        if writer:
                            writer.add_scalar(f'val/{loss_n}', loss_v, step + 1)
                        if loss_n in ['total', 'policy', 'value', 'compare_lc0']:
                            log_val_message_parts.append(f"{loss_n}: {loss_v:.4f}")
                    print("  " + ", ".join(log_val_message_parts))
                
                if was_training:
                    model.train()
                print(f"--- Finished Validation at Step {step+1} ---\n")

                torch.cuda.empty_cache()

            # Checkpoint
            if (step + 1) % ckpt_every == 0:
                save_checkpoint(step + 1, model, optimizer, scheduler, scaler, cfg, cfg['logging']['output_dir'])
                # Force garbage collection after checkpoint
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            step += 1



    except KeyboardInterrupt:
        print("Interrupted. Saving final checkpoint...")
        save_checkpoint(step, model, optimizer, scheduler, scaler, cfg, cfg['logging']['output_dir'])
        
        # Send interruption email
        if email_notifier:
            final_metrics = {name: np.mean(metrics[name]) if metrics[name] else 0.0 for name in metric_names}
            email_notifier.send_training_complete_email(
                final_step=step,
                final_metrics=final_metrics,
                output_dir=cfg['logging']['output_dir'],
                training_start_time=training_start_time
            )
    finally:
        if writer:
            if cfg['logging']['matplotlib'] and loss_history:
                print("Generating final plots...")
                plot_all_losses(loss_history, val_steps_history, cfg['logging']['output_dir'])
            writer.close()
        
        # Send completion email if training finished normally
        if step >= rt['max_steps'] and email_notifier:
            final_metrics = {name: np.mean(metrics[name]) if metrics[name] else 0.0 for name in metric_names}
            email_notifier.send_training_complete_email(
                final_step=step,
                final_metrics=final_metrics,
                output_dir=cfg['logging']['output_dir'],
                training_start_time=training_start_time
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Chess-ViT")
    parser.add_argument('--config', '-c', default='config.yaml', help='Path to config file')
    parser.add_argument('--email', help='Email address for daily training updates (sent at 10:00 PM)')
    args = parser.parse_args()
    main(args.config, args.email)