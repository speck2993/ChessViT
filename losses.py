import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# --- Individual Loss Functions ---

def wdl_loss_fn(output_logits: torch.Tensor, target_wdl: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for WDL prediction.
    Args:
        output_logits: Raw logits from the model (B, num_value_outputs=3).
        target_wdl: Target class indices (B,), where 0:loss, 1:draw, 2:win (example).
    """
    return F.cross_entropy(output_logits, target_wdl.long())

def material_loss_fn(output_logits: torch.Tensor, target_material_cat: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for material category prediction.
    Args:
        output_logits: Raw logits from the model (B, num_material_categories).
        target_material_cat: Target class indices (B,).
    """
    return F.cross_entropy(output_logits, target_material_cat.long())

def ply_loss_fn(output_ply: torch.Tensor, target_ply: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    """Scaled Huber loss for ply-to-end prediction.
    Args:
        output_ply: Predicted ply count from the model (B, 1) or (B,).
        target_ply: Target ply count (B, 1) or (B,).
        scale: Scaling factor to normalize target and output.
    """
    # Squeeze last dim if it's 1
    processed_output_ply = output_ply.squeeze(-1) if output_ply.ndim > 1 and output_ply.shape[-1] == 1 else output_ply
    processed_target_ply = target_ply.squeeze(-1) if target_ply.ndim > 1 and target_ply.shape[-1] == 1 else target_ply
    # Scale both
    proc_out = processed_output_ply.float() / (2*scale)
    proc_tgt = processed_target_ply.float() / (2*scale)
    # Delta computed relative to scale
    delta = 10.0 / (2*scale)
    return F.huber_loss(proc_out, proc_tgt, reduction="mean", delta=delta)

def new_policy_loss_fn(target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Numerically stable policy loss with proper handling of edge cases.
    
    Args:
        target: Target policy (B, N) with -1 for illegal moves
        output: Output logits (B, N)
    """
    # Reshape if necessary
    if output.ndim == 4:
        output = output.reshape(output.shape[0], -1)
    if target.ndim == 4:
        target = target.reshape(target.shape[0], -1)
    
    # Identify legal moves
    legal_mask = target >= 0  # Shape: (B, N)
    
    # Check if any sample has no legal moves
    legal_moves_per_sample = legal_mask.sum(dim=1)  # Shape: (B,)
    has_legal_moves = legal_moves_per_sample > 0
    
    if not has_legal_moves.all():
        num_invalid = (~has_legal_moves).sum().item()
        print(f"Warning: {num_invalid} samples have no legal moves in policy loss")
    
    # Use a more reasonable masking value that works well with fp16
    mask_value = -1e3 if output.dtype == torch.float16 else -1e4
    
    # Apply mask to illegal moves
    output_masked = output.masked_fill(~legal_mask, mask_value)
    
    # Compute log_softmax with numerical stability
    output_max = output_masked.max(dim=1, keepdim=True)[0].detach()
    output_stable = output_masked - output_max
    log_prob = F.log_softmax(output_stable, dim=1)
    
    # Process targets (set illegal moves to 0)
    target_processed = F.relu(target)
    
    # Compute negative log likelihood
    nll_per_sample = -(target_processed * log_prob).sum(dim=1)
    
    # Handle samples with no legal moves
    nll_per_sample = torch.where(has_legal_moves, nll_per_sample, torch.zeros_like(nll_per_sample))
    
    # Check for NaN/Inf before returning
    if torch.isnan(nll_per_sample).any() or torch.isinf(nll_per_sample).any():
        num_nan = torch.isnan(nll_per_sample).sum().item()
        num_inf = torch.isinf(nll_per_sample).sum().item()
        print(f"Warning: Policy loss has {num_nan} NaN and {num_inf} Inf values")
        # Replace NaN/Inf with 0
        nll_per_sample = torch.nan_to_num(nll_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
    
    return nll_per_sample.mean() 