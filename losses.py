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
    # Ensure target is long type for F.cross_entropy
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
    proc_out = processed_output_ply.float() / scale
    proc_tgt = processed_target_ply.float() / scale
    # Delta computed relative to scale
    delta = 10.0 / scale
    return F.huber_loss(proc_out, proc_tgt, reduction="mean", delta=delta)


def custom_policy_loss_fn(policy_logits: torch.Tensor, 
                          policy_target: torch.Tensor, 
                          legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Custom policy loss using cross-entropy, masking illegal moves.
    Args:
        policy_logits: Raw logits from model (B, num_policy_planes, H, W).
        policy_target: Target policy distribution (B, num_policy_planes, H, W), sums to 1 over legal moves.
        legal_mask: Boolean mask (B, num_policy_planes, H, W), True for legal moves. If None, assumes all are legal.
    """
    B, P, H, W = policy_logits.shape
    
    policy_logits_flat = policy_logits.reshape(B, -1)  # (B, P*H*W)
    policy_target_flat = policy_target.reshape(B, -1)  # (B, P*H*W)

    if legal_mask is not None:
        legal_mask_flat = legal_mask.reshape(B, -1)    # (B, P*H*W)
        # Set logits for illegal moves to a very small number (negative infinity effectively)
        policy_logits_flat = policy_logits_flat.masked_fill(~legal_mask_flat, torch.finfo(policy_logits_flat.dtype).min)
    
    log_probs = F.log_softmax(policy_logits_flat, dim=1)
    
    # Cross-entropy: sum over actions for each batch item, then mean over batch
    # policy_target_flat should be probabilities (e.g., from MCTS visits, sum to 1 per sample)
    loss_per_sample = - (policy_target_flat * log_probs).sum(dim=1)
    
    # Handle potential NaNs or Infs if a sample had no legal moves and target sum was zero.
    # If all logits became -inf due to no legal moves, log_softmax can be tricky.
    # masked_fill with finfo.min should be safe for softmax, but target might be all zeros.
    if torch.isnan(loss_per_sample).any() or torch.isinf(loss_per_sample).any():
        num_problematic = torch.isnan(loss_per_sample).sum() + torch.isinf(loss_per_sample).sum()
        print(f"Warning: NaN or Inf detected in policy loss for {num_problematic} / {B} samples. Replacing with 0.")
        loss_per_sample = torch.nan_to_num(loss_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
        
    return loss_per_sample.mean()


def nt_xent_loss_fn(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1, eps: float = 1e-6) -> torch.Tensor:
    """
    NT-Xent loss for contrastive learning.
    Args:
        z1: Batch of embeddings from the first view (B, D).
        z2: Batch of embeddings from the second view (B, D). Positive pairs for z1.
        temperature: Temperature scaling factor.
        eps: Small epsilon for numerical stability in normalization.
    """
    batch_size = z1.shape[0]
    if batch_size == 0: # Handle empty inputs gracefully
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)

    z1_norm = F.normalize(z1, p=2, dim=1, eps=eps)
    z2_norm = F.normalize(z2, p=2, dim=1, eps=eps)

    representations = torch.cat([z1_norm, z2_norm], dim=0)  # Shape (2*B, D)
    similarity_matrix = torch.mm(representations, representations.t()) # Shape (2*B, 2*B)
    
    # Create labels for positive pairs
    # For z1_norm[i], its positive pair is z2_norm[i] which is at index i + batch_size in `representations`
    # For z2_norm[i], its positive pair is z1_norm[i] which is at index i in `representations`
    labels_row1 = torch.arange(batch_size, device=z1.device) + batch_size # z1_norm[i] positive is z2_norm[i]
    labels_row2 = torch.arange(batch_size, device=z1.device)              # z2_norm[i] positive is z1_norm[i]
    labels = torch.cat([labels_row1, labels_row2], dim=0)

    # The standard F.cross_entropy handles the softmax internally and compares with target labels.
    # The `labels` created above correctly identify these positive pairs for F.cross_entropy.

    similarity_matrix = similarity_matrix / temperature
    
    # Cross-entropy loss
    # Logits are the rows of similarity_matrix, targets are the 'labels'
    return F.cross_entropy(similarity_matrix, labels)

# --- New Policy Loss Function ---
def new_policy_loss_fn(target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """New policy loss as proposed by user.
    Assumes target has -1 for illegal moves.
    Args:
        target: Target policy (B, N) or (B, P, H, W) with -1 for illegal moves.
        output: Output logits (B, N) or (B, P, H, W).
    """
    # Reshape if necessary
    if output.ndim == 4:
        output = output.reshape(output.shape[0], -1)
    if target.ndim == 4:
        target = target.reshape(target.shape[0], -1)

    # Illegal moves are marked by a value of -1 in the labels - we mask these with large negative values
    output_masked = output.masked_fill(target < 0, -1e4) 
    
    # The large negative values will still break the loss, so we replace them with 0 once we finish masking
    target_processed = F.relu(target) 
    
    log_prob = F.log_softmax(output_masked, dim=1) 
    
    nll_per_sample = -(target_processed * log_prob).sum(dim=1)
    nll = nll_per_sample.mean() 
    
    return nll

# --- Combined Loss Calculation ---

class ModelLosses:
    def __init__(self,
                 wdl_early_weight: float = 1.0,
                 material_early_weight: float = 1.0,
                 ply_early_weight: float = 1.0,
                 wdl_final_weight: float = 1.0,
                 ply_final_weight: float = 1.0,
                 policy_weight: float = 1.0,
                 contrastive_weight: float = 1.0,
                 ply_loss_scale: float = 20.0,
                 nt_xent_temperature: float = 0.1):
        self.weights = {
            "wdl_early": wdl_early_weight,
            "material_early": material_early_weight,
            "wdl_final": wdl_final_weight,
            "ply_final": ply_final_weight,
            "policy": policy_weight,
            "contrastive": contrastive_weight
        }
        self.ply_loss_scale = ply_loss_scale
        self.nt_xent_temperature = nt_xent_temperature

    def calculate_total_loss(self, 
                             model_outputs: Dict[str, torch.Tensor], 
                             targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the total weighted loss and individual loss components.
        Args:
            model_outputs: Dictionary from ViTChess.forward(). Expected keys like 
                           'early_value', 'early_material', 'early_moves_left',
                           'final_value', 'final_moves_left', 'policy',
                           'contrastive_cls_v_flip', 'contrastive_cls_h_flip', 
                           'contrastive_cls_hv_flip'.
                           Also needs 'cls_early_features' if contrastive_weight > 0.
            targets: Dictionary of target tensors. Expected keys: 
                     'wdl_target', 'material_target', 'ply_target', 
                     'policy_target', 'legal_mask'.
        Returns:
            total_loss: Scalar tensor for backpropagation.
            individual_losses: Dictionary of scalar float loss values for logging.
        """
        individual_losses_dict: Dict[str, float] = {}
        
        if not model_outputs:
            print("Warning: model_outputs is empty in calculate_total_loss. Defaulting loss to 0 on CPU.")
            return torch.tensor(0.0, device='cpu'), individual_losses_dict

        example_tensor = None
        for val in model_outputs.values():
            if isinstance(val, torch.Tensor):
                example_tensor = val
                break
        
        if example_tensor is None:
            print("Warning: No Tensors found in model_outputs. Defaulting loss to 0 on CPU.")
            return torch.tensor(0.0, device='cpu'), individual_losses_dict

        current_device = example_tensor.device
        current_dtype = example_tensor.dtype
        total_loss = torch.tensor(0.0, device=current_device, dtype=current_dtype)

        # --- Early Heads ---
        if 'early_value' in model_outputs and model_outputs['early_value'] is not None and 'wdl_target' in targets:
            loss = wdl_loss_fn(model_outputs['early_value'], targets['wdl_target'])
            individual_losses_dict['loss_wdl_early'] = loss.item()
            total_loss += self.weights.get('wdl_early', 0.0) * loss

        if 'early_material' in model_outputs and model_outputs['early_material'] is not None and 'material_target' in targets:
            loss = material_loss_fn(model_outputs['early_material'], targets['material_target'])
            individual_losses_dict['loss_material_early'] = loss.item()
            total_loss += self.weights.get('material_early', 0.0) * loss

        # --- Final Heads ---
        if 'final_value' in model_outputs and model_outputs['final_value'] is not None and 'wdl_target' in targets:
            loss = wdl_loss_fn(model_outputs['final_value'], targets['wdl_target'])
            individual_losses_dict['loss_wdl_final'] = loss.item()
            total_loss += self.weights.get('wdl_final', 0.0) * loss

        if 'final_moves_left' in model_outputs and model_outputs['final_moves_left'] is not None and 'ply_target' in targets:
            loss = ply_loss_fn(model_outputs['final_moves_left'], targets['ply_target'], scale=self.ply_loss_scale)
            individual_losses_dict['loss_ply_final'] = loss.item()
            total_loss += self.weights.get('ply_final', 0.0) * loss
            
        # --- Policy Head ---
        if 'policy' in model_outputs and model_outputs['policy'] is not None and 'policy_target' in targets:
            legal_mask = targets.get('legal_mask')
            if legal_mask is None:
                print("Warning: legal_mask not found in targets for policy loss. Assuming all moves are legal.")
            loss = custom_policy_loss_fn(model_outputs['policy'], targets['policy_target'], legal_mask)
            individual_losses_dict['loss_policy'] = loss.item()
            total_loss += self.weights.get('policy', 0.0) * loss
            
        # --- Contrastive Loss ---
        anchor_features = model_outputs.get('cls_early_features') 

        if anchor_features is not None and self.weights.get('contrastive', 0.0) > 0:
            contrastive_loss_components_sum = torch.tensor(0.0, device=current_device, dtype=current_dtype)
            num_contrastive_contributions = 0
            
            flip_keys_in_model_output = [
                'contrastive_cls_v_flip', 
                'contrastive_cls_h_flip', 
                'contrastive_cls_hv_flip'
            ]
            
            for flip_key in flip_keys_in_model_output:
                if flip_key in model_outputs and model_outputs[flip_key] is not None:
                    positive_features = model_outputs[flip_key]
                    if positive_features.numel() > 0 and anchor_features.numel() > 0 and \
                       positive_features.shape[0] == anchor_features.shape[0] and anchor_features.shape[0] > 0:
                        
                        loss = nt_xent_loss_fn(anchor_features, positive_features, temperature=self.nt_xent_temperature)
                        log_key_suffix = flip_key.replace('contrastive_cls_', '')
                        individual_losses_dict[f'loss_contrastive_{log_key_suffix}'] = loss.item()
                        contrastive_loss_components_sum += loss
                        num_contrastive_contributions += 1
            
            if num_contrastive_contributions > 0:
                avg_contrastive_loss = contrastive_loss_components_sum / num_contrastive_contributions
                individual_losses_dict['loss_contrastive_avg'] = avg_contrastive_loss.item()
                total_loss += self.weights['contrastive'] * avg_contrastive_loss
            elif self.weights.get('contrastive', 0.0) > 0:
                individual_losses_dict['loss_contrastive_avg'] = 0.0 
                print("Warning: Contrastive loss weight is > 0 but no valid contrastive pairs were processed.")

        individual_losses_dict['total_combined_loss'] = total_loss.item()
        return total_loss, individual_losses_dict 