from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from safetensors.torch import load_file
import numpy as np
import yaml

def load_config(path: str) -> dict:
    # Ensure UTF-8 encoding when reading config to avoid locale decoding errors
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

NUM_BITBOARD_PLANES = 14  # 12 piece planes + castling + en-passant

# ---------------------------------------------------------------------------
# Utility layers & helpers
# ---------------------------------------------------------------------------
class Mlp(nn.Module):
    """Lightweight feed‑forward block (Linear → GELU → Linear)."""
    def __init__(self, dim: int, hidden: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden = hidden or dim * 4
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate='tanh')  # Faster GELU approximation
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA with optional additive bias‑matrix (65×65).

    Args:
        bias (torch.Tensor | None): 65×65 distance matrix. Registered as a
            *parameter* so it can be frozen / unfrozen via .requires_grad_.
            For per-head bias, this initial bias will be repeated for each head.
        bias_scale (float): Scalar scale that is *always* trainable (helps the
            optimiser learn whether to trust the bias or not).
    """
    def __init__(self, dim: int, heads: int, bias: Optional[torch.Tensor] = None):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Bias handling ------------------------------------------------------
        if bias is None:
            # Initialize as (heads, 65, 65)
            initial_bias_data = torch.zeros(heads, 65, 65)
        else:
            # Repeat the provided (65,65) bias for each head
            initial_bias_data = bias.unsqueeze(0).repeat(heads, 1, 1)
        
        self.bias = nn.Parameter(initial_bias_data) # Shape: (heads, 65, 65)
                                                    # Caller decides .requires_grad for the initial state
        self.bias_scale = nn.Parameter(torch.tensor(1.0))  # *always* learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Check if we can use Flash Attention (requires PyTorch 2.0+)
        try:
            from torch.nn.functional import scaled_dot_product_attention
            USE_FLASH = True
        except ImportError:
            USE_FLASH = False
        
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Use Flash Attention if available and leverage it even when bias exists by passing it as an additive attn_mask
        if USE_FLASH:
            attn_mask = None
            if self.bias is not None and not torch.all(self.bias == 0):
                # Prepare additive bias mask scaled appropriately and broadcast to (B, heads, N, N)
                # self.bias shape: (heads, 65, 65) → slice to current sequence length N
                bias_subset = (self.bias[:, :N, :N] * self.bias_scale).to(dtype=q.dtype, device=q.device)  # (heads, N, N)
                # Expand across the batch dimension for broadcasting
                attn_mask = bias_subset.unsqueeze(0).expand(B, -1, -1, -1)  # (B, heads, N, N)
            # Flash-/memory-efficient attention path
            x = scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                scale=self.scale
            )
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            # Fallback implementation (manual attention computation) for older PyTorch / unsupported devices
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.bias is not None:
                head_biases = self.bias[:, :N, :N].unsqueeze(0)
                attn = attn + self.bias_scale * head_biases
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        return self.proj(x)

class Block(nn.Module):
    """Transformer encoder block (LN → MHSA → LN → MLP)."""
    def __init__(self, dim: int, heads: int, drop_path: float, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, heads, bias=bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = Mlp(dim)
        self.dp    = nn.Identity() if drop_path == 0 else DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp(self.attn(self.norm1(x)))
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x

class DropPath(nn.Module):
    """Per‑sample stochastic depth."""
    def __init__(self, p: float = 0.):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.:  # shortcut when inactive
            return x
        keep = 1 - self.p
        mask = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) < keep
        return x.div(keep) * mask

# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------
class ViTChess(nn.Module):
    """Condensed Vision‑Transformer for chess (single‑file version).

    Highlights ▸
      • Accepts a *distance matrix* at construction time → stored as parameter
        `self.bias` (optionally frozen).
      • Supports an `is960` flag per *batch* which is encoded into the CLS token
        *at run‑time* by adding a tiny (+/-) learnable bias (`self.chess960_bias`).
      • Hierarchical structure with early prediction heads and a specialized policy head.
      • Optional periodic mean pooling of patch tokens into CLS token.
      • Optional dropout on the CLS token.
    """
    def __init__(
        self,
        dim: int = 256,
        depth: int = 7, # Total depth
        early_depth: int = 3, # Number of blocks after which early heads are applied
        heads: int = 8,
        drop_path: float = 0.1,
        distance_matrix: Optional[torch.Tensor] = None,
        freeze_distance: bool = True,
        num_policy_planes: int = 73,
        num_value_outputs: int = 3,
        num_material_categories: int = 20,
        num_moves_left_outputs: int = 1,
        policy_cls_projection_dim: int = 64,
        policy_mlp_hidden_dim: Optional[int] = None,
        # New parameters for CLS pooling and dropout
        pool_every_k_blocks: Optional[int] = None, # e.g., 3. If None or 0, no pooling.
        cls_pool_alpha_init: float = 1.0,
        cls_pool_alpha_requires_grad: bool = True,
        cls_dropout_rate: float = 0.0,
        # New parameters for the enhanced policy head
        policy_head_conv_dim: int = 128,       # Number of channels for intermediate policy conv layers
        policy_head_mlp_hidden_dim: int = 256, # Hidden dim for the CLS bias MLP in policy head
        # New parameter for value head MLP
        value_head_mlp_hidden_dim: Optional[int] = None,
        # Added for consistency, though not strictly needed for ViTChess itself if defaults match
        dim_head: Optional[int] = None, # Will be dim // heads if None
    ):
        super().__init__()
        if early_depth >= depth: # early_depth is count of blocks, so index is early_depth-1
            raise ValueError("early_depth must be less than total depth")
        if pool_every_k_blocks is not None and pool_every_k_blocks <= 0:
            pool_every_k_blocks = None # Treat 0 or negative as None

        self.dim = dim
        self.depth = depth
        self.early_depth_count = early_depth # Number of blocks, so index is early_depth - 1
        self.policy_cls_projection_dim = policy_cls_projection_dim
        self.pool_every_k_blocks = pool_every_k_blocks
        self.cls_dropout_rate = cls_dropout_rate

        # --- Patch + position embedding ------------------------------------
        self.patch_embed = nn.Conv2d(NUM_BITBOARD_PLANES, dim, kernel_size=1)   # per-square 1x1 conv
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, 65, dim)) # For CLS + 64 square tokens
        # Global flags for variant, side-to-move, repetition
        self.chess960_bias = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # flag bias for Chess960
        self.stm_bias      = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # flag bias for side-to-move
        self.rep1_bias     = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # flag bias for repetition >=1
        self.rep2_bias     = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # flag bias for repetition >=2
        self.lichess_source_bias = nn.Parameter(torch.randn(1, 1, dim) * 0.02) # Learnable flag for Lichess source

        # --- Transformer encoder (Unified blocks) -------------------
        self.blocks = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path, steps=depth)
        for i in range(depth):
            blk = Block(dim, heads, dp_rates[i].item(), bias=distance_matrix)
            
            if distance_matrix is not None and blk.attn.bias is not None:
                blk.attn.bias.requires_grad = not freeze_distance
            self.blocks.append(blk)

        self.norm_early = nn.LayerNorm(dim) # Norm after early_depth blocks for early heads
        self.norm_final = nn.LayerNorm(dim) # Norm after all blocks for final heads

        # --- CLS Dropout and Pooling Alphas ---
        self.cls_dropout = nn.Dropout(cls_dropout_rate)
        self.alphas = nn.ParameterList()
        if self.pool_every_k_blocks is not None:
            num_potential_pools = (self.depth - 1) // self.pool_every_k_blocks
            for _ in range(num_potential_pools):
                self.alphas.append(
                    nn.Parameter(torch.tensor(cls_pool_alpha_init), requires_grad=cls_pool_alpha_requires_grad)
                )

        # --- Prediction Heads ---
        # Helper to create value heads (single Linear or MLP)
        def create_value_head(input_dim: int, hidden_dim: Optional[int], output_dim: int) -> nn.Module:
            if hidden_dim and hidden_dim > 0:
                return nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            else:
                return nn.Linear(input_dim, output_dim)

        # Early Prediction Heads (from CLS token after early_blocks)
        self.early_value_head = create_value_head(dim, value_head_mlp_hidden_dim, num_value_outputs)
        self.early_material_head = nn.Linear(dim, num_material_categories) # Material head remains a single Linear layer

        # Final Prediction Heads
        self.final_value_head = create_value_head(dim, value_head_mlp_hidden_dim, num_value_outputs)
        self.final_moves_left_head = nn.Linear(dim, num_moves_left_outputs)

        # --- New Policy Head Architecture ---
        self.policy_conv1 = nn.Conv2d(dim, policy_head_conv_dim, kernel_size=3, padding=1)
        self.policy_gelu1 = lambda x: F.gelu(x, approximate='tanh')
        self.policy_conv2 = nn.Conv2d(policy_head_conv_dim, policy_head_conv_dim, kernel_size=3, padding=1)
        self.policy_gelu2 = lambda x: F.gelu(x, approximate='tanh')

        # MLP for CLS token bias
        # Input dim: policy_head_conv_dim (from patch features) + dim (from CLS token)
        self.cls_bias_mlp = nn.Sequential(
            nn.Linear(policy_head_conv_dim + dim, policy_head_mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(policy_head_mlp_hidden_dim, policy_head_conv_dim) # Output matches conv_dim for bias
        )

        self.policy_conv3 = nn.Conv2d(policy_head_conv_dim, num_policy_planes, kernel_size=3, padding=1)

        self._init_weights()

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------
    def freeze_distance_bias(self, freeze: bool = True):
        """Convenience toggle after initial warm‑up for ALL blocks."""
        for blk in self.blocks: # Updated to self.blocks
            if hasattr(blk.attn, 'bias') and blk.attn.bias is not None:
                blk.attn.bias.requires_grad = not freeze

    def get_early_block_cls_features(self, bitboards: torch.Tensor,
                                       is960: torch.Tensor,
                                       stm: torch.Tensor,
                                       rep1: torch.Tensor,
                                       rep2: torch.Tensor,
                                       is_lichess: torch.Tensor):
        """Helper to get CLS token features up to early_depth, for contrastive loss on flipped boards."""
        B = bitboards.size(0)
        x = self.patch_embed(bitboards)      # (B, dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)             # (B, 64, dim)

        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, dim)
        # Apply global flag biases
        cls = cls + is960.view(-1, 1, 1).float() * self.chess960_bias
        cls = cls + stm.view(-1, 1, 1).float() * 2.0 * self.stm_bias - self.stm_bias
        cls = cls + rep1.view(-1, 1, 1).float() * self.rep1_bias
        cls = cls + rep2.view(-1, 1, 1).float() * self.rep2_bias
        cls = cls + is_lichess.view(-1, 1, 1).float() * self.lichess_source_bias

        x = torch.cat([cls, x], dim=1)               # (B, 65, dim)
        x = x + self.pos_embed                       # Add pos-emb after concat

        for i in range(self.early_depth_count):
            cls_input, patches_input = x[:, :1], x[:, 1:]
            # Apply CLS dropout before block input
            cls_dropped = self.cls_dropout(cls_input)
            x_for_block = torch.cat([cls_dropped, patches_input], dim=1)
            x = self.blocks[i](x_for_block)
            # No pooling simulation needed here, just reach the state for early_cls

        x_norm_early = self.norm_early(x)
        cls_token_early_norm = x_norm_early[:, 0]
        # Apply dropout again, consistent with how early_cls is derived in forward pass for heads
        return self.cls_dropout(cls_token_early_norm)

    def forward_batch_contrastive(self, bitboards_list: List[torch.Tensor], 
                              is960_list: List[torch.Tensor],
                              stm_list: List[torch.Tensor],
                              rep1: torch.Tensor,
                              rep2: torch.Tensor,
                              is_lichess: torch.Tensor):
        """
        Efficient batched forward pass for multiple board variants.
        
        Args:
            bitboards_list: List of [original, v_flip, h_flip, hv_flip] tensors
            is960_list: List of is960 flags for each variant
            stm_list: List of side-to-move flags for each variant
            rep1, rep2, is_lichess: Same for all variants
        
        Returns:
            List of CLS features for each variant
        """
        # Stack all variants into a single batch
        all_bitboards = torch.cat(bitboards_list, dim=0)
        B_per_variant = bitboards_list[0].size(0)
        num_variants = len(bitboards_list)
        total_B = B_per_variant * num_variants
        
        # Patch embed all at once
        x = self.patch_embed(all_bitboards)  # (total_B, dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)     # (total_B, 64, dim)
        
        # Create CLS tokens for all variants
        cls = self.cls_token.expand(total_B, -1, -1)  # (total_B, 1, dim)
        
        # Apply variant-specific biases
        if is960_list is not None:
            is960_all = torch.cat(is960_list, dim=0)
            cls = cls + is960_all.view(-1, 1, 1).float() * self.chess960_bias
        
        if stm_list is not None:
            stm_all = torch.cat(stm_list, dim=0)
            stm_sign = stm_all.float() * 2.0 - 1.0
            cls = cls + stm_sign.view(-1, 1, 1) * self.stm_bias
        
        # Apply shared biases (same for all variants)
        if rep1 is not None:
            rep1_expanded = rep1.repeat(num_variants)
            cls = cls + rep1_expanded.view(-1, 1, 1).float() * self.rep1_bias
        
        if rep2 is not None:
            rep2_expanded = rep2.repeat(num_variants)
            cls = cls + rep2_expanded.view(-1, 1, 1).float() * self.rep2_bias
        
        if is_lichess is not None:
            is_lichess_expanded = is_lichess.repeat(num_variants)
            cls = cls + is_lichess_expanded.view(-1, 1, 1).float() * self.lichess_source_bias
        
        # Concatenate and add positional embeddings
        x = torch.cat([cls, x], dim=1)  # (total_B, 65, dim)
        x = x + self.pos_embed
        
        # Process through early blocks
        for i in range(self.early_depth_count):
            cls_input, patches_input = x[:, :1], x[:, 1:]
            cls_dropped = self.cls_dropout(cls_input)
            x_for_block = torch.cat([cls_dropped, patches_input], dim=1)
            x = self.blocks[i](x_for_block)
        
        # Extract and normalize CLS features
        x_norm_early = self.norm_early(x)
        cls_features = self.cls_dropout(x_norm_early[:, 0])  # (total_B, dim)
        
        # Split back into separate variants
        cls_features_list = cls_features.chunk(num_variants, dim=0)
        
        return cls_features_list

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, bitboards: torch.Tensor, *,
                is960: torch.Tensor,
                stm: torch.Tensor,
                rep1: torch.Tensor,
                rep2: torch.Tensor,
                is_lichess: torch.Tensor,
                flipped_bitboards_v: Optional[torch.Tensor] = None,
                flipped_bitboards_h: Optional[torch.Tensor] = None,
                flipped_bitboards_hv: Optional[torch.Tensor] = None):
        """Args:
            bitboards: (B, 14, 8, 8) float16/32.
            is960: (B,) float tensor — 1 if starting position is Chess960.
            stm: (B,) float tensor — 1 if side to move is White.
            rep1: (B,) float tensor — 1 if repetition count >= 1.
            rep2: (B,) float tensor — 1 if repetition count >= 2.
            is_lichess: (B,) float tensor - 1 if from Lichess source.
            flipped_bitboards_v: Optional (B, 14, 8, 8) vertically flipped.
            flipped_bitboards_h: Optional (B, 14, 8, 8) horizontally flipped.
            flipped_bitboards_hv: Optional (B, 14, 8, 8) both H & V flipped.
        """
        outputs = {}

        # --- Process flipped boards for contrastive loss (early blocks only) ---
        contrastive_variants = [bitboards]
        # Ensure inputs are float for consistent processing
        f_is960 = is960.float()
        f_stm = stm.float()

        is960_variants = [f_is960]
        stm_variants = [f_stm]

        if flipped_bitboards_v is not None:
            contrastive_variants.append(flipped_bitboards_v)
            is960_variants.append(f_is960)  # is960 same for v_flip
            stm_variants.append(1.0 - f_stm)  # Toggle STM

        if flipped_bitboards_h is not None:
            contrastive_variants.append(flipped_bitboards_h)
            is960_variants.append(torch.ones_like(f_is960))  # Force 960
            stm_variants.append(f_stm)  # stm same for h_flip

        if flipped_bitboards_hv is not None:
            contrastive_variants.append(flipped_bitboards_hv)
            is960_variants.append(torch.ones_like(f_is960))  # Force 960
            stm_variants.append(1.0 - f_stm)  # Toggle STM

        if len(contrastive_variants) > 1:
            cls_features_all = self.forward_batch_contrastive(
                contrastive_variants, is960_variants, stm_variants, 
                rep1, rep2, is_lichess
            )
            outputs["early_cls"] = cls_features_all[0]
            current_idx = 1
            if flipped_bitboards_v is not None:
                outputs["contrastive_cls_v_flip"] = cls_features_all[current_idx]
                current_idx += 1
            if flipped_bitboards_h is not None:
                outputs["contrastive_cls_h_flip"] = cls_features_all[current_idx]
                current_idx += 1
            if flipped_bitboards_hv is not None:
                outputs["contrastive_cls_hv_flip"] = cls_features_all[current_idx]
        else: # Only original board, no flips
            outputs["early_cls"] = self.get_early_block_cls_features(
                bitboards, is960, stm, rep1, rep2, is_lichess
            )

        # --- Main Path: Initial Embedding & Positional Encoding ------------
        B = bitboards.size(0)
        x = self.patch_embed(bitboards)      # (B, dim, 8, 8)
        # Store original H, W for unflattening later if needed for policy MLP input
        _, C_dim, H_patches, W_patches = x.shape
        x = x.flatten(2).transpose(1, 2)             # (B, 64, dim)

        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, dim)
        # Add global flag biases
        cls = cls + is960.view(-1, 1, 1).float() * self.chess960_bias
        cls = cls + stm.view(-1, 1, 1).float() * 2.0 * self.stm_bias - self.stm_bias
        cls = cls + rep1.view(-1, 1, 1).float() * self.rep1_bias
        cls = cls + rep2.view(-1, 1, 1).float() * self.rep2_bias
        cls = cls + is_lichess.view(-1, 1, 1).float() * self.lichess_source_bias

        x = torch.cat([cls, x], dim=1)               # (B, 65, dim)
        x = x + self.pos_embed                       # Add pos-emb

        # --- Unified Transformer Blocks Processing ---
        alpha_idx_counter = 0
        for d in range(self.depth):
            cls_input_to_block, patches_input_to_block = x[:, :1], x[:, 1:]
            # Apply CLS dropout before it enters a block
            cls_dropped_for_block = self.cls_dropout(cls_input_to_block)
            x_for_block = torch.cat([cls_dropped_for_block, patches_input_to_block], dim=1)
            
            x = self.blocks[d](x_for_block) # Output of block d

            # Early heads application
            if (d + 1) == self.early_depth_count:
                x_normed_early = self.norm_early(x) # x is raw output of block[d]
                cls_token_early_norm, _ = x_normed_early[:, :1], x_normed_early[:, 1:]
                # Apply dropout before feeding to early heads
                cls_token_early_processed = self.cls_dropout(cls_token_early_norm)
                
                outputs["early_value"] = self.early_value_head(cls_token_early_processed.squeeze(1))
                outputs["early_material"] = self.early_material_head(cls_token_early_processed.squeeze(1))
                outputs["early_cls"] = cls_token_early_processed.squeeze(1) # For contrastive loss

            # Periodic pooling into CLS token
            if self.pool_every_k_blocks is not None and \
               (d + 1) % self.pool_every_k_blocks == 0 and \
               d < self.depth - 1: # Ensure not after the last block
                
                cls_from_block, patches_from_block = x[:, :1], x[:, 1:] # x is raw output of block[d]
                
                # CLS token that receives the pool might also be subject to dropout implicitly
                # as the cls_from_block is the result of a block that took a dropped CLS as input.
                # Or, we can apply dropout explicitly to cls_from_block before pooling.
                # The current setup has dropout on input to blocks.
                # For consistency with "dropout on CLS token", the CLS part of `x` (output of block) is used.

                mean_pooled_patches = torch.mean(patches_from_block, dim=1, keepdim=True)
                current_alpha = self.alphas[alpha_idx_counter]
                alpha_idx_counter += 1
                
                # The cls_from_block is the direct output.
                # If dropout is applied before block, this cls_from_block is influenced by that.
                # No additional dropout on cls_from_block before pooling here, to avoid double dropout if not intended.
                cls_updated_by_pool = cls_from_block + current_alpha * mean_pooled_patches
                x = torch.cat([cls_updated_by_pool, patches_from_block], dim=1)
        
        # --- Final Layer Norm and Final Heads ---
        x_norm_final = self.norm_final(x) # x is raw output of the last block (or pooled if last block was a pool point)
        
        cls_final_features_norm, patch_final_features_norm = x_norm_final[:, :1], x_norm_final[:, 1:]
        # Apply dropout before final heads
        cls_final_features_processed = self.cls_dropout(cls_final_features_norm).squeeze(1)
        patch_final_features = patch_final_features_norm # Patches are already normed

        outputs["final_value"] = self.final_value_head(cls_final_features_processed)
        outputs["final_moves_left"] = self.final_moves_left_head(cls_final_features_processed)

        # --- New Policy Head Forward Pass ---
        # patch_final_features is (B, 64, dim) - from x_norm_final[:, 1:]
        # cls_final_features_processed is (B, dim) - from self.cls_dropout(x_norm_final[:, 0]).squeeze(1)

        # 1. Reshape patch tokens to (B, dim, H_patches, W_patches)
        # H_patches, W_patches were stored earlier, assume they are 8, 8 for chess
        # We have patch_final_features = patch_final_features_norm from earlier (B, 64, dim)
        policy_input_spatial = patch_final_features.transpose(1, 2).reshape(
            B, self.dim, H_patches, W_patches
        ) # (B, dim, 8, 8)

        # 2. First two convolutions
        x_policy = self.policy_conv1(policy_input_spatial)
        x_policy = self.policy_gelu1(x_policy) # (B, policy_head_conv_dim, 8, 8)
        conv2_out = self.policy_conv2(x_policy)
        conv2_out_activated = self.policy_gelu2(conv2_out) # (B, policy_head_conv_dim, 8, 8)

        # 3. CLS Token Bias Integration
        # Reshape conv2_out_activated for MLP: (B, policy_head_conv_dim, 64) -> (B, 64, policy_head_conv_dim)
        conv2_out_flat = conv2_out_activated.flatten(2).transpose(1, 2) # (B, 64, policy_head_conv_dim)

        # Expand CLS token (cls_final_features_processed is (B, dim))
        cls_expanded_for_bias_mlp = cls_final_features_processed.unsqueeze(1).expand(-1, H_patches * W_patches, -1) # (B, 64, dim)

        # Concatenate for MLP input
        mlp_input = torch.cat([conv2_out_flat, cls_expanded_for_bias_mlp], dim=2) # (B, 64, policy_head_conv_dim + dim)
        
        cls_bias_values = self.cls_bias_mlp(mlp_input) # (B, 64, policy_head_conv_dim)

        # Add bias and reshape back to spatial
        conv2_out_biased_flat = conv2_out_flat + cls_bias_values
        conv2_out_biased_spatial = conv2_out_biased_flat.transpose(1, 2).reshape(
            B, -1, H_patches, W_patches # -1 should resolve to policy_head_conv_dim
        ) # (B, policy_head_conv_dim, 8, 8)

        # 4. Final convolution
        outputs["policy"] = self.policy_conv3(conv2_out_biased_spatial) # (B, num_policy_planes, 8, 8)
        
        return outputs

    # ---------------------------------------------------------------------
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        # Initialize linear heads and conv layers
        # Heads to initialize (material and moves_left are always Linear)
        heads_to_initialize = [
            self.early_material_head, 
            self.final_moves_left_head
        ]

        # Add value heads (could be Linear or Sequential)
        for value_head_module in [self.early_value_head, self.final_value_head]:
            if isinstance(value_head_module, nn.Linear):
                heads_to_initialize.append(value_head_module)
            elif isinstance(value_head_module, nn.Sequential):
                for layer in value_head_module:
                    if isinstance(layer, nn.Linear):
                        heads_to_initialize.append(layer)
            else:
                raise TypeError(f"Unexpected type for value head: {type(value_head_module)}")

        for head_or_layer in heads_to_initialize:
            if isinstance(head_or_layer, nn.Linear): # Ensure we are only initializing Linear layers directly
                nn.init.xavier_uniform_(head_or_layer.weight)
                if head_or_layer.bias is not None:
                    nn.init.zeros_(head_or_layer.bias)
        
        # Initialize new policy head conv layers and MLP
        for layer in [self.policy_conv1, self.policy_conv2, self.policy_conv3]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        for layer in self.cls_bias_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # NOTE: Transformer blocks (Block class) initialize their own internal layers.
        # NOTE: LayerNorms use default PyTorch initialization.

def load_model_from_checkpoint(
    config_path: str,
    checkpoint_prefix: str,
    device: torch.device | None = None,
) -> ViTChess:
    """
    Loads a ViTChess model according to `config_path` and restores weights
    from `<checkpoint_prefix>.safetensors`. Returns the model in eval mode.

    Args:
      config_path: path to your YAML (as used by your training script).
      checkpoint_prefix: path prefix for your checkpoint files
                         (i.e. `/…/ckpt_00010000` → loads `/…/ckpt_00010000.safetensors`).
      device: optional torch.device; if None, picks CUDA if available, else CPU.
    """
    # 1) config → dict
    cfg = load_config(config_path)

    # 2) select device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) load distance matrix
    dist_path = cfg['model']['distance_matrix_path']
    # The distance_matrix loaded here is (65,65). MultiHeadSelfAttention will expand it per head.
    distance_matrix_numpy = np.load(dist_path)
    distance_matrix = torch.from_numpy(distance_matrix_numpy).float() # Keep on CPU for now, model init will move to device

    # 4) build model with exactly the same args as in your training main()
    model = ViTChess(
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        early_depth=cfg['model']['early_depth'],
        heads=cfg['model']['heads'],
        drop_path=cfg['model'].get('drop_path', 0.1),
        distance_matrix=distance_matrix, # Pass the (65,65) matrix
        freeze_distance=True, # Initial state, can be changed later
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
        dim_head=cfg['model'].get('dim_head'),
        value_head_mlp_hidden_dim=cfg['model'].get('value_head_mlp_hidden_dim'),
    )
    model.to(device)

    # 5) restore weights from safetensors
    state_dict = load_file(checkpoint_prefix + ".safetensors", device=device)
    model.load_state_dict(state_dict)

    # 6) switch to eval mode
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Quick sanity check (can be deleted in production)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example instantiation with new parameters
    model = ViTChess(
        dim=256,
        depth=7,
        early_depth=3,
        heads=8,
        num_policy_planes=73,
        num_value_outputs=3,
        num_material_categories=17, # Example, adjust from dataset
        num_moves_left_outputs=1,
        policy_cls_projection_dim=64,
        policy_mlp_hidden_dim=256+64, # Example: input_dim to MLP
        pool_every_k_blocks=3,
        cls_pool_alpha_init=1.0,
        cls_pool_alpha_requires_grad=True,
        cls_dropout_rate=0.0,
        # New policy head params for testing
        policy_head_conv_dim=128,
        policy_head_mlp_hidden_dim=256,
        value_head_mlp_hidden_dim=128, # Example for testing new value head MLP
    )
    
    B = 2 # Batch size
    dummy_bitboards = torch.randn(B, NUM_BITBOARD_PLANES, 8, 8)
    dummy_is960 = torch.tensor([0, 1], dtype=torch.float) # Example for a batch of 2
    dummy_stm = torch.tensor([1, 0], dtype=torch.float)
    dummy_rep1 = torch.tensor([0, 1], dtype=torch.float)
    dummy_rep2 = torch.tensor([0, 1], dtype=torch.float)
    dummy_is_lichess = torch.tensor([1, 0], dtype=torch.float) # Example Lichess flags
    
    # Test with only main inputs
    print("--- Test with main inputs only ---")
    out = model(dummy_bitboards, is960=dummy_is960, stm=dummy_stm, rep1=dummy_rep1, rep2=dummy_rep2, is_lichess=dummy_is_lichess)
    for k, v in out.items():
        print(f"{k}: {v.shape}")

    # Test with flipped inputs as well
    print("\n--- Test with flipped inputs ---")
    dummy_v_flip = torch.randn(B, NUM_BITBOARD_PLANES, 8, 8)
    out_with_flips = model(
        dummy_bitboards,
        is960=dummy_is960,
        stm=dummy_stm, rep1=dummy_rep1, rep2=dummy_rep2,
        is_lichess=dummy_is_lichess,
        flipped_bitboards_v=dummy_v_flip,
        # flipped_bitboards_h=dummy_v_flip, # Can use same for testing shape
        # flipped_bitboards_hv=dummy_v_flip # Can use same for testing shape
    )
    for k, v in out_with_flips.items():
        print(f"{k}: {v.shape}")

# ---------------------------------------------------------------------------
# End of file ===============================================================