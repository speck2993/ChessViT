from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from safetensors.torch import load_file
import numpy as np
import yaml
from torch.utils.checkpoint import checkpoint

def load_config(path: str) -> dict:
    # Ensure UTF-8 encoding when reading config to avoid locale decoding errors
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

NUM_BITBOARD_PLANES = 14  # 12 piece planes + castling + en-passant

# ---------------------------------------------------------------------------
# Utility layers & helpers
# ---------------------------------------------------------------------------

class HybridPatchEmbed(nn.Module):
    """Combines local ResBlock processing with global per-square projections."""
    
    def __init__(
        self, 
        in_channels: int = 14,
        resblock_hidden: int = 32,
        global_proj_dim: int = 16,  # Per-square global features
        out_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Store dimensions
        self.resblock_hidden = resblock_hidden
        self.global_proj_dim = global_proj_dim
        self.out_dim = out_dim
        
        # ========== Local Processing Path ==========
        # ResBlock for local pattern detection
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, resblock_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(resblock_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(resblock_hidden, resblock_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(resblock_hidden),
        )
        self.resblock_proj = nn.Conv2d(in_channels, resblock_hidden, kernel_size=1)  # Skip connection
        
        # ========== Global Processing Path ==========
        # Per-square projections that see the entire board
        # Using 13 dims: 12 piece types + 1 empty
        self.square_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(13, global_proj_dim),
                nn.LayerNorm(global_proj_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(64)
        ])
        
        # ========== Fusion ==========
        # Combine local (resblock_hidden) + global (global_proj_dim) features
        fusion_dim = resblock_hidden + global_proj_dim
        
        # Initial projection to model dimension
        self.initial_proj = nn.Conv2d(fusion_dim, out_dim, kernel_size=1)
        
        # FFN to process the enriched embedding
        self.post_embed_ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        # Initialize carefully
        self._init_weights()
        
    def _init_weights(self):
        # Initialize global projections to start near zero
        for proj in self.square_projections:
            nn.init.normal_(proj[0].weight, std=0.02)
            nn.init.zeros_(proj[0].bias)
        
        # Standard initialization for other components
        nn.init.xavier_uniform_(self.initial_proj.weight)
        if self.initial_proj.bias is not None:
            nn.init.zeros_(self.initial_proj.bias)
        
    def forward(self, bitboards: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bitboards: (B, 14, 8, 8) - Standard chess bitboard representation
        Returns:
            tokens: (B, 64, out_dim) - Enriched token embeddings
        """
        B = bitboards.shape[0]
        
        # ========== Local Processing ==========
        # ResBlock for local pattern detection
        local_features = self.resblock(bitboards) + self.resblock_proj(bitboards)  # (B, resblock_hidden, 8, 8)
        local_features = F.relu(local_features)
        
        # ========== Global Processing ==========
        # Create one-hot piece representation for global projections
        pieces = bitboards[:, :12]  # First 12 channels are pieces
        empty = 1 - pieces.sum(dim=1, keepdim=True).clamp(0, 1)  # Empty squares
        
        piece_one_hot = torch.cat([pieces, empty], dim=1)  # (B, 13, 8, 8)
        piece_one_hot_flat = piece_one_hot.permute(0, 2, 3, 1).reshape(B, 64, 13)  # (B, 64, 13)
        
        # Apply per-square projections
        global_features_list = []
        for sq_idx in range(64):
            # Each square sees the full board state and projects it
            sq_features = self.square_projections[sq_idx](piece_one_hot_flat[:, sq_idx])  # (B, global_proj_dim)
            global_features_list.append(sq_features)
        
        global_features = torch.stack(global_features_list, dim=1)  # (B, 64, global_proj_dim)
        global_features = global_features.permute(0, 2, 1).reshape(B, -1, 8, 8)  # (B, global_proj_dim, 8, 8)
        
        # ========== Fusion ==========
        # Concatenate local and global features
        combined = torch.cat([local_features, global_features], dim=1)  # (B, fusion_dim, 8, 8)
        
        # Project to model dimension
        tokens = self.initial_proj(combined)  # (B, out_dim, 8, 8)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, 64, out_dim)
        
        # Apply FFN to process enriched embeddings
        tokens = tokens + self.post_embed_ffn(tokens)  # Residual connection
        
        return tokens

class AdaptivePooling(nn.Module):
    """Learn importance weights based on CLS state rather than uniform pooling."""
    def __init__(self, dim, temperature=1.0):
        super().__init__()
        # CLS and patches determine what's important to pool
        self.importance_net = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )
        self.temperature = temperature
        
    def forward(self, cls_token, patch_tokens):
        # cls_token shape: (B, 1, dim), patch_tokens shape: (B, N, dim)
        cls_expanded = cls_token.expand_as(patch_tokens)
        combined = torch.cat([patch_tokens, cls_expanded], dim=-1)
        
        # Compute importance scores
        importance = self.importance_net(combined) / self.temperature
        weights = F.softmax(importance, dim=1)
        
        # Weighted pooling
        pooled = (patch_tokens * weights).sum(dim=1, keepdim=True)
        
        return pooled

class SwiGLU(nn.Module):
    """SwiGLU activation function: x * swish(gate)"""
    def __init__(self, dim: int, hidden: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden = hidden or dim * 4
        # SwiGLU requires 2 linear layers for gate and value
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.value_proj = nn.Linear(dim, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        value = self.value_proj(x)
        x = gate * F.silu(value)  # SwiGLU: gate * SiLU(value)
        x = self.drop(x)
        x = self.out_proj(x)
        x = self.drop(x)
        return x

class SmolGenCLS(nn.Module):
    """SmolGen: Generate dynamic per-head attention biases from CLS token."""
    def __init__(self, dim: int, num_heads: int, latent_dim: int = 256, dropout_rate: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        
        # CLS → per-head latent with extra hidden layer for complex patterns
        self.cls_to_latent = nn.Sequential(
            nn.Linear(dim, latent_dim * 2, bias=False),
            nn.LayerNorm(latent_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim * 2, latent_dim * num_heads, bias=False),
            nn.LayerNorm(latent_dim * num_heads),
            nn.Dropout(dropout_rate)
        )
        
        # Direct mapping to bias (simplified, no low-rank factorization)
        self.to_bias = nn.Linear(latent_dim, 65 * 65, bias=False)
        
        # Initialize to near-zero for stable training
        nn.init.normal_(self.cls_to_latent[0].weight, std=0.02)
        nn.init.normal_(self.cls_to_latent[4].weight, std=0.02)
        nn.init.zeros_(self.to_bias.weight)
        
    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        B = cls_token.shape[0]
        
        # Get per-head latents
        latents = self.cls_to_latent(cls_token)  # (B, latent_dim * num_heads)
        latents = latents.view(B, self.num_heads, self.latent_dim)  # (B, heads, latent_dim)
        
        # Generate biases directly
        biases = self.to_bias(latents)  # (B, heads, 4225)
        
        return biases.view(B, self.num_heads, 65, 65)

class MultiHeadSelfAttention(nn.Module):
    """MHSA with QK normalization and SmolGen dynamic biasing."""
    def __init__(self, dim: int, heads: int, use_smolgen: bool = True, smolgen_latent_dim: int = 256, smolgen_dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # QK Normalization for stability
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        # SmolGen for dynamic bias
        self.use_smolgen = use_smolgen
        if use_smolgen:
            self.smolgen = SmolGenCLS(dim, heads, latent_dim=smolgen_latent_dim, dropout_rate=smolgen_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        cls_token = x[:, 0]  # Extract CLS for bias generation
        
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Scale queries
        q = q * self.scale
        
        # Compute attention
        attn = q @ k.transpose(-2, -1)
        
        # Add dynamic bias from SmolGen
        if self.use_smolgen:
            bias = self.smolgen(cls_token)  # (B, heads, 65, 65)
            attn = attn + bias[:, :, :N, :N]  # Handle if N < 65
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        return self.proj(x)

class Block(nn.Module):
    """Pre-normalized Transformer encoder block (LN → MHSA → LN → SwiGLU)."""
    def __init__(self, dim: int, heads: int, drop_path: float, use_smolgen: bool = True, 
                 smolgen_latent_dim: int = 256, smolgen_dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, use_smolgen=use_smolgen, 
                                         smolgen_latent_dim=smolgen_latent_dim, smolgen_dropout=smolgen_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim)
        self.dp = nn.Identity() if drop_path == 0 else DropPath(drop_path)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        def _inner_forward(x: torch.Tensor):
            # Pre-normalization: norm before attention and MLP
            x = x + self.dp(self.attn(self.norm1(x)))
            x = x + self.dp(self.mlp(self.norm2(x)))
            return x

        if self.gradient_checkpointing and self.training:
            return checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            return _inner_forward(x)

class ValueAttentionLayer(nn.Module):
    """Special final attention layer optimized for value/moves_left prediction."""
    def __init__(self, dim: int, heads: int, smolgen_latent_dim: int = 256, smolgen_dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, use_smolgen=True, 
                                         smolgen_latent_dim=smolgen_latent_dim, smolgen_dropout=smolgen_dropout)
        # No MLP - we want to preserve features for the heads
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple pre-norm attention only
        return x + self.attn(self.norm(x))

class ValueHead(nn.Module):
    """Flexible value head with spatial compression and full-dimension CLS token."""
    def __init__(self, dim: int, num_outputs: int, spatial_compress_dim: int, 
                 mlp_dims: List[int], dropout_rate: float = 0.1):
        super().__init__()
        self.spatial_compress = nn.Linear(dim, spatial_compress_dim)
        
        # MLP: dim (CLS) + 64 * spatial_compress_dim (Spatial) input
        input_dim = dim + (64 * spatial_compress_dim)
        
        layers = []
        current_dim = input_dim
        for hidden_dim in mlp_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, num_outputs))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 65, dim)
        cls_features = x[:, 0]                                    # (B, dim)
        spatial_features = self.spatial_compress(x[:, 1:])        # (B, 64, spatial_compress_dim)
        spatial_flat = spatial_features.view(x.shape[0], -1)      # (B, 64 * spatial_compress_dim)
        
        combined = torch.cat([cls_features, spatial_flat], dim=1) # (B, dim + 64*s_c_d)
        return self.mlp(combined)

class DropPath(nn.Module):
    """Per‑sample stochastic depth."""
    def __init__(self, p: float = 0.):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        keep = 1 - self.p
        mask = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) < keep
        return x.div(keep) * mask

# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------
class ViTChess(nn.Module):
    """Vision‑Transformer for chess with SmolGen dynamic attention biasing.

    New features:
    • SmolGen dynamic attention biasing replacing static distance matrix
    • Asymmetric value/moves_left heads with different compression ratios
    • Special value attention layer for final value/moves_left processing
    • Configurable SmolGen parameters and head architectures
    """
    def __init__(
        self,
        dim: int = 256,
        depth: int = 7,
        early_depth: int = 3,
        heads: int = 8,
        drop_path: float = 0.1,
        num_policy_planes: int = 73,
        num_value_outputs: int = 3,
        num_material_categories: int = 20,
        num_moves_left_outputs: int = 1,
        policy_cls_projection_dim: int = 64,
        policy_mlp_hidden_dim: Optional[int] = None,
        pool_every_k_blocks: Optional[int] = None,
        cls_pool_alpha_init: float = 1.0,
        cls_pool_alpha_requires_grad: bool = True,
        cls_dropout_rate: float = 0.0,
        policy_head_conv_dim: int = 128,
        policy_head_mlp_hidden_dim: int = 256,
        value_head_dropout_rate: float = 0.0,
        dim_head: Optional[int] = None,
        adaptive_pool_temperature: Optional[float] = None,
        # SmolGen parameters
        smolgen_start_layer: int = 2,
        smolgen_latent_dim: int = 256,
        smolgen_dropout: float = 0.1,
        # Value head parameters
        value_spatial_compress_dim: int = 16,
        value_head_mlp_dims: List[int] = [128, 64],
        moves_left_spatial_compress_dim: int = 8,
        moves_left_head_mlp_dims: List[int] = [128, 64],
        # Hybrid patch embedding parameters
        patch_resblock_hidden: int = 32,
        patch_global_proj_dim: int = 16,
        patch_embed_dropout: float = 0.1,
    ):
        super().__init__()
        if early_depth >= depth:
            raise ValueError("early_depth must be less than total depth")
        if pool_every_k_blocks is not None and pool_every_k_blocks <= 0:
            pool_every_k_blocks = None

        self.dim = dim
        self.depth = depth
        self.early_depth_count = early_depth
        self.policy_cls_projection_dim = policy_cls_projection_dim
        self.pool_every_k_blocks = pool_every_k_blocks
        self.cls_dropout_rate = cls_dropout_rate
        self.adaptive_pool_temperature = adaptive_pool_temperature
        self.smolgen_start_layer = smolgen_start_layer

        # Hybrid patch + position embedding
        self.patch_embed = HybridPatchEmbed(
            in_channels=NUM_BITBOARD_PLANES,
            resblock_hidden=patch_resblock_hidden,
            global_proj_dim=patch_global_proj_dim,
            out_dim=dim,
            dropout=patch_embed_dropout
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, dim))
        
        # Simplified global flags (removed is_lichess)
        self.chess960_bias = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.stm_bias = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.rep1_bias = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.rep2_bias = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Transformer encoder with SmolGen
        self.blocks = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path, steps=depth)
        for i in range(depth):
            # Use SmolGen starting from specified layer (give early layers time to build representations)
            use_smolgen = (i >= smolgen_start_layer)
            blk = Block(
                dim=dim,
                heads=heads,
                drop_path=dp_rates[i].item(),
                use_smolgen=use_smolgen,
                smolgen_latent_dim=smolgen_latent_dim,
                smolgen_dropout=smolgen_dropout
            )
            self.blocks.append(blk)

        # Final norm for pre-normalization architecture
        self.norm_final = nn.LayerNorm(dim)

        # Strategic representation head - normalized embeddings for clustering
        self.repr_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256, bias=False),
            nn.LayerNorm(256, elementwise_affine=False)  # L2 normalization effect
        )

        # Add special value attention layers
        self.early_value_attention = ValueAttentionLayer(dim, heads, smolgen_latent_dim, smolgen_dropout)
        self.final_value_attention = ValueAttentionLayer(dim, heads, smolgen_latent_dim, smolgen_dropout)

        # CLS Dropout and Pooling Alphas
        self.cls_dropout = nn.Dropout(cls_dropout_rate)
        self.alphas = nn.ParameterList()
        self.adaptive_pools = nn.ModuleList()
        if self.pool_every_k_blocks is not None:
            num_potential_pools = (self.depth - 1) // self.pool_every_k_blocks
            for _ in range(num_potential_pools):
                self.alphas.append(
                    nn.Parameter(torch.tensor(cls_pool_alpha_init), requires_grad=cls_pool_alpha_requires_grad)
                )
                if self.adaptive_pool_temperature is not None and self.adaptive_pool_temperature > 0:
                    self.adaptive_pools.append(AdaptivePooling(dim, temperature=self.adaptive_pool_temperature))

        # Prediction Heads
        def create_value_head(input_dim: int, hidden_dim: Optional[int], output_dim: int, dropout_rate: float) -> nn.Module:
            if hidden_dim and hidden_dim > 0:
                layers = [
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(),  # Using SiLU instead of GELU for consistency
                ]
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
                layers.append(nn.Linear(hidden_dim, output_dim))
                return nn.Sequential(*layers)
            else:
                return nn.Linear(input_dim, output_dim)

        # Early and Final Prediction Heads
        self.early_value_head = ValueHead(
            dim=dim,
            num_outputs=num_value_outputs,
            spatial_compress_dim=value_spatial_compress_dim, # Use same compression as final for consistency
            mlp_dims=value_head_mlp_dims,
            dropout_rate=value_head_dropout_rate
        )
        self.early_material_head = nn.Linear(dim, num_material_categories)
        
        # New flexible value heads
        self.final_value_head = ValueHead(
            dim=dim, 
            num_outputs=num_value_outputs,
            spatial_compress_dim=value_spatial_compress_dim,
            mlp_dims=value_head_mlp_dims,
            dropout_rate=value_head_dropout_rate
        )
        
        self.final_moves_left_head = ValueHead(
            dim=dim,
            num_outputs=num_moves_left_outputs,
            spatial_compress_dim=moves_left_spatial_compress_dim,
            mlp_dims=moves_left_head_mlp_dims,
            dropout_rate=value_head_dropout_rate
        )

        # Policy Head Architecture
        self.policy_conv1 = nn.Conv2d(dim, policy_head_conv_dim, kernel_size=3, padding=1)
        self.policy_conv2 = nn.Conv2d(policy_head_conv_dim, policy_head_conv_dim, kernel_size=3, padding=1)
        self.cls_bias_mlp = nn.Sequential(
            nn.Linear(policy_head_conv_dim + dim, policy_head_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(policy_head_mlp_hidden_dim, policy_head_conv_dim)
        )
        self.policy_conv3 = nn.Conv2d(policy_head_conv_dim, num_policy_planes, kernel_size=3, padding=1)

        self._init_weights()

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        for block in self.blocks:
            block.gradient_checkpointing = True

    def forward(self, bitboards: torch.Tensor, *,
                is960: torch.Tensor,
                stm: torch.Tensor,
                rep1: torch.Tensor,
                rep2: torch.Tensor,
                # Removed: is_lichess and all flipped variants
                ):

        outputs = {}
        B = bitboards.size(0)
        
        # Ensure inputs for biases are float
        f_is960 = is960.float()
        f_stm = stm.float()
        f_rep1 = rep1.float()
        f_rep2 = rep2.float()

        # Hybrid patch embedding (already returns flattened tokens)
        x_patches_flattened = self.patch_embed(bitboards)  # (B, 64, dim)

        # CLS token with biases
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        cls = cls + f_is960.view(-1, 1, 1) * self.chess960_bias
        stm_sign = f_stm * 2.0 - 1.0  # Convert 0,1 to -1,1
        cls = cls + stm_sign.view(-1, 1, 1) * self.stm_bias
        cls = cls + f_rep1.view(-1, 1, 1) * self.rep1_bias
        cls = cls + f_rep2.view(-1, 1, 1) * self.rep2_bias

        # Concatenate and add positional embeddings
        x = torch.cat([cls, x_patches_flattened], dim=1)  # (B, 1 + 64, dim)
        x = x + self.pos_embed

        # Transformer blocks with periodic pooling
        alpha_idx_counter = 0
        for d in range(self.depth):
            cls_input_to_block, patches_input_to_block = x[:, :1], x[:, 1:]
            cls_dropped_for_block = self.cls_dropout(cls_input_to_block)
            x_for_block = torch.cat([cls_dropped_for_block, patches_input_to_block], dim=1)
            
            x = self.blocks[d](x_for_block)

            # Early heads application
            if (d + 1) == self.early_depth_count:
                # Apply final norm and special attention layer for early outputs
                x_early_norm = self.norm_final(x)
                x_early_attn = self.early_value_attention(x_early_norm)
                
                # Extract early strategic embedding
                early_cls = x_early_norm[:, 0]
                outputs["early_embedding"] = self.repr_head(early_cls)
                
                outputs["early_value"] = self.early_value_head(x_early_attn)
                outputs["early_material"] = self.early_material_head(x_early_norm[:, 0])
                outputs["early_cls"] = self.cls_dropout(x_early_norm[:, 0])

            # Periodic pooling into CLS token
            if self.pool_every_k_blocks is not None and \
               (d + 1) % self.pool_every_k_blocks == 0 and \
               d < self.depth - 1:
                
                cls_from_block, patches_from_block = x[:, :1], x[:, 1:]

                # Use adaptive pooling if available, otherwise mean pool
                if self.adaptive_pools and alpha_idx_counter < len(self.adaptive_pools):
                    pooled_patches = self.adaptive_pools[alpha_idx_counter](cls_from_block, patches_from_block)
                else:
                    pooled_patches = torch.mean(patches_from_block, dim=1, keepdim=True)
                
                current_alpha = self.alphas[alpha_idx_counter]
                alpha_idx_counter += 1
                
                cls_updated_by_pool = cls_from_block + current_alpha * pooled_patches
                x = torch.cat([cls_updated_by_pool, patches_from_block], dim=1)

        # Final layer norm
        x_norm_final = self.norm_final(x)

        # Extract final strategic embedding
        final_cls = x_norm_final[:, 0]
        outputs["embedding"] = self.repr_head(final_cls)

        # Apply special value attention layer (shared for final value and moves_left)
        x_for_value_heads = self.final_value_attention(x_norm_final)

        # Extract features for different heads
        cls_final_features = self.cls_dropout(x_norm_final[:, 0])  # For policy
        
        outputs["final_cls_features"] = cls_final_features

        # Apply new flexible value heads
        outputs["final_value"] = self.final_value_head(x_for_value_heads)
        outputs["final_moves_left"] = self.final_moves_left_head(x_for_value_heads)

        # Policy Head (continues using main network features)
        policy_input_spatial = x_norm_final[:, 1:].transpose(1, 2).reshape(
            B, self.dim, 8, 8
        )

        x_policy = F.silu(self.policy_conv1(policy_input_spatial))
        conv2_out = F.silu(self.policy_conv2(x_policy))

        conv2_out_flat = conv2_out.flatten(2).transpose(1, 2)
        cls_expanded_for_bias_mlp = cls_final_features.unsqueeze(1).expand(-1, 64, -1)
        mlp_input = torch.cat([conv2_out_flat, cls_expanded_for_bias_mlp], dim=2)
        cls_bias_values = self.cls_bias_mlp(mlp_input)
        conv2_out_biased_flat = conv2_out_flat + cls_bias_values
        conv2_out_biased_spatial = conv2_out_biased_flat.transpose(1, 2).reshape(
            B, -1, 8, 8
        )
        outputs["policy"] = self.policy_conv3(conv2_out_biased_spatial)

        return outputs

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.initial_proj.weight)
        if self.patch_embed.initial_proj.bias is not None:
            nn.init.zeros_(self.patch_embed.initial_proj.bias)

        # Initialize heads
        heads_to_initialize = [self.early_material_head]

        for value_head_module in [self.early_value_head]:
            if isinstance(value_head_module, nn.Linear):
                heads_to_initialize.append(value_head_module)
            elif isinstance(value_head_module, nn.Sequential):
                for layer in value_head_module:
                    if isinstance(layer, nn.Linear):
                        heads_to_initialize.append(layer)

        for head_or_layer in heads_to_initialize:
            if isinstance(head_or_layer, nn.Linear):
                nn.init.xavier_uniform_(head_or_layer.weight)
                if head_or_layer.bias is not None:
                    nn.init.zeros_(head_or_layer.bias)

        # Initialize policy head
        for layer in [self.policy_conv1, self.policy_conv2, self.policy_conv3]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        for layer in self.cls_bias_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Initialize repr_head weights
        nn.init.xavier_uniform_(self.repr_head[1].weight)

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

    # 3) build model with exactly the same args as in your training main()
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
        dim_head=cfg['model'].get('dim_head'),
        adaptive_pool_temperature=cfg['model'].get('adaptive_pool_temperature'),
        # SmolGen parameters
        smolgen_start_layer=cfg['model'].get('smolgen_start_layer', 2),
        smolgen_latent_dim=cfg['model'].get('smolgen_latent_dim', 256),
        smolgen_dropout=cfg['model'].get('smolgen_dropout', 0.1),
        # Value head parameters
        value_spatial_compress_dim=cfg['model'].get('value_spatial_compress_dim', 16),
        value_head_mlp_dims=cfg['model'].get('value_head_mlp_dims', [128, 64]),
        moves_left_spatial_compress_dim=cfg['model'].get('moves_left_spatial_compress_dim', 8),
        moves_left_head_mlp_dims=cfg['model'].get('moves_left_head_mlp_dims', [128, 64]),
        # Hybrid patch embedding parameters
        patch_resblock_hidden=cfg['model'].get('patch_resblock_hidden', 32),
        patch_global_proj_dim=cfg['model'].get('patch_global_proj_dim', 16),
        patch_embed_dropout=cfg['model'].get('patch_embed_dropout', 0.1),
    )
    model.to(device)

    # 5) restore weights from safetensors
    state_dict = load_file(checkpoint_prefix + ".safetensors", device=str(device))
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
        num_material_categories=17,
        num_moves_left_outputs=1,
        policy_cls_projection_dim=64,
        policy_mlp_hidden_dim=256+64,
        pool_every_k_blocks=3,
        cls_pool_alpha_init=1.0,
        cls_pool_alpha_requires_grad=True,
        cls_dropout_rate=0.0,
        policy_head_conv_dim=128,
        policy_head_mlp_hidden_dim=256,
        value_head_dropout_rate=0.1,
        # SmolGen parameters
        smolgen_start_layer=2,
        smolgen_latent_dim=256,
        smolgen_dropout=0.1,
        # Value head parameters
        value_spatial_compress_dim=16,
        value_head_mlp_dims=[128, 64],
        moves_left_spatial_compress_dim=8,
        moves_left_head_mlp_dims=[128, 64],
        # Hybrid patch embedding parameters
        patch_resblock_hidden=32,
        patch_global_proj_dim=16,
        patch_embed_dropout=0.1,
    )
    
    B = 2
    dummy_bitboards = torch.randn(B, NUM_BITBOARD_PLANES, 8, 8)
    dummy_is960 = torch.tensor([0, 1], dtype=torch.float)
    dummy_stm = torch.tensor([1, 0], dtype=torch.float)
    dummy_rep1 = torch.tensor([0, 1], dtype=torch.float)
    dummy_rep2 = torch.tensor([0, 1], dtype=torch.float)
    
    # Test with only main inputs
    print("--- Test with main inputs only ---")
    out = model(dummy_bitboards, is960=dummy_is960, stm=dummy_stm, rep1=dummy_rep1, rep2=dummy_rep2)
    for k, v in out.items():
        print(f"{k}: {v.shape}")

# ---------------------------------------------------------------------------
# End of file ===============================================================