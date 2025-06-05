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
class ValueCrossAttention(nn.Module):
    """Lightweight cross-attention where CLS queries patches for value-relevant info."""
    def __init__(self, dim, num_heads=4, summary_dim=None):
        super().__init__()
        summary_dim = summary_dim or dim // 2
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.summary_dim = summary_dim
        
        # CLS -> Query, Patches -> Key/Value
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, summary_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, cls_token, patch_tokens):
        B = cls_token.shape[0]
        
        # Project
        q = self.q_proj(cls_token).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(patch_tokens).view(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Aggregate
        out = (attn @ v).transpose(1, 2).reshape(B, 1, -1)
        out = self.out_proj(out).squeeze(1)
        
        return out

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

class MultiHeadSelfAttention(nn.Module):
    """MHSA with QK normalization and optional distance bias.

    Args:
        bias (torch.Tensor | None): 65×65 distance matrix. Registered as a
            *parameter* so it can be frozen / unfrozen via .requires_grad_.
            For per-head bias, this initial bias will be repeated for each head.
        bias_scale (float): Scalar scale that is *always* trainable.
    """
    def __init__(self, dim: int, heads: int, bias: Optional[torch.Tensor] = None):
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

        # Bias handling
        if bias is None:
            initial_bias_data = torch.zeros(heads, 65, 65)
        else:
            initial_bias_data = bias.unsqueeze(0).repeat(heads, 1, 1)
        
        self.bias = nn.Parameter(initial_bias_data)
        self.bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Scale queries
        q = q * self.scale
        
        # Compute attention
        attn = q @ k.transpose(-2, -1)
        
        # Add bias if present
        if self.bias is not None:
            bias_subset = (self.bias[:, :N, :N] * self.bias_scale).to(dtype=attn.dtype, device=attn.device)
            attn = attn + bias_subset.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        return self.proj(x)

class Block(nn.Module):
    """Pre-normalized Transformer encoder block (LN → MHSA → LN → SwiGLU)."""
    def __init__(self, dim: int, heads: int, drop_path: float, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, bias=bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim)
        self.dp = nn.Identity() if drop_path == 0 else DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization: norm before attention and MLP
        x = x + self.dp(self.attn(self.norm1(x)))
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x

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
    """Simplified Vision‑Transformer for chess without contrastive learning.

    Changes from original:
    • Removed contrastive loss and flipped bitboard processing
    • Removed multi-source flagging (no is_lichess)
    • Added QK normalization in attention
    • Switched to SwiGLU activation
    • Implemented pre-normalization
    • Simplified forward pass significantly
    """
    def __init__(
        self,
        dim: int = 256,
        depth: int = 7,
        early_depth: int = 3,
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
        pool_every_k_blocks: Optional[int] = None,
        cls_pool_alpha_init: float = 1.0,
        cls_pool_alpha_requires_grad: bool = True,
        cls_dropout_rate: float = 0.0,
        policy_head_conv_dim: int = 128,
        policy_head_mlp_hidden_dim: int = 256,
        value_head_mlp_hidden_dim: Optional[int] = None,
        value_head_dropout_rate: float = 0.0,
        dim_head: Optional[int] = None,
        value_cross_attn_summary_dim: Optional[int] = None,
        moves_left_cross_attn_summary_dim: Optional[int] = None,
        adaptive_pool_temperature: Optional[float] = None,
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
        self.value_cross_attn_summary_dim = value_cross_attn_summary_dim
        self.moves_left_cross_attn_summary_dim = moves_left_cross_attn_summary_dim
        self.adaptive_pool_temperature = adaptive_pool_temperature

        # Patch + position embedding
        self.patch_embed = nn.Conv2d(NUM_BITBOARD_PLANES, dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, dim))
        
        # Simplified global flags (removed is_lichess)
        self.chess960_bias = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.stm_bias = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.rep1_bias = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.rep2_bias = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Transformer encoder
        self.blocks = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path, steps=depth)
        for i in range(depth):
            blk = Block(dim, heads, dp_rates[i].item(), bias=distance_matrix)
            if distance_matrix is not None and blk.attn.bias is not None:
                blk.attn.bias.requires_grad = not freeze_distance
            self.blocks.append(blk)

        # Final norm for pre-normalization architecture
        self.norm_final = nn.LayerNorm(dim)

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
        # Cross-attention layers
        self.value_cross_attn = None
        if self.value_cross_attn_summary_dim and self.value_cross_attn_summary_dim > 0:
            self.value_cross_attn = ValueCrossAttention(dim, num_heads=heads, summary_dim=self.value_cross_attn_summary_dim)

        self.moves_left_cross_attn = None
        if self.moves_left_cross_attn_summary_dim and self.moves_left_cross_attn_summary_dim > 0:
            self.moves_left_cross_attn = ValueCrossAttention(dim, num_heads=heads, summary_dim=self.moves_left_cross_attn_summary_dim)

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
        self.early_value_head = create_value_head(dim, value_head_mlp_hidden_dim, num_value_outputs, value_head_dropout_rate)
        self.early_material_head = nn.Linear(dim, num_material_categories)
        
        final_value_input_dim = dim
        if self.value_cross_attn:
            final_value_input_dim += self.value_cross_attn_summary_dim
        self.final_value_head = create_value_head(final_value_input_dim, value_head_mlp_hidden_dim, num_value_outputs, value_head_dropout_rate)

        final_moves_left_input_dim = dim
        if self.moves_left_cross_attn:
            final_moves_left_input_dim += self.moves_left_cross_attn_summary_dim
        self.final_moves_left_head = nn.Linear(final_moves_left_input_dim, num_moves_left_outputs)

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

    def freeze_distance_bias(self, freeze: bool = True):
        """Convenience toggle for distance bias in all blocks."""
        for blk in self.blocks:
            if hasattr(blk.attn, 'bias') and blk.attn.bias is not None:
                blk.attn.bias.requires_grad = not freeze

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

        # Patch embedding
        x_patches_embedded = self.patch_embed(bitboards)  # (B, dim, H, W)
        _, self.dim, H_patches, W_patches = x_patches_embedded.shape
        x_patches_flattened = x_patches_embedded.flatten(2).transpose(1, 2)  # (B, H*W, dim)

        # CLS token with biases
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        cls = cls + f_is960.view(-1, 1, 1) * self.chess960_bias
        stm_sign = f_stm * 2.0 - 1.0  # Convert 0,1 to -1,1
        cls = cls + stm_sign.view(-1, 1, 1) * self.stm_bias
        cls = cls + f_rep1.view(-1, 1, 1) * self.rep1_bias
        cls = cls + f_rep2.view(-1, 1, 1) * self.rep2_bias

        # Concatenate and add positional embeddings
        x = torch.cat([cls, x_patches_flattened], dim=1)  # (B, 1 + H*W, dim)
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
                # Apply final norm for early outputs (since we're using pre-norm)
                x_early = self.norm_final(x)
                cls_token_early = self.cls_dropout(x_early[:, 0])
                
                outputs["early_value"] = self.early_value_head(cls_token_early)
                outputs["early_material"] = self.early_material_head(cls_token_early)
                outputs["early_cls"] = cls_token_early

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

        # Final layer norm and final heads
        x_norm_final = self.norm_final(x)
        cls_final_features = self.cls_dropout(x_norm_final[:, 0])
        patch_final_features = x_norm_final[:, 1:]

        outputs["final_cls_features"] = cls_final_features

        # Value Head
        value_head_input = cls_final_features
        if self.value_cross_attn is not None:
            spatial_summary_value = self.value_cross_attn(cls_final_features.unsqueeze(1), patch_final_features)
            value_head_input = torch.cat([cls_final_features, spatial_summary_value], dim=-1)
        outputs["final_value"] = self.final_value_head(value_head_input)

        # Moves Left Head
        moves_left_head_input = cls_final_features
        if self.moves_left_cross_attn is not None:
            spatial_summary_moves = self.moves_left_cross_attn(cls_final_features.unsqueeze(1), patch_final_features)
            moves_left_head_input = torch.cat([cls_final_features, spatial_summary_moves], dim=-1)
        outputs["final_moves_left"] = self.final_moves_left_head(moves_left_head_input)

        # Policy Head
        policy_input_spatial = patch_final_features.transpose(1, 2).reshape(
            B, self.dim, H_patches, W_patches
        )

        x_policy = F.silu(self.policy_conv1(policy_input_spatial))
        conv2_out = F.silu(self.policy_conv2(x_policy))

        conv2_out_flat = conv2_out.flatten(2).transpose(1, 2)
        cls_expanded_for_bias_mlp = cls_final_features.unsqueeze(1).expand(-1, H_patches * W_patches, -1)
        mlp_input = torch.cat([conv2_out_flat, cls_expanded_for_bias_mlp], dim=2)
        cls_bias_values = self.cls_bias_mlp(mlp_input)
        conv2_out_biased_flat = conv2_out_flat + cls_bias_values
        conv2_out_biased_spatial = conv2_out_biased_flat.transpose(1, 2).reshape(
            B, -1, H_patches, W_patches
        )
        outputs["policy"] = self.policy_conv3(conv2_out_biased_spatial)

        return outputs

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        # Initialize heads
        heads_to_initialize = [self.early_material_head, self.final_moves_left_head]

        for value_head_module in [self.early_value_head, self.final_value_head]:
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
    distance_matrix_numpy = np.load(dist_path)
    distance_matrix = torch.from_numpy(distance_matrix_numpy).float()

    # 4) build model with exactly the same args as in your training main()
    model = ViTChess(
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        early_depth=cfg['model']['early_depth'],
        heads=cfg['model']['heads'],
        drop_path=cfg['model'].get('drop_path', 0.1),
        distance_matrix=distance_matrix,
        freeze_distance=True,
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
        value_head_mlp_hidden_dim=cfg['model'].get('value_head_mlp_hidden_dim'),
        value_head_dropout_rate=cfg['model'].get('value_head_dropout_rate', 0.0),
        dim_head=cfg['model'].get('dim_head'),
        value_cross_attn_summary_dim=cfg['model'].get('value_cross_attn_summary_dim'),
        moves_left_cross_attn_summary_dim=cfg['model'].get('moves_left_cross_attn_summary_dim'),
        adaptive_pool_temperature=cfg['model'].get('adaptive_pool_temperature')
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
        value_head_mlp_hidden_dim=128,
        value_head_dropout_rate=0.1
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