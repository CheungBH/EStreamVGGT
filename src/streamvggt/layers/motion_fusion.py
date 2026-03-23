import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionCompensatedFusion(nn.Module):
    """
    Flow-guided Token Warping + Dense Cross-Attention Module.
    Takes previous frame tokens, warps them using optical flow to align with 
    the current frame, and performs dense cross-attention to fuse the features.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Dense Cross-Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm_context = nn.LayerNorm(embed_dim)
        
        # We use PyTorch's native SDPA for efficiency where possible
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # MLP for post-attention processing
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
        # Learnable gating for residual connection (starts small to not disrupt pre-trained features initially)
        self.gate = nn.Parameter(torch.zeros(1))

    def _warp_tokens(self, tokens_prev: torch.Tensor, flow: torch.Tensor, H: int, W: int, patch_size: int):
        """
        Warp previous frame tokens to current frame coordinates using flow.
        
        Args:
            tokens_prev: (B, P, C) where P = (H/patch_size) * (W/patch_size)
            flow: (B, 2, H, W) full resolution optical flow (dx, dy)
            H, W: original image dimensions
            patch_size: size of the patch
        Returns:
            warped_tokens: (B, P, C)
        """
        B, P, C = tokens_prev.shape
        H_p, W_p = H // patch_size, W // patch_size
        assert P == H_p * W_p, f"Token sequence length {P} does not match patch grid {H_p}x{W_p}"
        
        # 1. Reshape tokens to 2D grid: (B, C, H_p, W_p)
        tokens_grid = tokens_prev.transpose(1, 2).view(B, C, H_p, W_p)
        
        # 2. Downsample flow to patch resolution
        # Note: flow is usually in pixel units. When downsampling, we also need to scale the magnitude
        # by dividing by patch_size so it matches the patch grid coordinates.
        flow_patch = F.interpolate(flow, size=(H_p, W_p), mode='bilinear', align_corners=False)
        flow_patch = flow_patch / patch_size
        
        # 3. Create base grid [-1, 1]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H_p, device=tokens_prev.device),
            torch.linspace(-1, 1, W_p, device=tokens_prev.device),
            indexing='ij'
        )
        base_grid = torch.stack([x, y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1) # (B, 2, H_p, W_p)
        
        # 4. Normalize flow to [-1, 1] range for grid_sample
        # grid_sample expects coordinates in [-1, 1], where 2.0 spans the whole image.
        # flow_patch is in patch units [0, W_p] and [0, H_p]
        norm_flow_x = 2.0 * flow_patch[:, 0, :, :] / max(W_p - 1, 1)
        norm_flow_y = 2.0 * flow_patch[:, 1, :, :] / max(H_p - 1, 1)
        norm_flow = torch.stack([norm_flow_x, norm_flow_y], dim=1)
        
        # Current coord = Previous coord + Flow => Previous coord = Current coord - Flow
        # Since we are backward warping (gathering from prev frame to current frame):
        warp_grid = base_grid - norm_flow 
        warp_grid = warp_grid.permute(0, 2, 3, 1) # (B, H_p, W_p, 2)
        
        # 5. Warp tokens
        warped_grid = F.grid_sample(tokens_grid, warp_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        # 6. Reshape back to sequence
        warped_tokens = warped_grid.flatten(2).transpose(1, 2) # (B, P, C)
        return warped_tokens

    def forward(self, x_curr: torch.Tensor, x_prev: torch.Tensor, flow: torch.Tensor, H: int, W: int, patch_size: int):
        """
        Args:
            x_curr: (B, P, C) Current frame patch tokens
            x_prev: (B, P, C) Previous frame patch tokens
            flow: (B, 2, H, W) Optical flow from prev to curr
            H, W: Image dimensions
            patch_size: Patch size
        """
        # 1. Warp previous tokens to align with current frame
        warped_prev = self._warp_tokens(x_prev, flow, H, W, patch_size)
        
        # 2. Normalize
        q = self.norm1(x_curr)
        k = v = self.norm_context(warped_prev)
        
        # 3. Cross Attention (Query = Current, Key/Value = Warped Previous)
        attn_out, _ = self.cross_attn(q, k, v)
        
        # 4. Residual and MLP
        fused = x_curr + self.gate * attn_out
        fused = fused + self.gate * self.mlp(self.norm2(fused))
        
        return fused
