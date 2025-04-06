import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NSA(nn.Module):
    def __init__(self, block_size=64, window_size=0, topk_blocks=16):
        """
        Args:
            block_size (int): Block size for aggregating tokens.
            window_size (int): Size of the local window for sliding attention.
            topk_blocks (int): Number of blocks to select for the selective branch.
        """
        super().__init__()
        self.block_size = block_size
        self.window_size = window_size
        self.topk_blocks = topk_blocks

    def compress_tokens(self, x):
        """
        Aggregates the input tensor x (e.g. keys or values) by blocks using average pooling.
        
        Args:
            x: Tensor of shape [B, T, D]
        
        Returns:
            Aggregated block representations of shape [B, n_blocks, D]
        """
        B, T, D = x.shape
        bs = self.block_size
        n_blocks = math.ceil(T / bs)
        pad_len = n_blocks * bs - T
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        # Reshape to [B, n_blocks, bs, D]
        x_blocks = x.view(B, n_blocks, bs, D)
        return x_blocks.mean(dim=2)

    def forward(self, queries, keys, values, gate_cmp):
        
        # Compression branch: aggregate keys and values into blocks
        comp_K = self.compress_tokens(keys)   # [B, n_blocks, D]
        comp_V = self.compress_tokens(values)   # [B, n_blocks, D]

        scale = queries.size(-1) ** -0.5
        raw_scores = torch.bmm(queries, comp_K.transpose(1, 2)) * scale  # [B, T, n_blocks]
        comp_attn = F.softmax(raw_scores, dim=-1)
        comp_out = torch.bmm(comp_attn, comp_V)  # [B, T, D]
        if gate_cmp.dim() == 2:
            gate_cmp = gate_cmp.unsqueeze(-1)
        comp_out = comp_out * gate_cmp
