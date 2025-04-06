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
    
    def selective_branch(self, queries, comp_keys, comp_values, gate):
        """
        Implements the selection branch: for each query, compute attention scores over the 
        compressed representations, select the top-k blocks, and aggregate the corresponding values.
        
        Args:
            queries: [B, T, D]
            comp_keys, comp_values: [B, n_blocks, D]
            gate: [B, T] or [B, T, 1] – weighting factor for this branch.
        
        Returns:
            Output of the selection branch with shape [B, T, D]
        """
        B, T, D = queries.shape
        scale = D ** -0.5
        # Compute raw attention scores between queries and compressed keys
        scores = torch.bmm(queries, comp_keys.transpose(1, 2)) * scale  # [B, T, n_blocks]

        # Create causal mask: each query can only attend to blocks corresponding to tokens before it.
        block_ids = torch.div(torch.arange(T, device=queries.device), self.block_size, rounding_mode='floor')
        block_ids = block_ids.unsqueeze(0).expand(B, T)  # [B, T]
        mask = block_ids.unsqueeze(2) < torch.arange(comp_keys.size(1), device=queries.device).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))
        probs = F.softmax(scores, dim=-1)  # [B, T, n_blocks]

        # Select top-k blocks for each query
        topk = min(self.topk_blocks, comp_keys.size(1))
        topk_vals, topk_inds = torch.topk(probs, topk, dim=-1)  # [B, T, topk]

        # Gather corresponding value representations for the selected blocks
        gathered_vals = []
        for b in range(B):
            gathered = comp_values[b][topk_inds[b]]  # [T, topk, D]
            gathered_vals.append(gathered)
        gathered_vals = torch.stack(gathered_vals, dim=0)  # [B, T, topk, D]

        # Aggregate the values using the attention weights of the selected blocks
        sel_out = (topk_vals.unsqueeze(-1) * gathered_vals).sum(dim=2)  # [B, T, D]
        if gate.dim() == 2:
            gate = gate.unsqueeze(-1)
        return sel_out * gate
    
    def sliding_branch(self, queries, keys, values, gate):
        """
        Implements the sliding window branch: for each query, compute local attention over the 
        last 'window_size' tokens of the original sequence.
        
        Args:
            queries, keys, values: [B, T, D]
            gate: [B, T] or [B, T, 1]
        
        Returns:
            Output of the sliding window branch with shape [B, T, D]
        """
        B, T, D = queries.shape
        scale = D ** -0.5
        local_out = torch.zeros_like(queries)

        for t in range(T):
            start = max(0, t - self.window_size + 1)
            q_t = queries[:, t:t+1, :]         # [B, 1, D]
            k_slice = keys[:, start:t+1, :]      # [B, L, D]
            v_slice = values[:, start:t+1, :]    # [B, L, D]
            attn = torch.bmm(q_t, k_slice.transpose(1, 2)) * scale  # [B, 1, L]
            attn = F.softmax(attn, dim=-1)
            context = torch.bmm(attn, v_slice)   # [B, 1, D]
            local_out[:, t, :] = context.squeeze(1)
        if gate.dim() == 2:
            gate = gate.unsqueeze(-1)
        return local_out * gate

    def forward(self, queries, keys, values, gate_cmp, gate_slc, gate_swa):

        # Compression branch
        comp_K = self.compress_tokens(keys)   # [B, n_blocks, D]
        comp_V = self.compress_tokens(values)   # [B, n_blocks, D]

        scale = queries.size(-1) ** -0.5
        raw_scores = torch.bmm(queries, comp_K.transpose(1, 2)) * scale  # [B, T, n_blocks]
        comp_attn = F.softmax(raw_scores, dim=-1)
        comp_out = torch.bmm(comp_attn, comp_V)  # [B, T, D]
        if gate_cmp.dim() == 2:
            gate_cmp = gate_cmp.unsqueeze(-1)
        comp_out = comp_out * gate_cmp

        # Selection branch
        sel_out = self.selective_branch(queries, comp_K, comp_V, gate_slc)

        # Sliding window branch
        if self.window_size > 0:
            slide_out = self.sliding_branch(queries, keys, values, gate_swa)
        else:
            slide_out = torch.zeros_like(queries)
            
        # Sum the outputs of all three branches
        return comp_out + sel_out + slide_out