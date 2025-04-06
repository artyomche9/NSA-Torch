import torch
import torch.nn as nn

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

    def forward(self, queries, keys, values):
        pass
