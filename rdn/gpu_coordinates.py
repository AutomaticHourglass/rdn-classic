"""
GPU-based Hierarchical Coordinate Calculation

This module implements hierarchical coordinate calculation using PyTorch
for GPU acceleration. Uses parallel scan for efficient bracket tracking.
"""

import torch
import torch.nn.functional as F


class GPUCoordinateCalculator:
    """
    Calculates hierarchical coordinates on GPU using vectorized operations.

    Uses parallel prefix sum (scan) to track bracket depth and generate
    coordinates efficiently on GPU.
    """

    def __init__(self, n_dim=8, bracket_pairs=None, bos_token=None, device='cuda', vocab_size=50000):
        """
        Args:
            n_dim: Number of coordinate dimensions
            bracket_pairs: List of (open, close) token ID tuples
            bos_token: BOS token ID (gets coordinate [0,0,0,...])
            device: Device to run on ('cuda' or 'cpu')
            vocab_size: Maximum token ID expected (for pre-allocation)
        """
        self.n_dim = n_dim
        self.device = device
        self.bos_token = bos_token

        # Default bracket pairs if not specified
        if bracket_pairs is None:
            bracket_pairs = [
                (40, 41),    # ( )
                (91, 93),    # [ ]
                (123, 125),  # { }
                (60, 62),    # < >
            ]

        # Create lookup tensors for fast bracket detection
        # We'll create a tensor that maps token_id -> bracket_type
        # 0 = not a bracket, 1 = opening, -1 = closing
        # Pre-allocate for entire vocabulary to avoid dynamic expansion
        self.bracket_map = torch.zeros(vocab_size, dtype=torch.int8, device=device)

        for open_tok, close_tok in bracket_pairs:
            if open_tok < vocab_size and close_tok < vocab_size:
                self.bracket_map[open_tok] = 1   # opening bracket
                self.bracket_map[close_tok] = -1  # closing bracket

    def calculate_coordinates_batch(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculate coordinates for a batch of token sequences.

        Args:
            token_ids: [B, T] tensor of token IDs

        Returns:
            coordinates: [B, T, n_dim] tensor of coordinates
        """
        B, T = token_ids.shape

        # Detect brackets: [B, T] with values {-1, 0, 1}
        # Clamp token IDs to valid range to avoid index errors
        clamped_ids = torch.clamp(token_ids, 0, len(self.bracket_map) - 1)
        bracket_deltas = self.bracket_map[clamped_ids]  # [B, T]

        # Cumulative sum to get depth at each position
        # This is the parallel scan operation
        depths = torch.cumsum(bracket_deltas, dim=1)  # [B, T]

        # Clamp depths to valid range [0, n_dim-1]
        depths = torch.clamp(depths, 0, self.n_dim - 1)

        # Initialize coordinates: [B, T, n_dim]
        coordinates = torch.zeros(B, T, self.n_dim, dtype=torch.long, device=self.device)

        # For each depth level, count positions at that level
        # This is where we build the actual coordinate values
        for dim in range(self.n_dim):
            # Mask for tokens at this depth
            at_depth = (depths == dim)  # [B, T]

            # Cumulative count of tokens at this depth
            # This gives us the position within this depth level
            position_at_depth = torch.cumsum(at_depth.long(), dim=1)  # [B, T]

            # Assign to coordinate dimension
            # Subtract 1 because cumsum starts at 1, but we want 0-indexed
            coordinates[:, :, dim] = torch.where(
                at_depth,
                position_at_depth - 1,
                torch.zeros_like(position_at_depth)
            )

        # Handle BOS token specially - should get [0, 0, 0, ...]
        if self.bos_token is not None:
            bos_mask = (token_ids == self.bos_token)  # [B, T]
            if bos_mask.any():
                coordinates[bos_mask] = 0

        return coordinates

    def calculate_coordinates(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculate coordinates for a single sequence or batch.

        Args:
            token_ids: [T] or [B, T] tensor of token IDs

        Returns:
            coordinates: [T, n_dim] or [B, T, n_dim] tensor of coordinates
        """
        if token_ids.dim() == 1:
            # Single sequence - add batch dim
            token_ids = token_ids.unsqueeze(0)
            coords = self.calculate_coordinates_batch(token_ids)
            return coords.squeeze(0)  # [T, n_dim]
        else:
            # Batch
            return self.calculate_coordinates_batch(token_ids)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convenience method for calling calculate_coordinates."""
        return self.calculate_coordinates(token_ids)


def create_coordinate_calculator(n_dim=8, bracket_pairs=None, bos_token=None, device='cuda', vocab_size=50000):
    """
    Factory function to create a GPU coordinate calculator.

    Args:
        n_dim: Number of coordinate dimensions
        bracket_pairs: List of (open, close) token ID tuples for brackets
        bos_token: BOS token ID
        device: Device to run on
        vocab_size: Maximum token ID expected (for pre-allocation)

    Returns:
        GPUCoordinateCalculator instance
    """
    return GPUCoordinateCalculator(
        n_dim=n_dim,
        bracket_pairs=bracket_pairs,
        bos_token=bos_token,
        device=device,
        vocab_size=vocab_size
    )
