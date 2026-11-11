"""Paper 2 DCRNN (paper2_STC/STC-code/models/dcrnn.py) with the diffusion step fixed.

Copied verbatim except one line in DiffusionConv.forward (marked "Paper 7 fix"). Paper 2 used
torch.einsum('nn,bnc->bnc', support, d), which reads the diagonal of the transition matrix; the
graph has no self-loops, so every diffusion term was zero and the model never mixed sensors.
The fix is the intended matrix product support @ d. State-dict keys are unchanged.
"""
"""
DCRNN (Diffusion Convolutional Recurrent Neural Network) Teacher.

Reference: Li et al., "Diffusion Convolutional Recurrent Neural Network:
Data-Driven Traffic Forecasting", ICLR 2018.

This is a clean PyTorch reimplementation for use as a Teacher in AT-KD.
Uses the same interface as STChronosTeacher for drop-in compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp


class DiffusionConv(nn.Module):
    """
    Diffusion convolution layer.
    Computes: sum_k (theta_k * T^k * X) for K diffusion steps
    where T is the transition matrix (random walk on graph).
    """
    def __init__(self, in_channels, out_channels, num_nodes, K=2):
        super().__init__()
        self.K = K
        self.num_nodes = num_nodes
        # Learnable weights for each diffusion step (forward + backward + identity)
        self.weights = nn.Parameter(torch.randn(K * 2 + 1, in_channels, out_channels) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x, supports):
        """
        Args:
            x: [B, N, C]
            supports: list of [N, N] transition matrices (forward, backward)
        Returns:
            out: [B, N, out_C]
        """
        B, N, C = x.shape
        
        # Collect diffusion results
        diffusion_results = [x]  # K=0: identity
        
        for support in supports:
            # support: [N, N]
            d = x
            for k in range(self.K):
                # T^k * X — matrix multiply per batch
                d = torch.einsum('nm,bmc->bnc', support, d)  # Paper 7 fix: was 'nn,bnc->bnc' (diagonal only)
                diffusion_results.append(d)
        
        # Stack and apply weights: [num_supports, B, N, C]
        out = torch.zeros(B, N, self.weights.shape[-1], device=x.device)
        for i, d in enumerate(diffusion_results):
            out = out + torch.einsum('bnc,co->bno', d, self.weights[i])
        
        return out + self.bias


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU Cell.
    Replaces standard matrix multiplications in GRU with diffusion convolutions.
    """
    def __init__(self, in_channels, hidden_channels, num_nodes, K=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes
        
        # Gate convolutions (r, u gates): input + hidden -> 2*hidden
        self.conv_gate = DiffusionConv(
            in_channels + hidden_channels, hidden_channels * 2, num_nodes, K
        )
        # Candidate convolution: input + hidden -> hidden  
        self.conv_candidate = DiffusionConv(
            in_channels + hidden_channels, hidden_channels, num_nodes, K
        )
    
    def forward(self, x, h, supports):
        """
        Args:
            x: input [B, N, C]
            h: hidden state [B, N, H]
            supports: list of transition matrices
        Returns:
            h_new: [B, N, H]
        """
        # Concatenate input and hidden
        combined = torch.cat([x, h], dim=-1)  # [B, N, C+H]
        
        # Gates
        gates = torch.sigmoid(self.conv_gate(combined, supports))  # [B, N, 2H]
        r, u = gates.chunk(2, dim=-1)  # reset, update: [B, N, H] each
        
        # Candidate
        combined_r = torch.cat([x, r * h], dim=-1)  # [B, N, C+H]
        c = torch.tanh(self.conv_candidate(combined_r, supports))  # [B, N, H]
        
        # Update
        h_new = u * h + (1 - u) * c
        return h_new


class DCRNNEncoder(nn.Module):
    """DCRNN Encoder: processes input sequence with DCGRU cells."""
    def __init__(self, in_channels, hidden_channels, num_nodes, num_layers=2, K=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        self.cells = nn.ModuleList()
        self.cells.append(DCGRUCell(in_channels, hidden_channels, num_nodes, K))
        for _ in range(1, num_layers):
            self.cells.append(DCGRUCell(hidden_channels, hidden_channels, num_nodes, K))
    
    def forward(self, x_seq, supports):
        """
        Args:
            x_seq: [B, T, N, C]
            supports: list of transition matrices
        Returns:
            hidden_states: list of [B, N, H] for each layer
        """
        B, T, N, C = x_seq.shape
        
        # Initialize hidden states
        h = [torch.zeros(B, N, self.hidden_channels, device=x_seq.device) 
             for _ in range(self.num_layers)]
        
        for t in range(T):
            x_t = x_seq[:, t]  # [B, N, C]
            for layer in range(self.num_layers):
                h[layer] = self.cells[layer](
                    x_t if layer == 0 else h[layer - 1], 
                    h[layer], supports
                )
        
        return h


class DCRNNDecoder(nn.Module):
    """DCRNN Decoder: generates output sequence with DCGRU cells."""
    def __init__(self, out_channels, hidden_channels, num_nodes, num_layers=2, K=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        self.cells = nn.ModuleList()
        self.cells.append(DCGRUCell(out_channels, hidden_channels, num_nodes, K))
        for _ in range(1, num_layers):
            self.cells.append(DCGRUCell(hidden_channels, hidden_channels, num_nodes, K))
        
        self.output_proj = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, h, supports, output_len, teacher_forcing=None, teacher_ratio=0.0):
        """
        Args:
            h: list of encoder hidden states [B, N, H] per layer
            supports: transition matrices
            output_len: number of steps to decode
            teacher_forcing: [B, T, N, C] ground truth for teacher forcing
            teacher_ratio: probability of using teacher forcing
        Returns:
            outputs: [B, T_out, N, C]
        """
        B, N, H = h[0].shape
        
        # First decoder input: zero or last encoder output
        decoder_input = torch.zeros(B, N, 1, device=h[0].device)
        
        outputs = []
        for t in range(output_len):
            for layer in range(self.num_layers):
                h[layer] = self.cells[layer](
                    decoder_input if layer == 0 else h[layer - 1],
                    h[layer], supports
                )
            
            output = self.output_proj(h[-1])  # [B, N, 1]
            outputs.append(output)
            
            # Teacher forcing
            if teacher_forcing is not None and torch.rand(1).item() < teacher_ratio:
                decoder_input = teacher_forcing[:, t:t+1, :, :].squeeze(1)
            else:
                decoder_input = output
        
        return torch.stack(outputs, dim=1)  # [B, T_out, N, C]


class DCRNNTeacher(nn.Module):
    """
    DCRNN Teacher Model for Knowledge Distillation.
    
    Compatible interface with STChronosTeacher:
    - forward(batch_series, edge_index) -> forecast
    - forward_with_attention(batch_series, edge_index) -> forecast, attention
    
    The "attention" from DCRNN is the learned diffusion weights,
    represented as an N×N matrix showing learned spatial dependencies.
    """
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64,
                 out_channels=1, num_layers=2, K=2, prediction_length=12,
                 dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.prediction_length = prediction_length
        self.K = K
        
        self.encoder = DCRNNEncoder(in_channels, hidden_channels, num_nodes, num_layers, K)
        self.decoder = DCRNNDecoder(out_channels, hidden_channels, num_nodes, num_layers, K)
        self.dropout = nn.Dropout(dropout)
        
        # Register supports as buffers (computed from adjacency)
        self.supports = None  # Set during first forward pass
    
    def _compute_supports(self, edge_index, device):
        """Compute forward and backward random walk transition matrices from edge_index."""
        N = self.num_nodes
        
        # Build adjacency matrix from edge_index
        adj = torch.zeros(N, N, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        
        # Forward transition: D_out^{-1} * A
        d_out = adj.sum(dim=1).clamp(min=1)
        t_forward = adj / d_out.unsqueeze(1)
        
        # Backward transition: D_in^{-1} * A^T
        d_in = adj.sum(dim=0).clamp(min=1)
        t_backward = adj.t() / d_in.unsqueeze(1)
        
        return [t_forward, t_backward]
    
    def forward(self, batch_series, edge_index):
        """
        Args:
            batch_series: [B, N, T] (same interface as STChronosTeacher)
            edge_index: [2, E]
        Returns:
            forecast: [B, N, H]
        """
        B, N, T = batch_series.shape
        
        # Compute supports if not cached
        if self.supports is None:
            self.supports = self._compute_supports(edge_index, batch_series.device)
        
        # Reshape: [B, N, T] -> [B, T, N, 1]
        x_seq = batch_series.permute(0, 2, 1).unsqueeze(-1)
        
        # Encode
        h = self.encoder(x_seq, self.supports)
        
        # Decode
        outputs = self.decoder(h, self.supports, self.prediction_length)
        # outputs: [B, T_out, N, 1]
        
        # Reshape to [B, N, T_out]
        forecast = outputs.squeeze(-1).permute(0, 2, 1)
        
        return forecast
    
    def forward_with_attention(self, batch_series, edge_index):
        """
        Same as forward(), but also returns a spatial dependency matrix for
        compatibility with the attention transfer KD interface.

        IMPORTANT CAVEAT: Unlike the GAT-based STChronosTeacher which returns
        learned, input-dependent attention weights, the DCRNN "attention" returned
        here is the STATIC graph topology -- specifically, the average of the
        forward and backward random-walk transition matrices derived from the
        fixed adjacency matrix. These are NOT learned attention weights and do
        NOT change with different inputs.

        This means attention transfer from DCRNN teaches the student to mimic
        the graph structure, not dynamic attention patterns. Results using
        DCRNN as teacher for AT-KD should be interpreted with this caveat.

        Args:
            batch_series: [B, N, T] input time series
            edge_index: [2, E] graph connectivity

        Returns:
            forecast: [B, N, H] predicted future values
            attn_dense: [B, 1, N, N] averaged transition matrix (static, not learned)
        """
        forecast = self.forward(batch_series, edge_index)
        B = batch_series.shape[0]
        N = self.num_nodes
        
        # Extract spatial attention from diffusion supports
        # Use the learned gate weights to compute effective spatial influence
        if self.supports is not None:
            # Combine forward + backward transition as "attention"
            combined = sum(self.supports) / len(self.supports)  # [N, N]
            attn = combined.unsqueeze(0).unsqueeze(0).expand(B, 1, N, N)
        else:
            attn = torch.zeros(B, 1, N, N, device=batch_series.device)
        
        return forecast, attn
