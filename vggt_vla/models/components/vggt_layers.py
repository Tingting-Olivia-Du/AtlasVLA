"""
VGGT 核心层
"""
import torch
import torch.nn as nn
from typing import Optional

class VGGTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        num_heads = config.num_heads
        mlp_ratio = config.mlp_ratio
        
        # Graph Convolution
        self.graph_conv = GraphConvolution(
            in_features=dim,
            out_features=dim,
            bias=True
        )
        self.norm_graph = nn.LayerNorm(dim)
        
        # Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True
        )
        self.norm_attn = nn.LayerNorm(dim)
        
        # FFN
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1)
        )
        self.norm_mlp = nn.LayerNorm(dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        B, N, D = x.shape
        
        # Graph Convolution
        if edge_index is not None:
            x_flat = x.view(B * N, D)
            x_graph = self.graph_conv(x_flat, edge_index)
            x_graph = x_graph.view(B, N, D)
            x = x + x_graph
            x = self.norm_graph(x)
        
        # Self-Attention
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask_2d = attn_mask.unsqueeze(1).expand(B, N, N)
            attn_mask_2d = attn_mask_2d.to(dtype=x.dtype)
            attn_mask_2d = (1.0 - attn_mask_2d) * -10000.0
        elif attn_mask is not None and attn_mask.dim() == 3:
            attn_mask_2d = attn_mask.to(dtype=x.dtype)
            attn_mask_2d = (1.0 - attn_mask_2d) * -10000.0
        else:
            attn_mask_2d = None
        
        attn_output, _ = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask_2d,
            need_weights=False
        )
        
        x = x + attn_output
        x = self.norm_attn(x)
        
        # FFN
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm_mlp(x)
        
        return x


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index):
        # Linear transformation
        x_transformed = torch.matmul(x, self.weight)
        
        # Message passing
        num_nodes = x.size(0)
        
        # Compute degree
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=x.dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalize
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Aggregate
        out = torch.zeros(num_nodes, self.out_features, dtype=x.dtype, device=x.device)
        edge_weight = norm.unsqueeze(-1) * x_transformed[row]
        out.scatter_add_(0, col.unsqueeze(-1).expand(-1, self.out_features), edge_weight)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
