"""
Graph 构建模块
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

class GraphBuilder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graph_type = config.graph_type
        self.k_neighbors = config.k_neighbors
        
    def build_graph(
        self,
        fusion_info: Dict,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        N_total = fusion_info['total_tokens']
        lang_start, lang_end = fusion_info['language_token_range']
        vis_start, vis_end = fusion_info['vision_token_range']
        grid_size = fusion_info['vision_spatial_structure']
        
        if self.graph_type == 'grid':
            return self._build_grid_graph(
                batch_size, N_total, 
                lang_start, lang_end, vis_start, vis_end,
                grid_size, device
            )
        elif self.graph_type == 'fully_connected':
            return self._build_fully_connected_graph(
                batch_size, N_total, device
            )
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
    
    def _build_grid_graph(
        self, batch_size, N_total,
        lang_start, lang_end, vis_start, vis_end,
        grid_size, device
    ):
        edges = []
        
        # Language chain
        for i in range(lang_start, lang_end - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        
        # Vision grid
        for i in range(grid_size):
            for j in range(grid_size):
                node_idx = vis_start + i * grid_size + j
                
                neighbors = []
                if i > 0:
                    neighbors.append(vis_start + (i-1) * grid_size + j)
                if i < grid_size - 1:
                    neighbors.append(vis_start + (i+1) * grid_size + j)
                if j > 0:
                    neighbors.append(vis_start + i * grid_size + (j-1))
                if j < grid_size - 1:
                    neighbors.append(vis_start + i * grid_size + (j+1))
                
                for neighbor in neighbors:
                    edges.append([node_idx, neighbor])
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
        
        # Batch
        batch_edge_index = []
        for b in range(batch_size):
            offset = b * N_total
            batch_edge_index.append(edge_index + offset)
        
        edge_index = torch.cat(batch_edge_index, dim=1)
        
        return edge_index, None
    
    def _build_fully_connected_graph(self, batch_size, N_total, device):
        edges = []
        for i in range(N_total):
            for j in range(N_total):
                if i != j:
                    edges.append([i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
        
        batch_edge_index = []
        for b in range(batch_size):
            offset = b * N_total
            batch_edge_index.append(edge_index + offset)
        
        edge_index = torch.cat(batch_edge_index, dim=1)
        
        return edge_index, None
