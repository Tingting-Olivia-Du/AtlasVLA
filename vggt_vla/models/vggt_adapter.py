"""
VGGT Adapter - 加载官方 facebook/VGGT-1B 权重并适配到 VLA 任务
专门处理单帧输入 + 语言指令

官方权重通过 model.pt 下载（与 vggt demo 一致），非 Transformers AutoModel。
"""
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

# 官方 VGGT-1B 权重 URL（与 vggt 仓库 demo 一致）
VGGT_1B_WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def _get_vggt_module():
    """导入本地 vggt 包中的 VGGT 类"""
    vggt_path = os.path.join(os.path.dirname(__file__), '../../vggt')
    if vggt_path not in sys.path:
        sys.path.insert(0, vggt_path)
    from vggt.models.vggt import VGGT
    return VGGT


class VGGTAdapter(nn.Module):
    """
    适配 facebook/VGGT-1B 到 VLA 任务:
    1. ✅ 处理单帧输入 (原始VGGT设计用于视频序列，我们适配为单帧)
    2. ✅ 注入 language tokens (通过特殊的融合机制)
    3. ✅ 提取适合 action prediction 的特征

    官方权重加载方式：本地 VGGT 结构 + 下载 model.pt 的 state_dict（与官方 demo 一致）。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        print("\n" + "="*60)
        print("Loading official facebook/VGGT-1B weights...")
        print("="*60)

        VGGT = _get_vggt_module()

        # 官方 VGGT-1B 结构（与官方 repo 一致，才能正确加载 state_dict）
        self.vggt = VGGT(
            img_size=518,
            patch_size=14,
            embed_dim=1024,
            enable_camera=True,
            enable_point=True,
            enable_depth=True,
            enable_track=True,
        )
        self.use_pretrained_vggt = False

        try:
            state_dict = torch.hub.load_state_dict_from_url(
                VGGT_1B_WEIGHTS_URL,
                map_location="cpu",
                progress=True,
            )
            # 若返回的是包装 dict（如 {'model': state_dict}），则取出
            if isinstance(state_dict, dict) and "model" in state_dict and len(state_dict) == 1:
                state_dict = state_dict["model"]
            self.vggt.load_state_dict(state_dict, strict=True)
            self.use_pretrained_vggt = True
            print("✓ Loaded official facebook/VGGT-1B weights from HuggingFace (model.pt)")
        except Exception as e:
            print(f"⚠ Could not load official weights: {e}")
            print("  Using VGGT structure with random initialization (no pretrained weights).")

        # 检查 aggregator（本 adapter 只使用这部分）
        if hasattr(self.vggt, 'aggregator'):
            agg = self.vggt.aggregator
            has_frame = hasattr(agg, 'frame_blocks')
            has_global = hasattr(agg, 'global_blocks')
            print(f"  - Aggregator: ✓ (frame_blocks={len(agg.frame_blocks) if has_frame else 0}, global_blocks={len(agg.global_blocks) if has_global else 0})")
        else:
            print("  - Warning: No aggregator found.")
        
        # VGGT的embedding维度
        self.vggt_embed_dim = 1024  # facebook/VGGT-1B 默认
        self.target_dim = config.embed_dim  # 我们的目标维度 (768)
        
        print(f"  VGGT embedding dim: {self.vggt_embed_dim}")
        print(f"  Target dim: {self.target_dim}")
        
        # Language token 投影层: target_dim -> vggt_embed_dim
        # Qwen3-0.6B输出需要投影到VGGT空间
        self.lang_adapter = nn.Sequential(
            nn.Linear(self.target_dim, self.vggt_embed_dim),
            nn.LayerNorm(self.vggt_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        print(f"  Language adapter: {self.target_dim} -> {self.vggt_embed_dim}")
        
        # Vision token 投影层 (如果维度不匹配)
        if self.target_dim != self.vggt_embed_dim:
            self.vision_adapter = nn.Sequential(
                nn.Linear(self.target_dim, self.vggt_embed_dim),
                nn.LayerNorm(self.vggt_embed_dim)
            )
            print(f"  Vision adapter: {self.target_dim} -> {self.vggt_embed_dim}")
        else:
            self.vision_adapter = nn.Identity()
            print("  Vision adapter: Identity (dims match)")
        
        # Cross-modal attention: language attends to vision
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.vggt_embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(self.vggt_embed_dim)
        
        # 特征提取层：从 VGGT 输出提取 VLA 特征
        # VGGT aggregator 输出是 list of [B, S, P, 2C]，我们取最后一层
        self.feature_projector = nn.Sequential(
            nn.Linear(self.vggt_embed_dim * 2, self.vggt_embed_dim),
            nn.LayerNorm(self.vggt_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.vggt_embed_dim, self.target_dim),
            nn.LayerNorm(self.target_dim)
        )
        print(f"  Feature projector: {self.vggt_embed_dim * 2} -> {self.target_dim}")
        
        # Action queries (可学习的query tokens用于action prediction)
        self.num_action_queries = 16
        self.action_queries = nn.Parameter(
            torch.randn(1, self.num_action_queries, self.target_dim)
        )
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        print(f"  Action queries: {self.num_action_queries} learnable tokens")
        
        # 冻结VGGT backbone (可选)
        if config.freeze_vggt:
            print("\n  🔒 Freezing VGGT backbone...")
            for param in self.vggt.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  ✓ Trainable parameters: {trainable / 1e6:.2f}M (adapter layers only)")
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  Trainable parameters: {trainable / 1e6:.2f}M (full model)")
        
        print("="*60 + "\n")
    
    def forward(
        self,
        vision_tokens: torch.Tensor,      # [B, N_v, D]
        language_tokens: torch.Tensor,    # [B, N_l, D]
        vision_info: Dict,
        language_info: Dict,
        language_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        处理单帧输入的forward pass
        
        Args:
            vision_tokens: [B, N_v, D] - 来自vision encoder的tokens (单帧)
            language_tokens: [B, N_l, D] - 来自Qwen3的language tokens
            
        Returns:
            vision_features: [B, N_v, D]
            language_features: [B, N_l, D]
            global_features: [B, num_queries, D]
            output_info: Dict
            
        流程:
        1. 将单帧vision tokens适配到VGGT空间
        2. Language tokens通过cross-attention与vision交互
        3. 使用VGGT处理fused features
        4. 提取action-relevant features
        """
        B = vision_tokens.size(0)
        N_v = vision_tokens.size(1)
        N_l = language_tokens.size(1)
        device = vision_tokens.device
        
        # ========== Step 1: 适配到VGGT空间 ==========
        vision_adapted = self.vision_adapter(vision_tokens)  # [B, N_v, 1024]
        language_adapted = self.lang_adapter(language_tokens)  # [B, N_l, 1024]
        
        # ========== Step 2: Cross-modal interaction ==========
        # Language attends to vision (language-conditioned visual features)
        language_enhanced, _ = self.cross_attn(
            query=language_adapted,
            key=vision_adapted,
            value=vision_adapted,
            need_weights=False
        )
        language_enhanced = self.cross_attn_norm(language_enhanced + language_adapted)
        
        # ========== Step 3: 使用VGGT aggregator处理tokens ==========
        # VGGT aggregator期望tokens形状:
        # - Frame attention: [B*S, P, C] 其中 P 是每个frame的token数
        # - Global attention: [B, S*P, C]
        # 对于单帧输入，S=1
        
        try:
            aggregator = self.vggt.aggregator if hasattr(self.vggt, 'aggregator') else None
            
            if aggregator is not None and hasattr(aggregator, 'frame_blocks') and hasattr(aggregator, 'global_blocks'):
                # 合并vision和language tokens
                # 形状: [B, N_v+N_l, C] = [B, P, C] 其中 P = N_v + N_l
                combined_tokens = torch.cat([vision_adapted, language_enhanced], dim=1)  # [B, P, C]
                
                B = combined_tokens.size(0)
                S = 1  # 单帧输入
                P = combined_tokens.size(1)  # 总token数 (N_v + N_l)
                C = combined_tokens.size(2)  # embed_dim (1024)
                
                # 检查维度是否匹配
                if C != self.vggt_embed_dim:
                    raise ValueError(f"Token dimension {C} does not match VGGT embed_dim {self.vggt_embed_dim}")
                
                # 构造位置编码 (如果需要)
                pos = None
                if hasattr(aggregator, 'position_getter') and aggregator.position_getter is not None:
                    # 从vision_info获取grid信息来构造位置编码
                    grid_size = vision_info.get('grid_size', int(N_v ** 0.5))
                    # 为vision tokens构造2D位置编码
                    # 为language tokens添加虚拟位置（使用vision tokens的最后一个位置）
                    if grid_size > 0 and N_v > 0:
                        try:
                            # 构造vision tokens的2D位置
                            vision_pos = aggregator.position_getter(
                                B * S, grid_size, grid_size, device=combined_tokens.device
                            )  # [B*S, grid_size*grid_size, 2]
                            
                            # 如果vision_pos的token数少于N_v，需要扩展或截断
                            if vision_pos.size(1) < N_v:
                                # 扩展：重复最后一个位置
                                last_pos = vision_pos[:, -1:, :]
                                padding = last_pos.expand(-1, N_v - vision_pos.size(1), -1)
                                vision_pos = torch.cat([vision_pos, padding], dim=1)
                            elif vision_pos.size(1) > N_v:
                                # 截断：只取前N_v个
                                vision_pos = vision_pos[:, :N_v, :]
                            
                            # 为language tokens添加位置（使用最后一个vision token的位置）
                            if N_l > 0:
                                lang_pos = vision_pos[:, -1:, :].expand(-1, N_l, -1)  # [B*S, N_l, 2]
                                # 合并位置编码
                                pos = torch.cat([vision_pos, lang_pos], dim=1)  # [B*S, P, 2]
                            else:
                                pos = vision_pos  # [B*S, N_v, 2]
                        except Exception as e:
                            # 如果位置编码构造失败，使用None
                            pos = None
                
                # 使用VGGT的alternating attention机制
                # 参考 aggregator._process_frame_attention 和 _process_global_attention
                frame_idx = 0
                global_idx = 0
                frame_intermediates = []
                global_intermediates = []
                
                # 获取alternating attention的配置
                aa_order = aggregator.aa_order if hasattr(aggregator, 'aa_order') else ["frame", "global"]
                aa_block_num = aggregator.aa_block_num if hasattr(aggregator, 'aa_block_num') else len(aggregator.frame_blocks)
                aa_block_size = aggregator.aa_block_size if hasattr(aggregator, 'aa_block_size') else 1
                
                tokens = combined_tokens
                
                for _ in range(aa_block_num):
                    for attn_type in aa_order:
                        if attn_type == "frame":
                            # Frame attention: [B*S, P, C]
                            if tokens.shape != (B * S, P, C):
                                tokens = tokens.view(B, S, P, C).view(B * S, P, C)
                            
                            if pos is not None and pos.shape != (B * S, P, 2):
                                pos_frame = pos.view(B, S, P, 2).view(B * S, P, 2)
                            else:
                                pos_frame = pos
                            
                            # 处理frame blocks
                            for _ in range(aa_block_size):
                                if frame_idx < len(aggregator.frame_blocks):
                                    if self.training:
                                        tokens = torch.utils.checkpoint.checkpoint(
                                            aggregator.frame_blocks[frame_idx], 
                                            tokens, pos_frame, 
                                            use_reentrant=aggregator.use_reentrant if hasattr(aggregator, 'use_reentrant') else False
                                        )
                                    else:
                                        tokens = aggregator.frame_blocks[frame_idx](tokens, pos=pos_frame)
                                    frame_idx += 1
                                    frame_intermediates.append(tokens.view(B, S, P, C))
                        
                        elif attn_type == "global":
                            # Global attention: [B, S*P, C]
                            if tokens.shape != (B, S * P, C):
                                tokens = tokens.view(B, S, P, C).view(B, S * P, C)
                            
                            if pos is not None and pos.shape != (B, S * P, 2):
                                pos_global = pos.view(B, S, P, 2).view(B, S * P, 2)
                            else:
                                pos_global = pos
                            
                            # 处理global blocks
                            for _ in range(aa_block_size):
                                if global_idx < len(aggregator.global_blocks):
                                    if self.training:
                                        tokens = torch.utils.checkpoint.checkpoint(
                                            aggregator.global_blocks[global_idx],
                                            tokens, pos_global,
                                            use_reentrant=aggregator.use_reentrant if hasattr(aggregator, 'use_reentrant') else False
                                        )
                                    else:
                                        tokens = aggregator.global_blocks[global_idx](tokens, pos=pos_global)
                                    global_idx += 1
                                    global_intermediates.append(tokens.view(B, S, P, C))
                
                # 合并frame和global的中间特征（类似aggregator的输出）
                # 取最后一层的输出
                if len(frame_intermediates) > 0 and len(global_intermediates) > 0:
                    # 对齐长度
                    min_len = min(len(frame_intermediates), len(global_intermediates))
                    frame_final = frame_intermediates[-1]  # [B, S, P, C]
                    global_final = global_intermediates[-1]  # [B, S, P, C]
                    
                    # Concat frame和global特征: [B, S, P, 2C]
                    concat_features = torch.cat([frame_final, global_final], dim=-1)  # [B, 1, P, 2C]
                    concat_features = concat_features.squeeze(1)  # [B, P, 2C]
                    
                    # 投影回目标维度
                    x_projected = self.feature_projector(concat_features)  # [B, P, D]
                else:
                    # Fallback: 如果没有中间特征，直接使用最后的tokens
                    tokens_final = tokens.view(B, S, P, C).squeeze(1)  # [B, P, C]
                    x_projected = self.feature_projector(torch.cat([tokens_final, tokens_final], dim=-1))
                
            else:
                # Fallback: aggregator不可用
                raise AttributeError("VGGT aggregator does not have required attributes")
                
        except Exception as e:
            import traceback
            # 只在第一次错误时打印详细traceback，避免日志过多
            if not hasattr(self, '_vggt_error_logged'):
                error_msg = f"Error in VGGT processing: {e}\n{traceback.format_exc()}"
                print(f"Warning: {error_msg}")
                print("Using simple concatenation fallback")
                self._vggt_error_logged = True
            else:
                # 后续错误只打印简短信息
                print(f"Warning: VGGT processing error (using fallback): {type(e).__name__}")
            
            x = torch.cat([vision_adapted, language_enhanced], dim=1)
            x_projected = self.feature_projector(torch.cat([x, x], dim=-1))
        
        # ========== Step 4: 分离features ==========
        vision_features = x_projected[:, :N_v, :]       # [B, N_v, D]
        language_features = x_projected[:, N_v:, :]     # [B, N_l, D]
        
        # ========== Step 5: 生成global features for action ==========
        # 使用可学习的action queries
        global_features = self.action_queries.expand(B, -1, -1)  # [B, num_queries, D]
        
        # Attention-based feature aggregation
        # Action queries attend to all features
        all_features = x_projected  # [B, N_v+N_l, D]
        pooled = all_features.mean(dim=1, keepdim=True).expand(-1, self.num_action_queries, -1)
        global_features = global_features + 0.1 * pooled  # Weighted combination
        
        output_info = {
            'vggt_embed_dim': self.vggt_embed_dim,
            'target_dim': self.target_dim,
            'num_vision_tokens': N_v,
            'num_language_tokens': N_l,
            'num_action_queries': self.num_action_queries,
            'single_frame_input': True  # 标记这是单帧输入
        }
        
        return vision_features, language_features, global_features, output_info


class SimpleVGGTBackbone(nn.Module):
    """
    简化版VGGT Backbone - 不依赖HuggingFace
    用于快速实验和调试
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        from .components.token_fusion import TokenFusion
        from .components.graph_builder import GraphBuilder
        from .components.vggt_layers import VGGTLayer
        
        self.token_fusion = TokenFusion(config)
        self.graph_builder = GraphBuilder(config)
        
        self.layers = nn.ModuleList([
            VGGTLayer(config)
            for _ in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Action queries
        self.num_action_queries = 16
        self.action_queries = nn.Parameter(
            torch.randn(1, self.num_action_queries, config.embed_dim)
        )
        nn.init.trunc_normal_(self.action_queries, std=0.02)
    
    def forward(
        self,
        vision_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        vision_info: Dict,
        language_info: Dict,
        language_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        
        B = vision_tokens.size(0)
        device = vision_tokens.device
        
        # Token Fusion
        fused_tokens, attention_mask, fusion_info = self.token_fusion(
            vision_tokens, language_tokens,
            vision_info, language_info, language_mask
        )
        
        # Build Graph
        edge_index, edge_attr = self.graph_builder.build_graph(
            fusion_info, batch_size=B, device=device
        )
        
        # VGGT Layers
        x = fused_tokens
        for layer in self.layers:
            x = layer(x, edge_index, attn_mask=attention_mask)
        
        x = self.norm(x)
        
        # Split features
        lang_start, lang_end = fusion_info['language_token_range']
        vis_start, vis_end = fusion_info['vision_token_range']
        
        language_features = x[:, lang_start:lang_end, :]
        vision_features = x[:, vis_start:vis_end, :]
        
        # Global features
        global_features = self.action_queries.expand(B, -1, -1)
        
        output_info = fusion_info.copy()
        output_info['num_action_queries'] = self.num_action_queries
        
        return vision_features, language_features, global_features, output_info
