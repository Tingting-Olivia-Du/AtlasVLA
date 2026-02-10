"""
Flow Matching Action Head for VLA
==================================

相比原始 Diffusion Action Head 的改进：
1. DDPM → Flow Matching：训练目标更简单（直接回归向量场），无需 noise schedule
2. 单步预测 → Action Chunking：预测未来 T 步动作，提高时序一致性
3. MLP → Transformer：用 DiT-style Transformer 做向量场预测
4. Concat → AdaLN-Zero + Cross-Attention：更强的条件注入
5. 支持高阶 ODE solver（Euler / Midpoint / RK4）
6. 支持 Classifier-Free Guidance（CFG）

参考：
- Flow Matching (Lipman et al., 2023)
- π₀ (Physical Intelligence, 2024) 
- Diffusion Policy (Chi et al., 2023)
- DiT (Peebles & Xie, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Literal
from einops import rearrange


# =============================================================================
# 基础模块
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦时间步编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class LearnedPositionEmbedding(nn.Module):
    """可学习的 action chunk 位置编码"""
    def __init__(self, num_positions: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_positions, dim)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions)  # [T, dim]


# =============================================================================
# DiT-Style Transformer Block（核心模块）
# =============================================================================

class AdaLNZeroBlock(nn.Module):
    """
    DiT-style Adaptive LayerNorm Zero block
    
    和普通 Transformer block 的区别：
    - 用 AdaLN 替代普通 LayerNorm，让 time/condition 信息调制每一层
    - Zero-initialization：初始时 block 行为接近 identity，训练更稳定
    - 同时支持 self-attention 和 cross-attention
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, cross_attn: bool = True):
        super().__init__()
        self.dim = dim
        self.cross_attn_enabled = cross_attn

        # Self-Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Cross-Attention（对 vision-language 特征做 attention）
        if cross_attn:
            self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
            self.cross_attn_layer = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.norm_cross_kv = nn.LayerNorm(dim)  # for context

        # FFN
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

        # AdaLN-Zero 调制参数生成
        # 生成 6 个参数：gamma1, beta1, alpha1 (for self-attn), gamma2, beta2, alpha2 (for ffn)
        num_ada_params = 6
        if cross_attn:
            num_ada_params += 3  # gamma_cross, beta_cross, alpha_cross
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, num_ada_params * dim),
        )

        # Zero-init：让初始输出为0，block初始行为≈identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,               # [B, T, dim] action tokens
        ada_cond: torch.Tensor,         # [B, dim] time + global condition
        context: Optional[torch.Tensor] = None,  # [B, N, dim] vision-language tokens
    ) -> torch.Tensor:
        
        # 生成调制参数
        ada_params = self.adaLN_modulation(ada_cond)  # [B, num_params * dim]
        
        if self.cross_attn_enabled:
            (gamma1, beta1, alpha1,
             gamma_c, beta_c, alpha_c,
             gamma2, beta2, alpha2) = ada_params.chunk(9, dim=-1)
        else:
            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = ada_params.chunk(6, dim=-1)

        # --- Self-Attention with AdaLN ---
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)  # 调制
        h, _ = self.self_attn(h, h, h)
        x = x + alpha1.unsqueeze(1) * h  # gate + 残差

        # --- Cross-Attention with AdaLN ---
        if self.cross_attn_enabled and context is not None:
            h = self.norm_cross(x)
            h = h * (1 + gamma_c.unsqueeze(1)) + beta_c.unsqueeze(1)
            kv = self.norm_cross_kv(context)
            h, _ = self.cross_attn_layer(h, kv, kv)
            x = x + alpha_c.unsqueeze(1) * h

        # --- FFN with AdaLN ---
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.ffn(h)
        x = x + alpha2.unsqueeze(1) * h

        return x


# =============================================================================
# Flow Matching Action Head（主模块）
# =============================================================================

class FlowMatchingActionHead(nn.Module):
    """
    Flow Matching Action Head for VLA
    
    架构概览：
    ┌─────────────────────────────────────────────┐
    │  Input                                       │
    │  - x_t: [B, T_chunk, action_dim] noisy actions│
    │  - t: [B] flow time                          │
    │  - cond: [B, input_dim] or [B, N, input_dim]│
    │         fused vision-language features        │
    │                                              │
    │  ┌──────────────┐                            │
    │  │ Action Embed  │ → [B, T_chunk, dim]       │
    │  │ + Pos Embed   │                           │
    │  └──────┬───────┘                            │
    │         │                                    │
    │  ┌──────▼───────┐  ┌──────────────┐         │
    │  │  DiT Block   │←─│ AdaLN(t+cond)│         │
    │  │  Self-Attn   │  └──────────────┘         │
    │  │  Cross-Attn  │←── context tokens          │
    │  │  FFN         │                            │
    │  └──────┬───────┘  × N_layers               │
    │         │                                    │
    │  ┌──────▼───────┐                            │
    │  │ Output Proj   │ → [B, T_chunk, action_dim]│
    │  │ (velocity v)  │                           │
    │  └──────────────┘                            │
    └─────────────────────────────────────────────┘
    
    训练：MSE(v_pred, x1 - x0)
    推理：ODE 求解 x_t → x_1
    """

    def __init__(
        self,
        # 动作空间配置
        action_dim: int = 7,            # 7 = 6-DOF pose + 1 gripper
        action_horizon: int = 16,       # action chunk 长度（预测未来多少步）

        # 条件输入配置
        input_dim: int = 1024,          # 融合后的 vision-language 特征维度
        use_token_context: bool = True, # True: 输入是 token 序列 [B,N,D]; False: 单向量 [B,D]

        # Transformer 配置
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,

        # 推理配置
        num_inference_steps: int = 10,
        ode_solver: Literal["euler", "midpoint", "rk4"] = "euler",

        # Classifier-Free Guidance
        cfg_scale: float = 1.0,         # 1.0 = 不使用 CFG；>1.0 = 使用 CFG
        cfg_dropout_prob: float = 0.1,  # 训练时 condition dropout 概率
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim
        self.num_inference_steps = num_inference_steps
        self.ode_solver = ode_solver
        self.cfg_scale = cfg_scale
        self.cfg_dropout_prob = cfg_dropout_prob
        self.use_token_context = use_token_context

        # ---------- 时间步嵌入 ----------
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---------- 条件特征投影 ----------
        # 全局条件（用于 AdaLN 调制）
        self.global_cond_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Token 序列条件（用于 Cross-Attention）
        if use_token_context:
            self.context_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        # CFG 用的 null embedding
        self.null_cond = nn.Parameter(torch.zeros(1, hidden_dim))
        if use_token_context:
            self.null_context = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # ---------- Action 输入投影 + 位置编码 ----------
        self.action_input_proj = nn.Linear(action_dim, hidden_dim)
        self.action_pos_embed = LearnedPositionEmbedding(action_horizon, hidden_dim)

        # ---------- DiT Transformer Blocks ----------
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cross_attn=use_token_context,
            )
            for _ in range(num_layers)
        ])

        # ---------- 输出投影（预测向量场 v） ----------
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # Zero-init 输出层
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # ---------- Action 归一化统计量（重要！） ----------
        # 训练前应该用数据集统计量初始化这些值
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

    # =========================================================================
    # 归一化 / 反归一化
    # =========================================================================

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """将原始动作归一化到 ~N(0,1)"""
        return (actions - self.action_mean) / (self.action_std + 1e-8)

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """将归一化动作恢复到原始尺度"""
        return actions * self.action_std + self.action_mean

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """用数据集统计量设置归一化参数"""
        self.action_mean.copy_(mean)
        self.action_std.copy_(std)

    # =========================================================================
    # 向量场预测（核心网络）
    # =========================================================================

    def predict_velocity(
        self,
        x_t: torch.Tensor,         # [B, T, action_dim] 当前含噪 action chunk
        t: torch.Tensor,           # [B] flow time ∈ [0, 1]
        global_cond: torch.Tensor,  # [B, hidden_dim] 全局条件
        context: Optional[torch.Tensor] = None,  # [B, N, hidden_dim] token 条件
    ) -> torch.Tensor:
        """预测向量场 v_θ(x_t, t, condition)"""
        B, T, _ = x_t.shape

        # Action token 投影 + 位置编码
        h = self.action_input_proj(x_t)  # [B, T, dim]
        h = h + self.action_pos_embed(T, x_t.device).unsqueeze(0)  # 加位置编码

        # Time embedding + global condition → AdaLN 调制信号
        t_emb = self.time_embed(t)          # [B, dim]
        ada_cond = t_emb + global_cond      # 简单相加融合

        # Transformer blocks
        for block in self.blocks:
            h = block(h, ada_cond, context)

        # 输出投影
        h = self.final_norm(h)
        v = self.output_proj(h)  # [B, T, action_dim]
        return v

    # =========================================================================
    # 准备条件特征
    # =========================================================================

    def prepare_condition(
        self,
        fused_features: torch.Tensor,
        force_uncond: bool = False,
    ):
        """
        准备条件特征，支持 CFG dropout
        
        Args:
            fused_features: [B, D] 或 [B, N, D]
            force_uncond: 强制使用 null condition（CFG 推理时用）
        
        Returns:
            global_cond: [B, hidden_dim]
            context: [B, N, hidden_dim] or None
        """
        B = fused_features.shape[0]

        if force_uncond:
            global_cond = self.null_cond.expand(B, -1)
            context = self.null_context.expand(B, -1, -1) if self.use_token_context else None
            return global_cond, context

        # 处理不同输入格式
        if fused_features.dim() == 2:
            # [B, D] → 只有全局条件
            global_cond = self.global_cond_proj(fused_features)
            context = None
        elif fused_features.dim() == 3:
            # [B, N, D] → 全局条件（mean pool）+ token 序列
            global_cond = self.global_cond_proj(fused_features.mean(dim=1))
            context = self.context_proj(fused_features) if self.use_token_context else None
        else:
            raise ValueError(f"Unexpected fused_features shape: {fused_features.shape}")

        # 训练时 CFG dropout：随机把 condition 替换为 null
        if self.training and self.cfg_dropout_prob > 0:
            mask = torch.rand(B, device=fused_features.device) < self.cfg_dropout_prob
            if mask.any():
                global_cond[mask] = self.null_cond.expand(mask.sum(), -1)
                if context is not None:
                    context[mask] = self.null_context.expand(mask.sum(), -1, -1)

        return global_cond, context

    # =========================================================================
    # 前向传播
    # =========================================================================

    def forward(
        self,
        fused_features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_features: [B, D] or [B, N, D] - vision-language 特征
            actions: [B, T_chunk, action_dim] - 训练时的 GT action chunk
                     
        Returns:
            训练时: {"loss": scalar, "velocity_pred": [B,T,A], "velocity_target": [B,T,A]}
            推理时: {"action": [B,T,A], "action_chunk": [B,T,A]}
        """
        if self.training and actions is not None:
            return self._train_step(fused_features, actions)
        else:
            return self._inference(fused_features)

    # =========================================================================
    # 训练
    # =========================================================================

    def _train_step(
        self,
        fused_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Flow Matching 训练步骤
        
        核心公式：
            x_t = (1-t) * x_0 + t * x_1       (直线插值)
            target = x_1 - x_0                  (GT 向量场)
            loss = MSE(v_theta(x_t, t, c), target)
        """
        B = actions.shape[0]
        device = actions.device

        # 归一化动作
        x1 = self.normalize_actions(actions)  # [B, T, A]

        # 采样噪声和时间
        x0 = torch.randn_like(x1)
        # 注意：t 不要采样到 0 和 1 的边界，避免数值问题
        t = torch.rand(B, device=device) * 0.998 + 0.001  # t ∈ (0.001, 0.999)

        # 直线插值
        t_expand = t[:, None, None]  # [B, 1, 1]
        x_t = (1 - t_expand) * x0 + t_expand * x1

        # GT 向量场
        velocity_target = x1 - x0

        # 准备条件（含 CFG dropout）
        global_cond, context = self.prepare_condition(fused_features)

        # 预测向量场
        velocity_pred = self.predict_velocity(x_t, t, global_cond, context)

        # Loss
        loss = F.mse_loss(velocity_pred, velocity_target)

        return {
            "loss": loss,
            "velocity_pred": velocity_pred.detach(),
            "velocity_target": velocity_target.detach(),
        }

    # =========================================================================
    # 推理
    # =========================================================================

    @torch.no_grad()
    def _inference(
        self,
        fused_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Flow Matching 推理：ODE 求解从 x_0 (噪声) 到 x_1 (动作)
        支持 Euler / Midpoint / RK4
        """
        B = fused_features.shape[0]
        device = fused_features.device

        # 准备条件
        global_cond, context = self.prepare_condition(fused_features)

        # CFG：额外准备 unconditional 条件
        if self.cfg_scale > 1.0:
            uncond_global, uncond_context = self.prepare_condition(
                fused_features, force_uncond=True
            )

        # 从纯噪声出发
        x_t = torch.randn(B, self.action_horizon, self.action_dim, device=device)

        dt = 1.0 / self.num_inference_steps

        # ODE 求解
        for i in range(self.num_inference_steps):
            t_val = i / self.num_inference_steps
            t = torch.full((B,), t_val, device=device)

            # 计算向量场（可能带 CFG）
            v = self._get_velocity_with_cfg(
                x_t, t, global_cond, context,
                uncond_global if self.cfg_scale > 1.0 else None,
                uncond_context if self.cfg_scale > 1.0 else None,
            )

            # ODE 积分步
            if self.ode_solver == "euler":
                x_t = x_t + v * dt

            elif self.ode_solver == "midpoint":
                # Midpoint method（二阶精度）
                t_mid = torch.full((B,), t_val + 0.5 * dt, device=device)
                x_mid = x_t + v * (0.5 * dt)
                v_mid = self._get_velocity_with_cfg(
                    x_mid, t_mid, global_cond, context,
                    uncond_global if self.cfg_scale > 1.0 else None,
                    uncond_context if self.cfg_scale > 1.0 else None,
                )
                x_t = x_t + v_mid * dt

            elif self.ode_solver == "rk4":
                # Runge-Kutta 4th order
                t1 = t
                t2 = torch.full((B,), t_val + 0.5 * dt, device=device)
                t3 = t2
                t4 = torch.full((B,), t_val + dt, device=device)

                k1 = v
                k2 = self._get_velocity_with_cfg(
                    x_t + k1 * (0.5 * dt), t2, global_cond, context,
                    uncond_global if self.cfg_scale > 1.0 else None,
                    uncond_context if self.cfg_scale > 1.0 else None,
                )
                k3 = self._get_velocity_with_cfg(
                    x_t + k2 * (0.5 * dt), t3, global_cond, context,
                    uncond_global if self.cfg_scale > 1.0 else None,
                    uncond_context if self.cfg_scale > 1.0 else None,
                )
                k4 = self._get_velocity_with_cfg(
                    x_t + k3 * dt, t4, global_cond, context,
                    uncond_global if self.cfg_scale > 1.0 else None,
                    uncond_context if self.cfg_scale > 1.0 else None,
                )
                x_t = x_t + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)

        # 反归一化
        action_chunk = self.unnormalize_actions(x_t)  # [B, T, A]

        # 分离 pose 和 gripper
        pose = action_chunk[..., :6]
        gripper = torch.sigmoid(action_chunk[..., 6:7])  # gripper ∈ [0,1]
        action_chunk = torch.cat([pose, gripper], dim=-1)

        return {
            "action": action_chunk[:, 0],          # 当前步动作 [B, A]
            "action_chunk": action_chunk,           # 完整 chunk [B, T, A]
        }

    def _get_velocity_with_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        global_cond: torch.Tensor,
        context: Optional[torch.Tensor],
        uncond_global: Optional[torch.Tensor] = None,
        uncond_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算向量场，支持 Classifier-Free Guidance"""
        if self.cfg_scale <= 1.0 or uncond_global is None:
            return self.predict_velocity(x_t, t, global_cond, context)

        # CFG: v = v_uncond + scale * (v_cond - v_uncond)
        v_cond = self.predict_velocity(x_t, t, global_cond, context)
        v_uncond = self.predict_velocity(x_t, t, uncond_global, uncond_context)
        return v_uncond + self.cfg_scale * (v_cond - v_uncond)

    # =========================================================================
    # 多样本推理（利用 diffusion/flow 模型的优势）
    # =========================================================================

    @torch.no_grad()
    def sample_multiple(
        self,
        fused_features: torch.Tensor,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """
        采样多个候选动作，可用于：
        - 选择最优动作（配合 value function）
        - 评估动作分布的不确定性
        
        Args:
            fused_features: [B, D] or [B, N, D]
            num_samples: 每个样本采样多少个候选
            
        Returns:
            action_chunks: [B, num_samples, T, A]
        """
        B = fused_features.shape[0]

        # 扩展 features：[B, D] → [B*num_samples, D]
        if fused_features.dim() == 2:
            expanded = fused_features.unsqueeze(1).expand(-1, num_samples, -1)
            expanded = expanded.reshape(B * num_samples, -1)
        else:
            expanded = fused_features.unsqueeze(1).expand(-1, num_samples, -1, -1)
            expanded = expanded.reshape(B * num_samples, *fused_features.shape[1:])

        # 推理
        result = self._inference(expanded)
        chunks = result["action_chunk"]  # [B*S, T, A]

        return chunks.reshape(B, num_samples, self.action_horizon, self.action_dim)


# =============================================================================
# Temporal Ensemble（推理时的 action chunk 融合）
# =============================================================================

class TemporalEnsemble:
    """
    Temporal Ensemble for Action Chunking
    
    问题：每步推理产生一个 action chunk，连续两步的 chunk 有重叠部分。
    解决：对重叠部分加权平均，用指数衰减权重（越新的 chunk 权重越大）。
    
    这是 Diffusion Policy 论文中的关键技巧。
    """
    def __init__(self, action_horizon: int, action_dim: int, decay: float = 0.01):
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.decay = decay

        # Buffer: 存储 pending actions 和权重
        self.action_buffer = None  # [T_buffer, A]
        self.weight_buffer = None  # [T_buffer]
        self.step = 0

    def reset(self):
        self.action_buffer = None
        self.weight_buffer = None
        self.step = 0

    def add_chunk(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        添加新的 action chunk 并返回当前步的 ensemble 动作
        
        Args:
            action_chunk: [T_chunk, A] 新预测的动作序列
            
        Returns:
            action: [A] 当前步的加权平均动作
        """
        T = action_chunk.shape[0]
        device = action_chunk.device

        if self.action_buffer is None:
            # 首次：直接用
            self.action_buffer = action_chunk.clone()
            self.weight_buffer = torch.ones(T, device=device)
        else:
            # 移除已执行的第一步
            self.action_buffer = self.action_buffer[1:]
            self.weight_buffer = self.weight_buffer[1:]

            # 当前 buffer 长度
            cur_len = self.action_buffer.shape[0]

            if cur_len > 0:
                # 重叠部分：加权融合
                overlap = min(cur_len, T)
                new_weight = math.exp(-self.decay * self.step)

                # 加权求和
                self.action_buffer[:overlap] = (
                    self.action_buffer[:overlap] * self.weight_buffer[:overlap, None]
                    + action_chunk[:overlap] * new_weight
                ) / (self.weight_buffer[:overlap, None] + new_weight)
                self.weight_buffer[:overlap] += new_weight

                # 非重叠部分（新 chunk 多出来的部分）
                if T > overlap:
                    self.action_buffer = torch.cat([
                        self.action_buffer,
                        action_chunk[overlap:]
                    ], dim=0)
                    self.weight_buffer = torch.cat([
                        self.weight_buffer,
                        torch.full((T - overlap,), new_weight, device=device)
                    ], dim=0)
            else:
                self.action_buffer = action_chunk.clone()
                self.weight_buffer = torch.ones(T, device=device)

        self.step += 1
        return self.action_buffer[0]  # 返回当前步动作


# =============================================================================
# Loss 计算（供外部训练脚本调用）
# =============================================================================

def compute_flow_matching_loss(
    model: FlowMatchingActionHead,
    fused_features: torch.Tensor,
    gt_action_chunk: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    计算 Flow Matching loss
    
    Args:
        model: FlowMatchingActionHead
        fused_features: [B, D] 或 [B, N, D]
        gt_action_chunk: [B, T, A] ground truth 动作序列
        
    Returns:
        dict with "loss" and optional debug info
    """
    return model(fused_features, actions=gt_action_chunk)


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    head = FlowMatchingActionHead(
        action_dim=7,
        action_horizon=16,
        input_dim=1024,
        use_token_context=True,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        num_inference_steps=10,
        ode_solver="euler",         # "euler", "midpoint", "rk4"
        cfg_scale=1.5,              # >1 启用 classifier-free guidance
        cfg_dropout_prob=0.1,
    ).to(device)

    # ---------- 训练 ----------
    head.train()
    batch_size = 4

    # 模拟输入
    # 方式1：token 序列输入（来自 VLM 的 token 特征）
    fused_features = torch.randn(batch_size, 50, 1024, device=device)  # 50 tokens
    # 方式2：单向量输入
    # fused_features = torch.randn(batch_size, 1024, device=device)

    gt_actions = torch.randn(batch_size, 16, 7, device=device)  # GT action chunk

    result = head(fused_features, actions=gt_actions)
    print(f"Training loss: {result['loss'].item():.4f}")

    # ---------- 推理 ----------
    head.eval()
    result = head(fused_features)
    print(f"Predicted action shape: {result['action'].shape}")           # [B, 7]
    print(f"Predicted chunk shape: {result['action_chunk'].shape}")      # [B, 16, 7]

    # ---------- 多样本采样 ----------
    samples = head.sample_multiple(fused_features, num_samples=10)
    print(f"Multi-sample shape: {samples.shape}")                        # [B, 10, 16, 7]

    # ---------- Temporal Ensemble 使用 ----------
    ensemble = TemporalEnsemble(action_horizon=16, action_dim=7)

    for step in range(5):
        result = head(fused_features[:1])  # 单样本
        chunk = result["action_chunk"][0]  # [16, 7]
        action = ensemble.add_chunk(chunk)
        print(f"Step {step}: ensemble action shape = {action.shape}")    # [7]

    # ---------- 参数量统计 ----------
    total_params = sum(p.numel() for p in head.parameters())
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")