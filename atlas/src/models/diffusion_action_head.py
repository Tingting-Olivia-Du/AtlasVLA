"""
Diffusion-based Action Prediction Head
使用Diffusion Model预测动作，可以处理多模态动作分布
相比简单的回归head，diffusion model可以：
1. 处理动作的多模态分布
2. 生成更稳定的预测
3. 可以采样多个候选动作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [B] - 时间步
        Returns:
            embeddings: [B, dim] - 位置编码
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class DiffusionActionHead(nn.Module):
    """
    Diffusion-based动作预测头
    
    架构：
    - 使用简化的U-Net结构预测噪声
    - 支持条件特征（融合后的多模态特征）
    - 训练时预测噪声，推理时逐步去噪
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """
        Args:
            input_dim: 输入特征维度（融合后的特征）
            action_dim: 动作维度（默认7：6-DOF pose + gripper）
            hidden_dim: 隐藏层维度
            num_timesteps: Diffusion时间步数
            beta_start: 噪声调度起始值
            beta_end: 噪声调度结束值
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps
        
        # 噪声调度（线性）
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 条件特征投影
        self.cond_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 简化的U-Net结构（用于预测噪声）
        self.noise_predictor = nn.Sequential(
            # 输入层：动作 + 时间嵌入 + 条件特征
            nn.Linear(action_dim + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            
            # 中间层
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            
            # 输出层：预测噪声
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(
        self,
        fused_features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        return_noise: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            fused_features: [B, input_dim] - 融合后的多模态特征
            actions: [B, action_dim] - 训练时的真实动作（用于添加噪声）
            timesteps: [B] - 训练时的时间步（如果None，则随机采样）
            return_noise: 是否返回预测的噪声（用于调试）
            
        Returns:
            dict包含:
                - action: [B, action_dim] - 预测的动作
                - noise: [B, action_dim] - 预测的噪声（如果return_noise=True）
        """
        B = fused_features.shape[0]
        device = fused_features.device
        
        # 投影条件特征
        cond = self.cond_proj(fused_features)  # [B, hidden_dim]
        
        if self.training and actions is not None:
            # 训练模式：预测噪声
            if timesteps is None:
                # 随机采样时间步
                timesteps = torch.randint(0, self.num_timesteps, (B,), device=device)
            
            # 时间步嵌入
            t_emb = self.time_embed(timesteps)  # [B, hidden_dim]
            
            # 添加噪声到动作
            noise = torch.randn_like(actions)
            sqrt_alphas_cumprod_t = self.alphas_cumprod[timesteps].sqrt().view(B, 1)
            sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[timesteps]).sqrt().view(B, 1)
            noisy_actions = sqrt_alphas_cumprod_t * actions + sqrt_one_minus_alphas_cumprod_t * noise
            
            # 预测噪声
            noise_input = torch.cat([noisy_actions, t_emb, cond], dim=-1)
            predicted_noise = self.noise_predictor(noise_input)
            
            if return_noise:
                return {
                    "action": actions,  # 训练时返回原始动作
                    "noise": predicted_noise,
                    "noisy_action": noisy_actions,
                    "timesteps": timesteps
                }
            else:
                return {
                    "action": actions,
                    "noise": predicted_noise
                }
        else:
            # 推理模式：逐步去噪
            # 从纯噪声开始
            actions = torch.randn(B, self.action_dim, device=device)
            
            # 逐步去噪
            for t in reversed(range(self.num_timesteps)):
                timesteps_t = torch.full((B,), t, device=device, dtype=torch.long)
                t_emb = self.time_embed(timesteps_t)
                
                # 预测噪声
                noise_input = torch.cat([actions, t_emb, cond], dim=-1)
                predicted_noise = self.noise_predictor(noise_input)
                
                # 去噪步骤
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                
                # 预测原始动作
                pred_action = (actions - (1 - alpha_cumprod_t).sqrt() * predicted_noise) / alpha_cumprod_t.sqrt()
                
                # 更新动作（简化版本，完整版本需要更复杂的采样）
                if t > 0:
                    noise = torch.randn_like(actions)
                    actions = alpha_t.sqrt() * pred_action + beta_t.sqrt() * noise
                else:
                    actions = pred_action
            
            # 分离pose和gripper
            pose = actions[:, :6]
            gripper = actions[:, 6:7]
            
            # Gripper使用sigmoid确保在[0,1]范围
            gripper = torch.sigmoid(gripper)
            
            return {
                "action": torch.cat([pose, gripper], dim=-1),
                "pose": pose,
                "gripper": gripper
            }
