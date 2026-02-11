"""
语言编码器 - 使用 Qwen3-0.6B-Base
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, List

class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load Qwen3 model
        print(f"Loading language model: {config.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {config.model_name}: {e}")
            print("Falling back to Qwen2-0.5B...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2-0.5B",
                trust_remote_code=True
            )
        
        try:
            self.language_model = AutoModel.from_pretrained(
                config.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not load model from {config.model_name}: {e}")
            print("Falling back to Qwen2-0.5B...")
            self.language_model = AutoModel.from_pretrained(
                "Qwen/Qwen2-0.5B",
                trust_remote_code=True
            )
        
        self.language_hidden_size = self.language_model.config.hidden_size
        
        # Projector: 将language model的hidden size投影到目标维度
        self.projector = nn.Sequential(
            nn.Linear(self.language_hidden_size, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim)
        )
        
        if config.freeze_encoder:
            for param in self.language_model.parameters():
                param.requires_grad = False
            print(f"✓ Language encoder frozen ({config.model_name})")
        
        # 1D Positional encoding
        self.max_length = config.max_length
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_length, config.output_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        print(f"✓ Language encoder initialized: {config.model_name}")
        print(f"  - Hidden size: {self.language_hidden_size}")
        print(f"  - Output dim: {config.output_dim}")
        print(f"  - Max length: {config.max_length}")
        
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            max_length=self.config.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.language_model.device)
        attention_mask = encoded['attention_mask'].to(self.language_model.device)
        
        # Encode
        with torch.set_grad_enabled(not self.config.freeze_encoder):
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        language_features = outputs.last_hidden_state
        language_tokens = self.projector(language_features)
        
        # Add positional encoding
        seq_len = language_tokens.size(1)
        language_tokens = language_tokens + self.pos_embed[:, :seq_len, :]
        
        actual_lengths = attention_mask.sum(dim=1)
        language_info = {
            'max_length': seq_len,
            'actual_lengths': actual_lengths,
            'has_spatial_structure': False,
            'num_tokens': seq_len
        }
        
        return language_tokens, attention_mask, language_info
