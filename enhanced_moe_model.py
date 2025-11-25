"""
增强版中文电商MoE语言模型
支持0.6B和0.1B参数规模，包含完整的训练功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
import math
from typing import Optional, Tuple, Dict, Any
import json
import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """模型配置类"""
    vocab_size: int = 151936
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 2048
    num_experts: int = 8
    num_experts_per_tok: int = 2
    max_position_embeddings: int = 4096
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

class RMSNorm(nn.Module):
    """RMSNorm实现"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Expert(nn.Module):
    """专家网络，使用SwiGLU激活"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.act_fn(gate) * up)

class MoELayer(nn.Module):
    """MoE层实现"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(self.num_experts)
        ])
        
        # 路由器
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # 辅助损失系数
        self.aux_loss_coef = 1e-2
        self.router_z_loss_coef = 1e-3
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # 计算路由概率
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # 准备输出
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # 逐个专家处理
        for i in range(self.top_k):
            expert_mask = selected_experts[:, i]
            expert_weights = routing_weights[:, i:i+1]
            
            for expert_id in range(self.num_experts):
                expert_mask_i = expert_mask == expert_id
                if expert_mask_i.any():
                    expert_input = hidden_states[expert_mask_i]
                    expert_output = self.experts[expert_id](expert_input)
                    final_hidden_states[expert_mask_i] += expert_weights[expert_mask_i] * expert_output
        
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        
        # 计算辅助损失
        aux_loss = self._compute_aux_loss(router_logits, routing_weights, selected_experts)
        
        return final_hidden_states, aux_loss
    
    def _compute_aux_loss(self, router_logits: torch.Tensor, routing_weights: torch.Tensor, 
                         selected_experts: torch.Tensor) -> torch.Tensor:
        """计算辅助损失用于负载均衡"""
        # Router z-loss
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        
        # 负载均衡损失
        aux_loss = self.router_z_loss_coef * router_z_loss
        
        return aux_loss

class Attention(nn.Module):
    """多头注意力机制"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = self._init_rope()
        
    def _init_rope(self):
        """初始化RoPE位置编码"""
        from transformers import LlamaRotaryEmbedding
        return LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=10000,
        )
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # 注意力计算
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """应用RoPE位置编码"""
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """旋转一半的维度"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MoELayer(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # 自注意力
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states
        
        # MoE层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, aux_loss

class ChineseEcommerceMoE(nn.Module):
    """中文电商MoE语言模型"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, 
                labels=None, return_dict=None):
        
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            raise ValueError("input_ids must be provided")
        
        # 位置编码
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        # 输入嵌入
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        # 注意力掩码
        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds.dtype, device=input_ids.device
            )
        
        # 通过解码器层
        total_aux_loss = 0
        for decoder_layer in self.layers:
            hidden_states, aux_loss = decoder_layer(hidden_states, attention_mask, position_ids)
            total_aux_loss += aux_loss
        
        hidden_states = self.norm(hidden_states)
        
        # 语言模型头
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # 计算语言模型损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
            # 总损失 = 语言模型损失 + 辅助损失
            loss = lm_loss + total_aux_loss / self.num_hidden_layers
        
        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
            'aux_loss': total_aux_loss
        }
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, dtype, device):
        """准备解码器注意力掩码"""
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, dtype=torch.bool, device=device)
        
        # 创建因果掩码
        batch_size, seq_length = input_shape
        causal_mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device))
        
        # 扩展维度
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # 结合掩码
        combined_mask = extended_attention_mask & causal_mask
        
        # 转换为注意力权重掩码
        attention_mask = torch.where(combined_mask, 0.0, -torch.finfo(dtype).max)
        
        return attention_mask
    
    @property
    def num_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def num_active_parameters(self):
        """计算激活参数数量"""
        dense_params = sum(p.numel() for name, p in self.named_parameters() if 'mlp.experts' not in name)
        expert_params = sum(p.numel() for name, p in self.named_parameters() if 'mlp.experts' in name)
        active_expert_params = expert_params * (self.config.num_experts_per_tok / self.config.num_experts)
        return dense_params + active_expert_params


def create_moe_model(model_size: str = "0.1B") -> ChineseEcommerceMoE:
    """创建不同大小的MoE模型"""
    configs = {
        "0.1B": ModelConfig(
            vocab_size=151936,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=1024,
            num_experts=8,
            num_experts_per_tok=2,
            max_position_embeddings=4096
        ),
        "0.6B": ModelConfig(
            vocab_size=151936,
            hidden_size=1024,
            num_hidden_layers=16,
            num_attention_heads=16,
            intermediate_size=2048,
            num_experts=8,
            num_experts_per_tok=2,
            max_position_embeddings=4096
        )
    }
    
    if model_size not in configs:
        raise ValueError(f"Model size {model_size} not supported. Choose from {list(configs.keys())}")
    
    return ChineseEcommerceMoE(configs[model_size])


def load_tokenizer():
    """加载Qwen3 tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load Qwen3 tokenizer: {e}")
        print("Using default tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        return tokenizer


if __name__ == "__main__":
    # 测试不同规模的模型
    for size in ["0.1B", "0.6B"]:
        model = create_moe_model(size)
        print(f"{size} MoE Model - Total parameters: {model.num_parameters:,}")
        print(f"{size} MoE Model - Active parameters: {model.num_active_parameters:,}")
        print(f"{size} MoE Model - Parameter efficiency: {model.num_active_parameters/model.num_parameters:.2%}")
        print("-" * 50)