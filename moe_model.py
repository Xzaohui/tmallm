"""
中文电商领域MoE语言模型
基于Qwen3架构的0.1-0.2B参数MoE模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import math
from typing import Optional, Tuple
import json


class RMSNorm(nn.Module):
    """RMSNorm实现，用于替代LayerNorm"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm


class Expert(nn.Module):
    """单个专家网络，使用SwiGLU激活函数"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class MoELayer(nn.Module):
    """MoE层实现，包含多个专家和路由机制"""
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim) for _ in range(num_experts)
        ])
        
        # 路由器
        self.router = nn.Linear(dim, num_experts, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 重塑输入以进行专家处理
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        
        # 计算路由概率
        router_logits = self.router(hidden_states_reshaped)
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # 选择top-k专家
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 处理选中的专家
        final_output = torch.zeros_like(hidden_states_reshaped)
        
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i]
            
            # 为每个专家创建掩码
            for exp_id in range(self.num_experts):
                mask = (expert_idx == exp_id)
                if mask.any():
                    expert_input = hidden_states_reshaped[mask]
                    expert_output = self.experts[exp_id](expert_input)
                    weighted_output = expert_output * expert_probs[mask].unsqueeze(-1)
                    final_output[mask] += weighted_output
        
        return final_output.view(batch_size, seq_len, hidden_dim)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        
        # 计算Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力到V
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer块，包含注意力和MoE层"""
    def __init__(self, dim: int, num_heads: int, intermediate_dim: int, 
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.moe = MoELayer(dim, intermediate_dim, num_experts, top_k)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 注意力子层
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output
        
        # MoE子层
        moe_output = self.moe(self.norm2(x))
        x = x + moe_output
        
        return x


class ChineseEcommerceMoE(nn.Module):
    """中文电商MoE语言模型"""
    def __init__(self, vocab_size: int = 151936, dim: int = 768, num_layers: int = 12,
                 num_heads: int = 12, intermediate_dim: int = 2048, 
                 num_experts: int = 8, top_k: int = 2, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # 词嵌入
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.embed_positions = nn.Embedding(max_seq_len, dim)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, intermediate_dim, num_experts, top_k)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # 嵌入
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)
        
        # 输出
        x = self.norm(x)
        logits = self.lm_head(x)
        
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss
        
        return logits
    
    @property
    def num_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def num_active_parameters(self):
        """计算激活参数数量（用于MoE模型）"""
        # 对于MoE模型，激活参数大约是总参数的1/4（假设top_k=2，num_experts=8）
        dense_params = sum(p.numel() for name, p in self.named_parameters() if 'moe.experts' not in name)
        expert_params = sum(p.numel() for name, p in self.named_parameters() if 'moe.experts' in name)
        active_expert_params = expert_params * (2 / 8)  # top_k / num_experts
        return dense_params + active_expert_params


def create_moe_model(model_size: str = "0.1B") -> ChineseEcommerceMoE:
    """创建不同大小的MoE模型"""
    configs = {
        "0.1B": {
            "vocab_size": 151936,
            "dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "intermediate_dim": 1024,
            "num_experts": 8,
            "top_k": 2,
            "max_seq_len": 4096
        },
        "0.2B": {
            "vocab_size": 151936,
            "dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_dim": 1536,
            "num_experts": 8,
            "top_k": 2,
            "max_seq_len": 4096
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Model size {model_size} not supported. Choose from {list(configs.keys())}")
    
    return ChineseEcommerceMoE(**configs[model_size])


# 加载Qwen3 tokenizer
def load_tokenizer():
    """加载Qwen3 tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load Qwen3 tokenizer: {e}")
        print("Using default tokenizer...")
        # 如果无法加载，使用基本的分词器
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        return tokenizer


if __name__ == "__main__":
    # 测试模型
    model = create_moe_model("0.1B")
    print(f"0.1B MoE Model - Total parameters: {model.num_parameters:,}")
    print(f"0.1B MoE Model - Active parameters: {model.num_active_parameters:,}")
    
    model_02 = create_moe_model("0.2B")
    print(f"0.2B MoE Model - Total parameters: {model_02.num_parameters:,}")
    print(f"0.2B MoE Model - Active parameters: {model_02.num_active_parameters:,}")
    
    # 测试前向传播
    tokenizer = load_tokenizer()
    test_text = "这是一个电商产品描述的例子"
    inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        print(f"Output shape: {outputs.shape}")