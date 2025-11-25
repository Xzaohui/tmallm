"""
MoE模型强化学习对齐脚本 (RLHF)
实现PPO算法进行人类反馈强化学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import json
import os
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional, Tuple
import numpy as np
import random
from datasets import load_dataset
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from enhanced_moe_model import create_moe_model, ChineseEcommerceMoE, load_tokenizer


class RewardModel(nn.Module):
    """奖励模型 - 评估生成文本的质量"""
    def __init__(self, model: ChineseEcommerceMoE):
        super().__init__()
        self.model = model
        self.reward_head = nn.Linear(model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        """前向传播，返回奖励分数"""
        # 获取最后一个隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs['hidden_states']
        
        # 使用最后一个token的隐藏状态计算奖励
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        reward = self.reward_head(last_hidden)  # [batch_size, 1]
        
        return reward.squeeze(-1)  # [batch_size]


class PPOTrainer:
    """PPO训练器 - 强化学习对齐"""
    def __init__(self, 
                 policy_model: ChineseEcommerceMoE,
                 reference_model: ChineseEcommerceMoE,
                 reward_model: RewardModel,
                 tokenizer,
                 learning_rate: float = 1e-6,
                 ppo_epochs: int = 4,
                 mini_batch_size: int = 4,
                 gradient_accumulation_steps: int = 1,
                 cliprange: float = 0.2,
                 cliprange_value: float = 0.2,
                 vf_coef: float = 0.1,
                 ent_coef: float = 0.01,
                 target_kl: float = 0.01,
                 save_dir: str = "./rlhf_checkpoints"):
        
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        
        # PPO参数
        self.learning_rate = learning_rate
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.cliprange = cliprange
        self.cliprange_value = cliprange_value
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_model.to(self.device)
        self.reference_model.to(self.device)
        self.reward_model.to(self.device)
        
        # 冻结参考模型
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # 保存目录
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练状态
        self.global_step = 0
        
        # 初始化wandb
        try:
            wandb.init(
                project="chinese-ecommerce-moe-rlhf",
                config={
                    "learning_rate": learning_rate,
                    "ppo_epochs": ppo_epochs,
                    "mini_batch_size": mini_batch_size,
                    "cliprange": cliprange,
                    "vf_coef": vf_coef,
                    "ent_coef": ent_coef
                }
            )
        except:
            logger.warning("Wandb not available, continuing without logging...")
    
    def generate_responses(self, queries: List[str], max_new_tokens: int = 128) -> Tuple[List[str], List[torch.Tensor]]:
        """生成响应和对应的logits"""
        responses = []
        response_logits = []
        
        for query in queries:
            # 编码输入
            inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成响应
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # 解码响应
            response = self.tokenizer.decode(outputs.sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            responses.append(response)
            
            # 计算logits
            logits = torch.stack(outputs.scores, dim=1)  # [batch_size, seq_len, vocab_size]
            response_logits.append(logits[0])  # 移除batch维度
        
        return responses, response_logits
    
    def compute_rewards(self, queries: List[str], responses: List[str]) -> torch.Tensor:
        """计算奖励分数"""
        rewards = []
        
        for query, response in zip(queries, responses):
            # 组合查询和响应
            full_text = query + response
            inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 计算奖励
            with torch.no_grad():
                reward = self.reward_model(**inputs)
            rewards.append(reward.item())
        
        return torch.tensor(rewards, device=self.device)
    
    def compute_kl_penalty(self, policy_logits: List[torch.Tensor], reference_logits: List[torch.Tensor]) -> List[torch.Tensor]:
        """计算KL散度惩罚"""
        kl_penalties = []
        
        for p_logits, r_logits in zip(policy_logits, reference_logits):
            # 确保形状一致
            min_len = min(p_logits.shape[0], r_logits.shape[0])
            p_logits = p_logits[:min_len]
            r_logits = r_logits[:min_len]
            
            # 计算KL散度
            kl_div = F.kl_div(
                F.log_softmax(p_logits, dim=-1),
                F.softmax(r_logits, dim=-1),
                reduction='none'
            ).sum(-1)  # [seq_len]
            
            kl_penalties.append(kl_div)
        
        return kl_penalties
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """执行一个PPO训练步骤"""
        queries = batch['queries']
        
        # 生成响应
        responses, policy_logits = self.generate_responses(queries)
        
        # 使用参考模型生成logits
        reference_logits = []
        for query in queries:
            inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                ref_outputs = self.reference_model(**inputs, return_dict=True)
                ref_logits = ref_outputs['logits']
            reference_logits.append(ref_logits[0])  # 移除batch维度
        
        # 计算奖励
        rewards = self.compute_rewards(queries, responses)
        
        # 计算KL惩罚
        kl_penalties = self.compute_kl_penalty(policy_logits, reference_logits)
        
        # PPO训练
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_kl_loss = 0
        
        for epoch in range(self.ppo_epochs):
            # 这里应该实现完整的PPO算法
            # 为了简化，我们使用基本的策略梯度
            
            # 计算策略损失
            policy_loss = -rewards.mean()
            
            # KL惩罚
            kl_loss = torch.cat(kl_penalties).mean()
            
            # 总损失
            loss = policy_loss + 0.1 * kl_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_kl_loss += kl_loss.item()
        
        return {
            'total_loss': total_loss / self.ppo_epochs,
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'kl_loss': total_kl_loss / self.ppo_epochs,
            'avg_reward': rewards.mean().item()
        }
    
    def train(self, dataloader: DataLoader, num_episodes: int = 1000):
        """PPO训练主循环"""
        logger.info(f"Starting RLHF training on {self.device}")
        logger.info(f"Policy model parameters: {self.policy_model.num_parameters:,}")
        
        self.policy_model.train()
        
        for episode in range(num_episodes):
            # 获取批次数据
            batch = next(iter(dataloader))
            
            # 训练一步
            metrics = self.train_step(batch)
            
            self.global_step += 1
            
            # 记录指标
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{num_episodes}")
                logger.info(f"Total Loss: {metrics['total_loss']:.4f}")
                logger.info(f"Policy Loss: {metrics['policy_loss']:.4f}")
                logger.info(f"KL Loss: {metrics['kl_loss']:.4f}")
                logger.info(f"Avg Reward: {metrics['avg_reward']:.4f}")
                
                # 记录到wandb
                try:
                    wandb.log({
                        'episode': episode,
                        'total_loss': metrics['total_loss'],
                        'policy_loss': metrics['policy_loss'],
                        'kl_loss': metrics['kl_loss'],
                        'avg_reward': metrics['avg_reward']
                    })
                except:
                    pass
            
            # 定期保存模型
            if episode % 100 == 0:
                self.save_checkpoint(episode, metrics)
        
        logger.info("RLHF training completed!")
    
    def save_checkpoint(self, episode: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'global_step': self.global_step,
            'policy_model_state_dict': self.policy_model.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        save_path = os.path.join(self.save_dir, f'checkpoint_episode_{episode}.pt')
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")


class RLHFDataManager:
    """RLHF数据管理器"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def create_rlhf_datasets(self, batch_size: int = 4) -> DataLoader:
        """创建RLHF训练数据"""
        # 示例查询数据
        queries = [
            "请为这款新手机写一段产品描述",
            "请回复这个客户的咨询：手机支持快充吗？",
            "请为双十一活动写一段营销文案",
            "请分析这个商品评价的情感",
            "请介绍这款笔记本电脑的特点",
        ]
        
        # 创建数据加载器
        dataset = [{'queries': queries[i:i+batch_size]} for i in range(0, len(queries), batch_size)]
        
        return DataLoader(dataset, batch_size=1, shuffle=True)


def main():
    """主函数"""
    # 配置参数
    config = {
        'model_size': '0.6B',
        'learning_rate': 1e-6,
        'num_episodes': 1000,
        'batch_size': 4,
        'save_dir': './rlhf_checkpoints'
    }
    
    # 加载tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    # 创建模型
    logger.info(f"Creating {config['model_size']} models...")
    policy_model = create_moe_model(config['model_size'])
    reference_model = create_moe_model(config['model_size'])
    
    # 创建奖励模型
    reward_model = RewardModel(policy_model)
    
    # 数据管理器
    logger.info("Preparing RLHF data...")
    data_manager = RLHFDataManager(tokenizer)
    dataloader = data_manager.create_rlhf_datasets(config['batch_size'])
    
    # PPO训练器
    trainer = PPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        learning_rate=config['learning_rate'],
        save_dir=config['save_dir']
    )
    
    # 开始训练
    trainer.train(dataloader, num_episodes=config['num_episodes'])
    
    logger.info("RLHF training completed successfully!")


if __name__ == "__main__":
    main()