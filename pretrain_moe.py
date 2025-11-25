"""
MoE模型预训练脚本
包含完整的预训练流程、数据处理、训练监控等功能
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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

from enhanced_moe_model import create_moe_model, ChineseEcommerceMoE, ModelConfig, load_tokenizer


class PretrainDataset(Dataset):
    """预训练数据集"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 分词处理
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # 语言模型训练：预测下一个token
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # 忽略padding部分
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class DataManager:
    """数据管理器 - 支持多种数据源"""
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_wikipedia_data(self, split: str = 'train', limit: int = 100000):
        """加载维基百科数据"""
        try:
            dataset = load_dataset('wikipedia', '20220301.zh', split=split, streaming=True)
            texts = []
            for i, item in enumerate(dataset):
                if i >= limit:
                    break
                texts.append(item['text'][:self.max_length*4])  # 截断文本
            return texts
        except Exception as e:
            logger.warning(f"Failed to load Wikipedia data: {e}")
            return self._load_sample_data()
    
    def load_news_data(self, limit: int = 50000):
        """加载新闻数据"""
        try:
            dataset = load_dataset('lcw99/chinese-news', split='train', streaming=True)
            texts = []
            for i, item in enumerate(dataset):
                if i >= limit:
                    break
                texts.append(item['title'] + ' ' + item['content'])
            return texts
        except Exception as e:
            logger.warning(f"Failed to load news data: {e}")
            return []
    
    def load_ecommerce_data(self):
        """加载电商示例数据"""
        return [
            # 产品描述
            "这款智能手机采用6.7英寸OLED显示屏，配备高通骁龙8处理器，内存12GB+256GB，支持5G网络。",
            "时尚女装连衣裙，采用优质棉麻面料，舒适透气，适合夏季穿着，多种颜色可选。",
            "高端笔记本电脑，搭载最新处理器，性能强劲，适合办公和游戏，续航时间长。",
            
            # 用户评价
            "商品质量很好，物流速度快，客服态度也不错，整体购物体验非常满意。",
            "价格实惠，性价比高，包装有点简陋但产品没问题，会再次购买。",
            "收到货了质量不错，跟描述的一样，朋友都说好看，推荐购买！",
            
            # 客服对话
            "客服：您好，欢迎光临我们的店铺，请问有什么可以帮助您的吗？",
            "顾客：我想咨询一下这款产品的详细信息。",
            "客服：这款产品是我们店铺的畅销商品，质量有保障，支持7天无理由退换。",
            
            # 营销文案
            "双十一大促销，全场商品5折起，还有满减优惠，限时抢购！",
            "新品上市，前100名购买者享受特价优惠，赠送精美礼品。",
        ]
    
    def _load_sample_data(self):
        """加载示例数据"""
        return [
            "这是一个示例文本，用于模型预训练。",
            "中文自然语言处理是人工智能的重要分支。",
            "深度学习技术在近年来取得了巨大进展。",
            "预训练语言模型在各种任务上表现出色。",
        ]
    
    def create_pretrain_datasets(self, batch_size: int = 16) -> Dict[str, DataLoader]:
        """创建预训练数据集"""
        logger.info("Loading pretraining data...")
        
        # 加载各类数据
        wiki_data = self.load_wikipedia_data(limit=20000)
        news_data = self.load_news_data(limit=10000)
        ecommerce_data = self.load_ecommerce_data()
        
        # 合并数据
        all_data = wiki_data + news_data + ecommerce_data
        np.random.shuffle(all_data)
        
        # 分割数据集
        total_size = len(all_data)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        train_texts = all_data[:train_size]
        val_texts = all_data[train_size:train_size + val_size]
        test_texts = all_data[train_size + val_size:]
        
        logger.info(f"Dataset sizes - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # 创建数据加载器
        datasets = {}
        for split, texts in [('train', train_texts), ('val', val_texts), ('test', test_texts)]:
            dataset = PretrainDataset(texts, self.tokenizer, self.max_length)
            shuffle = (split == 'train')
            datasets[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True
            )
        
        return datasets


class MoEPreTrainer:
    """MoE预训练器"""
    def __init__(self, model: ChineseEcommerceMoE, tokenizer, 
                 learning_rate: float = 5e-4, num_epochs: int = 10,
                 warmup_steps: int = 1000, save_dir: str = "./pretrain_checkpoints",
                 use_wandb: bool = True):
        
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # 保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练状态
        self.global_step = 0
        self.best_perplexity = float('inf')
        
        # 初始化wandb
        if use_wandb:
            try:
                wandb.init(
                    project="chinese-ecommerce-moe-pretrain",
                    config={
                        "model_size": "0.6B",
                        "learning_rate": learning_rate,
                        "num_epochs": num_epochs,
                        "warmup_steps": warmup_steps,
                        "total_params": model.num_parameters,
                        "active_params": model.num_active_parameters
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_lm_loss = 0
        total_aux_loss = 0
        num_steps = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            loss = outputs['loss']
            lm_loss = loss - outputs['aux_loss']  # 近似计算
            aux_loss = outputs['aux_loss']
            
            # 记录损失
            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_aux_loss += aux_loss.item()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lm_loss': f'{lm_loss.item():.4f}',
                'aux_loss': f'{aux_loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 记录到wandb
            if self.use_wandb and self.global_step % 100 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'lm_loss': lm_loss.item(),
                    'aux_loss': aux_loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })
        
        return {
            'train_loss': total_loss / num_steps,
            'lm_loss': total_lm_loss / num_steps,
            'aux_loss': total_aux_loss / num_steps
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_lm_loss = 0
        total_aux_loss = 0
        num_steps = len(dataloader)
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
                
                loss = outputs['loss']
                lm_loss = loss - outputs['aux_loss']
                aux_loss = outputs['aux_loss']
                
                total_loss += loss.item()
                total_lm_loss += lm_loss.item()
                total_aux_loss += aux_loss.item()
                
                progress_bar.set_postfix({
                    'eval_loss': f'{loss.item():.4f}'
                })
        
        avg_loss = total_loss / num_steps
        avg_lm_loss = total_lm_loss / num_steps
        avg_aux_loss = total_aux_loss / num_steps
        perplexity = torch.exp(torch.tensor(avg_lm_loss))
        
        return {
            'eval_loss': avg_loss,
            'lm_loss': avg_lm_loss,
            'aux_loss': avg_aux_loss,
            'perplexity': perplexity.item()
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.model.config.__dict__
        }
        
        # 保存最新检查点
        save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
        
        # 保存为HuggingFace格式
        hf_save_path = os.path.join(self.save_dir, f'hf_model_epoch_{epoch}')
        os.makedirs(hf_save_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), os.path.join(hf_save_path, 'pytorch_model.bin'))
        
        # 保存配置
        with open(os.path.join(hf_save_path, 'config.json'), 'w') as f:
            json.dump(self.model.config.__dict__, f, indent=2)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(hf_save_path)
        logger.info(f"HuggingFace model saved: {hf_save_path}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """完整训练流程"""
        logger.info(f"Starting pretraining on {self.device}")
        logger.info(f"Model parameters: {self.model.num_parameters:,}")
        logger.info(f"Active parameters: {self.model.num_active_parameters:,}")
        logger.info(f"Parameter efficiency: {self.model.num_active_parameters/self.model.num_parameters:.2%}")
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # 评估
            eval_metrics = self.evaluate(val_dataloader)
            
            # 记录结果
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
            logger.info(f"Perplexity: {eval_metrics['perplexity']:.4f}")
            
            # 保存最佳模型
            is_best = eval_metrics['perplexity'] < self.best_perplexity
            if is_best:
                self.best_perplexity = eval_metrics['perplexity']
            
            self.save_checkpoint(epoch, {**train_metrics, **eval_metrics}, is_best)
            
            # 记录到wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **train_metrics,
                    **eval_metrics
                })
        
        logger.info("Pretraining completed!")
        logger.info(f"Best perplexity: {self.best_perplexity:.4f}")


def main():
    """主函数"""
    # 配置参数
    config = {
        'model_size': '0.6B',
        'batch_size': 16,
        'learning_rate': 5e-4,
        'num_epochs': 10,
        'warmup_steps': 1000,
        'max_length': 512,
        'save_dir': './pretrain_checkpoints'
    }
    
    # 加载tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    # 创建模型
    logger.info(f"Creating {config['model_size']} model...")
    model = create_moe_model(config['model_size'])
    
    # 数据管理器
    logger.info("Preparing data...")
    data_manager = DataManager(tokenizer, config['max_length'])
    dataloaders = data_manager.create_pretrain_datasets(config['batch_size'])
    
    # 训练器
    trainer = MoEPreTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        warmup_steps=config['warmup_steps'],
        save_dir=config['save_dir']
    )
    
    # 开始训练
    trainer.train(dataloaders['train'], dataloaders['val'])
    
    logger.info("Pretraining completed successfully!")


if __name__ == "__main__":
    main()