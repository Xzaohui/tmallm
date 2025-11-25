"""
MoE模型指令微调脚本
支持多种指令微调格式和训练策略
"""

import torch
import torch.nn as nn
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


class InstructionDataset(Dataset):
    """指令微调数据集"""
    def __init__(self, instructions: List[Dict], tokenizer, max_length: int = 1024):
        self.instructions = instructions
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        item = self.instructions[idx]
        
        # 构建输入文本
        if 'system' in item and item['system']:
            input_text = f"<|system|>\n{item['system']}\n<|user|>\n{item['instruction']}\n<|assistant|>\n"
        else:
            input_text = f"<|user|>\n{item['instruction']}\n<|assistant|>\n"
        
        # 构建完整文本（包含输出）
        full_text = input_text + item['output'] + self.tokenizer.eos_token
        
        # 分词
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # 创建标签 - 只预测assistant部分
        labels = input_ids.clone()
        
        # 找到assistant开始的位置
        assistant_token = self.tokenizer.convert_tokens_to_ids('<|assistant|>')
        assistant_start = torch.where(input_ids == assistant_token)[0]
        if len(assistant_start) > 0:
            assistant_start = assistant_start[0] + 1  # 跳过assistant token
            labels[:assistant_start] = -100  # 不计算前面的损失
        
        # 忽略padding部分
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class InstructionDataManager:
    """指令数据管理器"""
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_alpaca_data(self, limit: int = 50000):
        """加载Alpaca格式数据"""
        try:
            dataset = load_dataset('tatsu-lab/alpaca', split='train')
            instructions = []
            for i, item in enumerate(dataset):
                if i >= limit:
                    break
                instructions.append({
                    'instruction': item['instruction'],
                    'input': item['input'],
                    'output': item['output'],
                    'system': ''
                })
            return instructions
        except Exception as e:
            logger.warning(f"Failed to load Alpaca data: {e}")
            return []
    
    def load_belle_data(self, limit: int = 50000):
        """加载BELLE中文数据"""
        try:
            dataset = load_dataset('BelleGroup/train_0.5M_CN', split='train', streaming=True)
            instructions = []
            for i, item in enumerate(dataset):
                if i >= limit:
                    break
                instructions.append({
                    'instruction': item['instruction'],
                    'input': item['input'],
                    'output': item['output'],
                    'system': '你是一个 helpful AI assistant。'
                })
            return instructions
        except Exception as e:
            logger.warning(f"Failed to load BELLE data: {e}")
            return []
    
    def load_ecommerce_instructions(self):
        """加载电商领域指令数据"""
        return [
            {
                'instruction': '请为这款新手机写一段产品描述',
                'input': '',
                'output': '这款智能手机采用6.7英寸OLED显示屏，配备高通骁龙8处理器，内存12GB+256GB，支持5G网络，拍照效果出色，续航能力强，是您的理想选择。',
                'system': '你是一个专业的电商产品描述撰写专家。'
            },
            {
                'instruction': '请回复这个客户的咨询',
                'input': '客户问：这款手机支持快充吗？',
                'output': '您好！这款手机支持65W超级快充，30分钟可以充电到80%，完全满足您的日常使用需求。同时我们还提供一年质保服务，请您放心购买。',
                'system': '你是一个专业的电商客服。'
            },
            {
                'instruction': '请为双十一活动写一段营销文案',
                'input': '',
                'output': '🎉双十一狂欢节来啦！全场商品5折起，满299减50，满599减120！限时秒杀每日0点开启，iPhone 15仅需4999元！还有机会赢取免单大奖，快来抢购吧！🛒',
                'system': '你是一个创意营销文案专家。'
            },
            {
                'instruction': '请分析这个商品评价的情感',
                'input': '商品收到了，质量一般般，包装也很简陋，感觉不值这个价格。',
                'output': '这条评价的情感倾向是负面的。评价者表达了对产品质量和包装的不满，认为性价比不高。建议关注产品质量控制和包装改进。',
                'system': '你是一个情感分析专家。'
            }
        ]
    
    def create_instruction_datasets(self, batch_size: int = 8) -> Dict[str, DataLoader]:
        """创建指令微调数据集"""
        logger.info("Loading instruction tuning data...")
        
        # 加载各类指令数据
        alpaca_data = self.load_alpaca_data(limit=10000)
        belle_data = self.load_belle_data(limit=10000)
        ecommerce_data = self.load_ecommerce_instructions()
        
        # 合并数据
        all_instructions = alpaca_data + belle_data + ecommerce_data
        np.random.shuffle(all_instructions)
        
        # 分割数据集
        total_size = len(all_instructions)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        train_instructions = all_instructions[:train_size]
        val_instructions = all_instructions[train_size:train_size + val_size]
        test_instructions = all_instructions[train_size + val_size:]
        
        logger.info(f"Dataset sizes - Train: {len(train_instructions)}, Val: {len(val_instructions)}, Test: {len(test_instructions)}")
        
        # 创建数据加载器
        datasets = {}
        for split, instructions in [('train', train_instructions), ('val', val_instructions), ('test', test_instructions)]:
            dataset = InstructionDataset(instructions, self.tokenizer, self.max_length)
            shuffle = (split == 'train')
            datasets[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=2,
                pin_memory=True
            )
        
        return datasets


class InstructionTuner:
    """指令微调器"""
    def __init__(self, model: ChineseEcommerceMoE, tokenizer,
                 learning_rate: float = 2e-5, num_epochs: int = 5,
                 warmup_steps: int = 100, save_dir: str = "./instruction_checkpoints",
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
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # 保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # 初始化wandb
        if use_wandb:
            try:
                wandb.init(
                    project="chinese-ecommerce-moe-instruction",
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
            if self.use_wandb and self.global_step % 50 == 0:
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
        
        return {
            'eval_loss': total_loss / num_steps,
            'lm_loss': total_lm_loss / num_steps,
            'aux_loss': total_aux_loss / num_steps
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
        logger.info(f"Starting instruction tuning on {self.device}")
        logger.info(f"Model parameters: {self.model.num_parameters:,}")
        logger.info(f"Active parameters: {self.model.num_active_parameters:,}")
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # 评估
            eval_metrics = self.evaluate(val_dataloader)
            
            # 记录结果
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
            
            # 保存最佳模型
            is_best = eval_metrics['eval_loss'] < self.best_eval_loss
            if is_best:
                self.best_eval_loss = eval_metrics['eval_loss']
            
            self.save_checkpoint(epoch, {**train_metrics, **eval_metrics}, is_best)
            
            # 记录到wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **train_metrics,
                    **eval_metrics
                })
        
        logger.info("Instruction tuning completed!")
        logger.info(f"Best eval loss: {self.best_eval_loss:.4f}")


def main():
    """主函数"""
    # 配置参数
    config = {
        'model_size': '0.6B',
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'warmup_steps': 100,
        'max_length': 1024,
        'save_dir': './instruction_checkpoints'
    }
    
    # 加载tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    # 添加特殊token
    special_tokens = {
        'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # 创建模型
    logger.info(f"Creating {config['model_size']} model...")
    model = create_moe_model(config['model_size'])
    model.resize_token_embeddings(len(tokenizer))
    
    # 数据管理器
    logger.info("Preparing instruction data...")
    data_manager = InstructionDataManager(tokenizer, config['max_length'])
    dataloaders = data_manager.create_instruction_datasets(config['batch_size'])
    
    # 训练器
    trainer = InstructionTuner(
        model=model,
        tokenizer=tokenizer,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        warmup_steps=config['warmup_steps'],
        save_dir=config['save_dir']
    )
    
    # 开始训练
    trainer.train(dataloaders['train'], dataloaders['val'])
    
    logger.info("Instruction tuning completed successfully!")


if __name__ == "__main__":
    main()