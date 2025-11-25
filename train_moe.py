"""
中文电商MoE模型训练脚本
包含数据加载、训练循环、评估和保存功能
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import json
import os
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional
import numpy as np
from moe_model import create_moe_model, ChineseEcommerceMoE


class EcommerceDataset(Dataset):
    """中文电商数据集"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
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


class EcommerceDataManager:
    """电商数据管理器"""
    def __init__(self, tokenizer, data_dir: str = "./data"):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_length = 512
        
    def load_sample_data(self) -> Dict[str, List[str]]:
        """加载示例电商数据"""
        # 示例电商文本数据
        sample_texts = [
            # 产品描述
            "这款智能手机采用6.7英寸OLED显示屏，配备高通骁龙8处理器，内存12GB+256GB，支持5G网络，拍照效果出色。",
            "时尚女装连衣裙，采用优质棉麻面料，舒适透气，适合夏季穿着，多种颜色可选，尺码齐全。",
            "高端笔记本电脑，搭载最新处理器，性能强劲，适合办公和游戏，续航时间长，轻薄便携。",
            
            # 用户评价
            "商品质量很好，物流速度快，客服态度也不错，整体购物体验非常满意，会再次购买。",
            "价格实惠，性价比高，但是包装有点简陋，希望商家能改进一下包装质量。",
            "收到货了，质量不错，跟描述的一样，朋友都说好看，推荐购买！",
            
            # 客服对话
            "客服：您好，欢迎光临我们的店铺，请问有什么可以帮助您的吗？",
            "顾客：我想咨询一下这款产品的详细信息，可以介绍一下吗？",
            "客服：当然可以，这款产品是我们店铺的畅销商品，质量有保障，支持7天无理由退换。",
            
            # 营销文案
            "双十一大促销，全场商品5折起，还有满减优惠，限时抢购，先到先得！",
            "新品上市，前100名购买者享受特价优惠，赠送精美礼品，数量有限，速来抢购！",
            "年终大促，品牌特卖，正品保障，假一赔十，支持货到付款和7天无理由退货。",
        ]
        
        # 扩展数据集
        extended_texts = []
        for text in sample_texts:
            extended_texts.append(text)
            # 添加一些变体
            extended_texts.append(f"【电商推荐】{text}")
            extended_texts.append(f"【限时优惠】{text}")
        
        # 分割训练、验证、测试集
        total_size = len(extended_texts)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        np.random.shuffle(extended_texts)
        
        return {
            'train': extended_texts[:train_size],
            'val': extended_texts[train_size:train_size + val_size],
            'test': extended_texts[train_size + val_size:]
        }
    
    def create_dataloaders(self, batch_size: int = 16) -> Dict[str, DataLoader]:
        """创建数据加载器"""
        data = self.load_sample_data()
        
        dataloaders = {}
        for split, texts in data.items():
            dataset = EcommerceDataset(texts, self.tokenizer, self.max_length)
            shuffle = (split == 'train')
            dataloaders[split] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=2
            )
        
        return dataloaders


class MoETrainer:
    """MoE模型训练器"""
    def __init__(self, model: ChineseEcommerceMoE, tokenizer, 
                 learning_rate: float = 5e-4, num_epochs: int = 10,
                 warmup_steps: int = 1000, save_dir: str = "./checkpoints"):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.save_dir = save_dir
        
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
        
    def compute_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """计算辅助损失用于负载均衡"""
        # 路由器z-loss
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        
        # 负载均衡损失
        router_prob = F.softmax(router_logits, dim=-1)
        aux_loss = router_z_loss * 1e-3  # 缩放因子
        
        return aux_loss
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_steps = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, labels=labels)
            if isinstance(outputs, tuple):
                logits, loss = outputs
            else:
                loss = outputs
            
            # 总损失
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (step + 1):.4f}'
            })
            
            # 记录到wandb
            if wandb.run:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        return {'train_loss': total_loss / total_steps}
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_steps = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, labels=labels)
                if isinstance(outputs, tuple):
                    logits, loss = outputs
                else:
                    loss = outputs
                
                total_loss += loss.item()
        
        avg_loss = total_loss / total_steps
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return {
            'eval_loss': avg_loss,
            'perplexity': perplexity.item()
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'model_config': {
                'vocab_size': self.model.embed_tokens.num_embeddings,
                'dim': self.model.dim,
                'num_layers': self.model.num_layers,
                'max_seq_len': self.model.max_seq_len
            }
        }
        
        save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
        
        # 也保存为HuggingFace格式
        hf_save_path = os.path.join(self.save_dir, f'hf_model_epoch_{epoch}')
        os.makedirs(hf_save_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), os.path.join(hf_save_path, 'pytorch_model.bin'))
        
        # 保存配置
        config = {
            'model_type': 'chinese_ecommerce_moe',
            'vocab_size': self.model.embed_tokens.num_embeddings,
            'dim': self.model.dim,
            'num_layers': self.model.num_layers,
            'max_seq_len': self.model.max_seq_len,
            'num_experts': 8,
            'top_k': 2
        }
        
        with open(os.path.join(hf_save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(hf_save_path)
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """完整训练流程"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.num_parameters:,}")
        print(f"Active parameters: {self.model.num_active_parameters:,}")
        
        best_perplexity = float('inf')
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # 评估
            eval_metrics = self.evaluate(val_dataloader)
            
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
            print(f"Perplexity: {eval_metrics['perplexity']:.4f}")
            
            # 保存最佳模型
            if eval_metrics['perplexity'] < best_perplexity:
                best_perplexity = eval_metrics['perplexity']
                self.save_checkpoint(epoch, {**train_metrics, **eval_metrics})
            
            # 记录到wandb
            if wandb.run:
                wandb.log({
                    'epoch': epoch,
                    **train_metrics,
                    **eval_metrics
                })


def main():
    """主函数"""
    # 初始化wandb
    try:
        wandb.init(
            project="chinese-ecommerce-moe",
            config={
                "model_size": "0.1B",
                "learning_rate": 5e-4,
                "batch_size": 16,
                "num_epochs": 10,
                "warmup_steps": 1000
            }
        )
    except:
        print("Wandb not available, continuing without logging...")
    
    # 加载tokenizer
    print("Loading tokenizer...")
    from moe_model import load_tokenizer
    tokenizer = load_tokenizer()
    
    # 创建模型
    print("Creating model...")
    model = create_moe_model("0.1B")
    
    # 数据管理器
    print("Preparing data...")
    data_manager = EcommerceDataManager(tokenizer)
    dataloaders = data_manager.create_dataloaders(batch_size=16)
    
    # 训练器
    trainer = MoETrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=5e-4,
        num_epochs=10,
        warmup_steps=1000
    )
    
    # 开始训练
    trainer.train(dataloaders['train'], dataloaders['val'])
    
    print("Training completed!")


if __name__ == "__main__":
    main()