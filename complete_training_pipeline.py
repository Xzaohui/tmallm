"""
完整训练流程脚本
整合预训练、指令微调、强化学习对齐等所有训练阶段
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_moe_model import create_moe_model, load_tokenizer


class TrainingPipeline:
    """完整训练流程管理器"""
    
    def __init__(self, model_size: str = "0.6B", base_dir: str = "./training_pipeline"):
        self.model_size = model_size
        self.base_dir = base_dir
        
        # 创建目录结构
        self.dirs = {
            'pretrain': os.path.join(base_dir, 'pretrain'),
            'instruction': os.path.join(base_dir, 'instruction'),
            'rlhf': os.path.join(base_dir, 'rlhf'),
            'final': os.path.join(base_dir, 'final_model')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # 训练状态
        self.training_state = {
            'current_stage': 'not_started',
            'completed_stages': [],
            'best_metrics': {}
        }
        
        self.load_training_state()
    
    def load_training_state(self):
        """加载训练状态"""
        state_file = os.path.join(self.base_dir, 'training_state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                self.training_state = json.load(f)
    
    def save_training_state(self):
        """保存训练状态"""
        state_file = os.path.join(self.base_dir, 'training_state.json')
        with open(state_file, 'w') as f:
            json.dump(self.training_state, f, indent=2)
    
    def stage_pretrain(self, **kwargs):
        """预训练阶段"""
        logger.info("Starting pretraining stage...")
        
        try:
            # 导入预训练模块
            from pretrain_moe import MoEPreTrainer, DataManager
            
            # 加载模型和tokenizer
            model = create_moe_model(self.model_size)
            tokenizer = load_tokenizer()
            
            # 数据管理器
            data_manager = DataManager(tokenizer, max_length=kwargs.get('max_length', 512))
            dataloaders = data_manager.create_pretrain_datasets(kwargs.get('batch_size', 16))
            
            # 训练器
            trainer = MoEPreTrainer(
                model=model,
                tokenizer=tokenizer,
                learning_rate=kwargs.get('learning_rate', 5e-4),
                num_epochs=kwargs.get('num_epochs', 10),
                warmup_steps=kwargs.get('warmup_steps', 1000),
                save_dir=self.dirs['pretrain']
            )
            
            # 开始训练
            trainer.train(dataloaders['train'], dataloaders['val'])
            
            # 保存最佳模型路径
            self.training_state['pretrain_best_model'] = os.path.join(self.dirs['pretrain'], 'best_model.pt')
            self.training_state['completed_stages'].append('pretrain')
            self.training_state['current_stage'] = 'pretrain_completed'
            self.save_training_state()
            
            logger.info("Pretraining stage completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pretraining stage failed: {e}")
            return False
    
    def stage_instruction_tune(self, **kwargs):
        """指令微调阶段"""
        logger.info("Starting instruction tuning stage...")
        
        try:
            # 导入指令微调模块
            from instruction_tune_moe import InstructionTuner, InstructionDataManager
            
            # 加载预训练模型
            if 'pretrain_best_model' in self.training_state:
                logger.info(f"Loading pre-trained model from {self.training_state['pretrain_best_model']}")
                model = create_moe_model(self.model_size)
                checkpoint = torch.load(self.training_state['pretrain_best_model'], map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.warning("No pre-trained model found, starting from scratch")
                model = create_moe_model(self.model_size)
            
            tokenizer = load_tokenizer()
            
            # 添加特殊token
            special_tokens = {
                'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>']
            }
            tokenizer.add_special_tokens(special_tokens)
            model.resize_token_embeddings(len(tokenizer))
            
            # 数据管理器
            data_manager = InstructionDataManager(tokenizer, max_length=kwargs.get('max_length', 1024))
            dataloaders = data_manager.create_instruction_datasets(kwargs.get('batch_size', 8))
            
            # 训练器
            trainer = InstructionTuner(
                model=model,
                tokenizer=tokenizer,
                learning_rate=kwargs.get('learning_rate', 2e-5),
                num_epochs=kwargs.get('num_epochs', 5),
                warmup_steps=kwargs.get('warmup_steps', 100),
                save_dir=self.dirs['instruction']
            )
            
            # 开始训练
            trainer.train(dataloaders['train'], dataloaders['val'])
            
            # 保存最佳模型路径
            self.training_state['instruction_best_model'] = os.path.join(self.dirs['instruction'], 'best_model.pt')
            self.training_state['completed_stages'].append('instruction')
            self.training_state['current_stage'] = 'instruction_completed'
            self.save_training_state()
            
            logger.info("Instruction tuning stage completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Instruction tuning stage failed: {e}")
            return False
    
    def stage_rlhf_align(self, **kwargs):
        """RLHF对齐阶段"""
        logger.info("Starting RLHF alignment stage...")
        
        try:
            # 导入RLHF模块
            from rlhf_align_moe import PPOTrainer, RewardModel, RLHFDataManager
            
            # 加载指令微调模型作为策略模型
            if 'instruction_best_model' in self.training_state:
                logger.info(f"Loading instruction-tuned model from {self.training_state['instruction_best_model']}")
                policy_model = create_moe_model(self.model_size)
                checkpoint = torch.load(self.training_state['instruction_best_model'], map_location='cpu')
                policy_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.warning("No instruction-tuned model found, starting from scratch")
                policy_model = create_moe_model(self.model_size)
            
            # 创建参考模型（复制策略模型）
            reference_model = create_moE_model(self.model_size)
            reference_model.load_state_dict(policy_model.state_dict())
            
            # 创建奖励模型
            reward_model = RewardModel(policy_model)
            
            tokenizer = load_tokenizer()
            
            # 数据管理器
            data_manager = RLHFDataManager(tokenizer)
            dataloader = data_manager.create_rlhf_datasets(kwargs.get('batch_size', 4))
            
            # 训练器
            trainer = PPOTrainer(
                policy_model=policy_model,
                reference_model=reference_model,
                reward_model=reward_model,
                tokenizer=tokenizer,
                learning_rate=kwargs.get('learning_rate', 1e-6),
                save_dir=self.dirs['rlhf']
            )
            
            # 开始训练
            trainer.train(dataloader, num_episodes=kwargs.get('num_episodes', 1000))
            
            # 保存最终模型
            final_model_path = os.path.join(self.dirs['final'], 'final_model.pt')
            torch.save({
                'model_state_dict': policy_model.state_dict(),
                'tokenizer': tokenizer,
                'config': policy_model.config.__dict__
            }, final_model_path)
            
            self.training_state['final_model'] = final_model_path
            self.training_state['completed_stages'].append('rlhf')
            self.training_state['current_stage'] = 'completed'
            self.save_training_state()
            
            logger.info("RLHF alignment stage completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"RLHF alignment stage failed: {e}")
            return False
    
    def run_full_pipeline(self, config: Dict):
        """运行完整训练流程"""
        logger.info("Starting full training pipeline...")
        logger.info(f"Model size: {self.model_size}")
        logger.info(f"Base directory: {self.base_dir}")
        
        # 阶段1: 预训练
        if 'pretrain' not in self.training_state['completed_stages']:
            logger.info("=== Stage 1: Pretraining ===")
            pretrain_config = config.get('pretrain', {})
            if not self.stage_pretrain(**pretrain_config):
                logger.error("Pretraining failed, stopping pipeline")
                return False
        else:
            logger.info("Pretraining already completed, skipping...")
        
        # 阶段2: 指令微调
        if 'instruction' not in self.training_state['completed_stages']:
            logger.info("=== Stage 2: Instruction Tuning ===")
            instruction_config = config.get('instruction', {})
            if not self.stage_instruction_tune(**instruction_config):
                logger.error("Instruction tuning failed, stopping pipeline")
                return False
        else:
            logger.info("Instruction tuning already completed, skipping...")
        
        # 阶段3: RLHF对齐
        if 'rlhf' not in self.training_state['completed_stages']:
            logger.info("=== Stage 3: RLHF Alignment ===")
            rlhf_config = config.get('rlhf', {})
            if not self.stage_rlhf_align(**rlhf_config):
                logger.error("RLHF alignment failed, stopping pipeline")
                return False
        else:
            logger.info("RLHF alignment already completed, skipping...")
        
        logger.info("Full training pipeline completed successfully!")
        logger.info(f"Final model saved at: {self.training_state.get('final_model', 'Not found')}")
        
        return True
    
    def get_training_summary(self):
        """获取训练摘要"""
        return {
            'model_size': self.model_size,
            'completed_stages': self.training_state['completed_stages'],
            'current_stage': self.training_state['current_stage'],
            'final_model': self.training_state.get('final_model'),
            'best_metrics': self.training_state.get('best_metrics', {})
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MoE Model Complete Training Pipeline')
    parser.add_argument('--model_size', type=str, default='0.6B', choices=['0.1B', '0.6B'],
                        help='Model size to train')
    parser.add_argument('--base_dir', type=str, default='./training_pipeline',
                        help='Base directory for training')
    parser.add_argument('--config', type=str, default='training_config.json',
                        help='Training configuration file')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'pretrain', 'instruction', 'rlhf'],
                        help='Training stage to run')
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # 默认配置
        config = {
            "pretrain": {
                "batch_size": 16,
                "learning_rate": 5e-4,
                "num_epochs": 10,
                "warmup_steps": 1000,
                "max_length": 512
            },
            "instruction": {
                "batch_size": 8,
                "learning_rate": 2e-5,
                "num_epochs": 5,
                "warmup_steps": 100,
                "max_length": 1024
            },
            "rlhf": {
                "batch_size": 4,
                "learning_rate": 1e-6,
                "num_episodes": 1000
            }
        }
        
        # 保存默认配置
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
    
    # 创建训练流程
    pipeline = TrainingPipeline(model_size=args.model_size, base_dir=args.base_dir)
    
    # 运行指定阶段
    if args.stage == 'all':
        success = pipeline.run_full_pipeline(config)
    elif args.stage == 'pretrain':
        success = pipeline.stage_pretrain(**config.get('pretrain', {}))
    elif args.stage == 'instruction':
        success = pipeline.stage_instruction_tune(**config.get('instruction', {}))
    elif args.stage == 'rlhf':
        success = pipeline.stage_rlhf_align(**config.get('rlhf', {}))
    
    # 输出训练摘要
    summary = pipeline.get_training_summary()
    logger.info("Training Summary:")
    logger.info(json.dumps(summary, indent=2))
    
    if success:
        logger.info("Training pipeline completed successfully!")
    else:
        logger.error("Training pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()