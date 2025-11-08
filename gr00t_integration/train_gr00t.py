# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
基于Isaac-GR00T的微调训练脚本
用于宇树机器人双臂5指灵巧手抓取任务
"""

import os
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Optional
import time

# 导入GR00T模块
import sys
sys.path.append(str(Path(__file__).parent.parent))
from gr00t_integration.config import GR00TConfig, load_config
from gr00t_integration.gr00t_model import GR00TModelWrapper, load_gr00t_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnitreeDataset(Dataset):
    """宇树机器人数据集"""
    
    def __init__(
        self,
        data_dir: str,
        config: GR00TConfig,
        split: str = "train",
        normalize: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            config: GR00TConfig配置对象
            split: 数据集分割 ("train" 或 "test")
            normalize: 是否标准化数据
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.normalize = normalize
        
        # 查找所有episode
        episodes_dir = self.data_dir / split
        if not episodes_dir.exists():
            # 如果没有分割目录，从根目录查找
            episodes_dir = self.data_dir
        
        self.episodes = sorted([
            d for d in episodes_dir.iterdir()
            if d.is_dir() and (d.name.startswith('episode') or d.name.isdigit())
        ])
        
        logger.info(f"找到 {len(self.episodes)} 个episodes ({split})")
        
        # 构建样本索引
        self.samples = []
        for ep_dir in tqdm(self.episodes, desc=f"索引{split}数据集"):
            try:
                # 加载数据文件
                actions_file = ep_dir / "actions.npy"
                states_file = ep_dir / "states.npy"
                images_file = ep_dir / "images.npy"
                
                if not actions_file.exists() or not states_file.exists():
                    logger.warning(f"跳过episode {ep_dir.name}: 缺少必要文件")
                    continue
                
                actions = np.load(actions_file)
                states = np.load(states_file)
                
                # 加载图像（如果存在）
                images = None
                if images_file.exists():
                    images = np.load(images_file)
                
                # 每个时间步是一个样本
                for t in range(len(actions)):
                    self.samples.append({
                        'episode': ep_dir,
                        'timestep': t,
                        'has_images': images is not None
                    })
            except Exception as e:
                logger.warning(f"加载episode {ep_dir.name}失败: {e}")
                continue
        
        logger.info(f"总共 {len(self.samples)} 个样本 ({split})")
        
        # 计算统计信息（用于标准化）
        if self.normalize:
            self._compute_statistics()
    
    def _compute_statistics(self):
        """计算数据统计信息"""
        logger.info("计算数据统计信息...")
        
        all_states = []
        all_actions = []
        
        # 采样计算统计信息（避免内存过大）
        sample_indices = np.linspace(0, len(self.samples) - 1, min(1000, len(self.samples)), dtype=int)
        
        for idx in tqdm(sample_indices, desc="采样数据"):
            sample = self.samples[idx]
            ep_dir = sample['episode']
            t = sample['timestep']
            
            states = np.load(ep_dir / "states.npy")
            actions = np.load(ep_dir / "actions.npy")
            
            all_states.append(states[t])
            all_actions.append(actions[t])
        
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        
        self.state_mean = np.mean(all_states, axis=0)
        self.state_std = np.std(all_states, axis=0) + 1e-8
        self.action_mean = np.mean(all_actions, axis=0)
        self.action_std = np.std(all_actions, axis=0) + 1e-8
        
        logger.info(f"状态均值范围: [{self.state_mean.min():.3f}, {self.state_mean.max():.3f}]")
        logger.info(f"动作均值范围: [{self.action_mean.min():.3f}, {self.action_mean.max():.3f}]")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        ep_dir = sample['episode']
        t = sample['timestep']
        
        # 加载状态和动作
        states = np.load(ep_dir / "states.npy")
        actions = np.load(ep_dir / "actions.npy")
        
        state = states[t].astype(np.float32)
        action = actions[t].astype(np.float32)
        
        # 标准化
        if self.normalize:
            state = (state - self.state_mean) / self.state_std
            action = (action - self.action_mean) / self.action_std
        
        # 转换为tensor
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.FloatTensor(action)
        
        # 加载图像（如果存在）
        images_tensor = None
        if sample['has_images']:
            images = np.load(ep_dir / "images.npy")
            if t < len(images):
                image = images[t].astype(np.float32)
                images_tensor = torch.FloatTensor(image)
        
        return {
            'state': state_tensor,
            'action': action_tensor,
            'images': images_tensor
        }


class GR00TTrainer:
    """GR00T模型训练器"""
    
    def __init__(self, config: GR00TConfig):
        """
        初始化训练器
        
        Args:
            config: GR00TConfig配置对象
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config.get("training.output_dir", "./outputs/gr00t_training"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config.save(self.output_dir / "config.yaml")
        
        # 加载数据集
        data_dir = config.get("data.data_dir", "./data/teleoperate")
        normalize = config.get("data.normalize", True)
        
        self.train_dataset = UnitreeDataset(
            data_dir=data_dir,
            config=config,
            split="train",
            normalize=normalize
        )
        
        self.test_dataset = UnitreeDataset(
            data_dir=data_dir,
            config=config,
            split="test",
            normalize=normalize
        )
        
        # 创建数据加载器
        batch_size = config.get("training.batch_size", 32)
        num_workers = config.get("data.num_workers", 4)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # 加载模型
        checkpoint_path = config.get("gr00t.checkpoint_path", None)
        self.model = load_gr00t_model(config, checkpoint_path, device=str(self.device))
        
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 优化器
        learning_rate = config.get("training.learning_rate", 1e-5)
        weight_decay = config.get("training.weight_decay", 1e-4)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        epochs = config.get("training.epochs", 50)
        warmup_epochs = config.get("training.warmup_epochs", 5)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练记录
        self.train_losses = []
        self.test_losses = []
        self.best_test_loss = float('inf')
        self.current_epoch = 0
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [训练]")
        for batch in pbar:
            states = batch['state'].to(self.device)
            actions = batch['action'].to(self.device)
            images = batch['images']
            if images is not None:
                images = images.to(self.device)
            
            # 前向传播
            pred_actions = self.model(states, images)
            loss = self.criterion(pred_actions, actions)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            grad_clip = self.config.get("training.gradient_clip", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def test_epoch(self) -> float:
        """测试一个epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"Epoch {self.current_epoch+1} [测试]")
            for batch in pbar:
                states = batch['state'].to(self.device)
                actions = batch['action'].to(self.device)
                images = batch['images']
                if images is not None:
                    images = images.to(self.device)
                
                pred_actions = self.model(states, images)
                loss = self.criterion(pred_actions, actions)
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def save_checkpoint(self, epoch: int, test_loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'test_loss': test_loss,
            'train_loss': self.train_losses[-1] if self.train_losses else 0,
            'config': self.config.config
        }
        
        # 保存最新的检查点
        checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")
    
    def train(self):
        """执行完整的训练过程"""
        logger.info("\n" + "="*60)
        logger.info("开始GR00T微调训练")
        logger.info("="*60)
        
        epochs = self.config.get("training.epochs", 50)
        save_interval = self.config.get("training.save_interval", 10)
        eval_interval = self.config.get("training.eval_interval", 5)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 更新学习率
            self.scheduler.step()
            
            # 测试
            if (epoch + 1) % eval_interval == 0:
                test_loss = self.test_epoch()
                self.test_losses.append(test_loss)
            else:
                test_loss = None
            
            elapsed_time = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"训练损失: {train_loss:.6f}")
            if test_loss is not None:
                logger.info(f"测试损失: {test_loss:.6f}")
            logger.info(f"学习率: {lr:.2e}")
            logger.info(f"耗时: {elapsed_time:.2f}秒")
            
            # 保存检查点
            is_best = False
            if test_loss is not None:
                is_best = test_loss < self.best_test_loss
                if is_best:
                    self.best_test_loss = test_loss
            
            if (epoch + 1) % save_interval == 0 or is_best:
                self.save_checkpoint(epoch + 1, test_loss or train_loss, is_best)
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'best_test_loss': self.best_test_loss
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("训练完成!")
        logger.info(f"最佳测试损失: {self.best_test_loss:.6f}")
        logger.info(f"模型保存在: {self.output_dir}")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="GR00T模型微调训练")
    parser.add_argument("--config", type=str, required=True,
                      help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default=None,
                      help="数据目录（覆盖配置）")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="输出目录（覆盖配置）")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="预训练检查点路径")
    parser.add_argument("--device", type=str, default="cuda",
                      choices=["cuda", "cpu"],
                      help="训练设备")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.data_dir:
        config.config["data"]["data_dir"] = args.data_dir
    if args.output_dir:
        config.config["training"]["output_dir"] = args.output_dir
    if args.checkpoint:
        config.config["gr00t"]["checkpoint_path"] = args.checkpoint
    
    # 创建训练器并训练
    trainer = GR00TTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
