#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
行为克隆 (Behavior Cloning) 训练脚本
使用监督学习训练策略网络模仿专家演示
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
import pickle
from typing import Dict, List


class RobotDataset(Dataset):
    """机器人数据集"""
    
    def __init__(self, data_dir: str, use_images: bool = False):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            use_images: 是否使用图像数据
        """
        self.data_dir = Path(data_dir)
        self.use_images = use_images
        
        # 查找所有episode
        self.episodes = sorted([d for d in self.data_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('episode')])
        
        print(f"找到 {len(self.episodes)} 个episodes")
        
        # 构建样本索引
        self.samples = []
        for ep_dir in tqdm(self.episodes, desc="索引数据集"):
            actions = np.load(ep_dir / "actions.npy")
            states = np.load(ep_dir / "states.npy")
            
            # 每个时间步是一个样本
            for t in range(len(actions)):
                self.samples.append((ep_dir, t))
        
        print(f"总共 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ep_dir, t = self.samples[idx]
        
        # 加载状态和动作
        states = np.load(ep_dir / "states.npy")
        actions = np.load(ep_dir / "actions.npy")
        
        state = states[t]
        action = actions[t]
        
        # 转换为tensor
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.FloatTensor(action)
        
        return state_tensor, action_tensor


class BCPolicy(nn.Module):
    """行为克隆策略网络"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [512, 512, 256]):
        """
        初始化策略网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 构建网络层
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """前向传播"""
        return self.network(state)


class BCTrainer:
    """行为克隆训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据集
        self.train_dataset = RobotDataset(
            data_dir=Path(config['data_dir']) / 'train',
            use_images=config.get('use_images', False)
        )
        
        self.test_dataset = RobotDataset(
            data_dir=Path(config['data_dir']) / 'test',
            use_images=config.get('use_images', False)
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        # 获取数据维度
        sample_state, sample_action = self.train_dataset[0]
        self.state_dim = sample_state.shape[0]
        self.action_dim = sample_action.shape[0]
        
        print(f"状态维度: {self.state_dim}")
        print(f"动作维度: {self.action_dim}")
        
        # 创建模型
        self.model = BCPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.get('hidden_dims', [512, 512, 256])
        ).to(self.device)
        
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练记录
        self.train_losses = []
        self.test_losses = []
        self.best_test_loss = float('inf')
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="训练")
        for states, actions in pbar:
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            # 前向传播
            pred_actions = self.model(states)
            loss = self.criterion(pred_actions, actions)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def test_epoch(self) -> float:
        """测试一个epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for states, actions in tqdm(self.test_loader, desc="测试"):
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                pred_actions = self.model(states)
                loss = self.criterion(pred_actions, actions)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, test_loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'test_loss': test_loss,
            'config': self.config
        }
        
        # 保存最新的检查点
        checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: {best_path}")
    
    def train(self):
        """执行完整的训练过程"""
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 测试
            test_loss = self.test_epoch()
            self.test_losses.append(test_loss)
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印统计信息
            lr = self.optimizer.param_groups[0]['lr']
            print(f"训练损失: {train_loss:.6f}")
            print(f"测试损失: {test_loss:.6f}")
            print(f"学习率: {lr:.2e}")
            
            # 保存检查点
            is_best = test_loss < self.best_test_loss
            if is_best:
                self.best_test_loss = test_loss
            
            self.save_checkpoint(epoch, test_loss, is_best)
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'best_test_loss': self.best_test_loss
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "="*60)
        print("训练完成!")
        print(f"最佳测试损失: {self.best_test_loss:.6f}")
        print(f"模型保存在: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="行为克隆训练")
    parser.add_argument("--config", type=str, required=True,
                       help="配置文件路径")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="数据目录")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="学习率")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="训练设备")
    
    args = parser.parse_args()
    
    # 加载配置文件
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # 命令行参数覆盖配置文件
    config.update({
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'device': args.device
    })
    
    # 创建训练器并训练
    trainer = BCTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
