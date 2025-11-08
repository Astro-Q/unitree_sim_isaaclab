#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练和微调模块
支持基于采集数据的模型训练和微调
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import cv2

class GraspingDataset(Dataset):
    """抓取数据集"""
    
    def __init__(self, data_path: str, transform=None):
        """
        初始化数据集
        
        Args:
            data_path: 数据路径（包含observations.npy和actions.npy）
            transform: 数据变换（可选）
        """
        self.data_path = Path(data_path)
        self.transform = transform
        
        # 加载数据
        obs_path = self.data_path / "observations.npy"
        act_path = self.data_path / "actions.npy"
        
        if not obs_path.exists() or not act_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {obs_path} 或 {act_path}")
        
        self.observations = np.load(obs_path, allow_pickle=True)
        self.actions = np.load(act_path, allow_pickle=True)
        
        print(f"加载数据集: {len(self.observations)} 个样本")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = self.observations[idx]
        act = self.actions[idx]
        
        # 处理观察数据
        images = []
        for key in sorted(obs.keys()):
            if key.startswith('image_'):
                img = obs[key]
                if isinstance(img, np.ndarray):
                    # 转换为tensor并归一化
                    img_tensor = torch.from_numpy(img).float()
                    if len(img_tensor.shape) == 3:
                        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                    img_tensor = img_tensor / 255.0  # 归一化到[0,1]
                    images.append(img_tensor)
        
        # 处理状态数据
        joint_pos = torch.tensor(obs.get('joint_positions', []), dtype=torch.float32)
        joint_vel = torch.tensor(obs.get('joint_velocities', []), dtype=torch.float32)
        
        # 处理动作数据
        if isinstance(act, dict):
            action_values = []
            for key in sorted(act.keys()):
                if isinstance(act[key], (list, np.ndarray)):
                    action_values.extend(act[key] if isinstance(act[key], list) else act[key].tolist())
            action = torch.tensor(action_values, dtype=torch.float32)
        else:
            action = torch.tensor(act, dtype=torch.float32)
        
        return {
            'images': images,
            'joint_positions': joint_pos,
            'joint_velocities': joint_vel,
            'action': action
        }


class GraspingPolicy(nn.Module):
    """抓取策略网络"""
    
    def __init__(self, 
                 image_channels: int = 3,
                 image_size: Tuple[int, int] = (224, 224),
                 joint_dim: int = 29,
                 action_dim: int = 29,
                 hidden_dim: int = 256):
        """
        初始化策略网络
        
        Args:
            image_channels: 图像通道数
            image_size: 图像尺寸
            joint_dim: 关节维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 图像编码器（使用简单的CNN）
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim)
        )
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(joint_dim * 2, hidden_dim),  # joint_pos + joint_vel
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动作输出层
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 假设动作在[-1, 1]范围内
        )
    
    def forward(self, images: list, joint_positions: torch.Tensor, 
                joint_velocities: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 图像列表（每个元素是 [B, C, H, W]）
            joint_positions: 关节位置 [B, joint_dim]
            joint_velocities: 关节速度 [B, joint_dim]
        
        Returns:
            动作 [B, action_dim]
        """
        # 编码图像（使用第一张图像，可以扩展为多图像融合）
        if len(images) > 0:
            img_feat = self.image_encoder(images[0])
        else:
            img_feat = torch.zeros(joint_positions.shape[0], 
                                  self.image_encoder[-2].out_features,
                                  device=joint_positions.device)
        
        # 编码状态
        state_input = torch.cat([joint_positions, joint_velocities], dim=-1)
        state_feat = self.state_encoder(state_input)
        
        # 融合特征
        fused = torch.cat([img_feat, state_feat], dim=-1)
        fused_feat = self.fusion(fused)
        
        # 输出动作
        action = self.action_head(fused_feat)
        
        return action


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str,
                 config_path: Optional[str] = None,
                 pretrained_model: Optional[str] = None):
        """
        初始化训练器
        
        Args:
            data_path: 训练数据路径
            output_dir: 模型输出目录
            config_path: 配置文件路径（可选）
            pretrained_model: 预训练模型路径（用于微调）
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = GraspingPolicy(
            image_channels=self.config.get('image_channels', 3),
            image_size=tuple(self.config.get('image_size', [224, 224])),
            joint_dim=self.config.get('joint_dim', 29),
            action_dim=self.config.get('action_dim', 29),
            hidden_dim=self.config.get('hidden_dim', 256)
        ).to(self.device)
        
        # 加载预训练模型（如果提供）
        if pretrained_model:
            print(f"加载预训练模型: {pretrained_model}")
            self.model.load_state_dict(torch.load(pretrained_model, map_location=self.device))
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4)
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        default_config = {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 1e-4,
            'val_split': 0.2,
            'image_channels': 3,
            'image_size': [224, 224],
            'joint_dim': 29,
            'action_dim': 29,
            'hidden_dim': 256,
            'save_interval': 10
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def train(self):
        """训练模型"""
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        
        # 创建数据集
        dataset = GraspingDataset(self.data_path)
        
        # 划分训练集和验证集
        val_size = int(len(dataset) * self.config['val_split'])
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
        
        # 训练循环
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for batch in pbar:
                images = [img.to(self.device) for img in batch['images']]
                joint_pos = batch['joint_positions'].to(self.device)
                joint_vel = batch['joint_velocities'].to(self.device)
                action_target = batch['action'].to(self.device)
                
                # 前向传播
                action_pred = self.model(images, joint_pos, joint_vel)
                
                # 计算损失
                loss = self.criterion(action_pred, action_target)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            val_loss = self._validate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model('best_model.pth')
                print(f"保存最佳模型 (Val Loss: {val_loss:.6f})")
            
            # 定期保存检查点
            if (epoch + 1) % self.config['save_interval'] == 0:
                self._save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        # 保存训练历史
        self._save_history()
        print("训练完成！")
    
    def fine_tune(self):
        """微调模型"""
        print("=" * 60)
        print("开始微调")
        print("=" * 60)
        
        # 使用较小的学习率进行微调
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.get('fine_tune_lr', 1e-5)
        
        print(f"微调学习率: {self.optimizer.param_groups[0]['lr']}")
        
        # 调用训练函数
        self.train()
    
    def _validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = [img.to(self.device) for img in batch['images']]
                joint_pos = batch['joint_positions'].to(self.device)
                joint_vel = batch['joint_velocities'].to(self.device)
                action_target = batch['action'].to(self.device)
                
                action_pred = self.model(images, joint_pos, joint_vel)
                loss = self.criterion(action_pred, action_target)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        return val_loss
    
    def _save_model(self, filename: str):
        """保存模型"""
        model_path = self.output_dir / filename
        torch.save(self.model.state_dict(), model_path)
    
    def _save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
