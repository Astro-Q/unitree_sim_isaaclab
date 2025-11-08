# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Isaac-GR00T 模型微调训练脚本
基于GR00T N1.5预训练模型进行微调
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
from typing import Dict, List, Optional

# 导入GR00T模块
import sys
sys.path.append(str(Path(__file__).parent.parent))
from gr00t import GR00TModel, load_gr00t_pretrained, prepare_gr00t_inputs


class GR00TDataset(Dataset):
    """GR00T数据集 - 支持多模态输入"""
    
    def __init__(
        self,
        data_dir: str,
        use_images: bool = True,
        image_size: tuple = (224, 224)
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            use_images: 是否使用图像数据
            image_size: 图像尺寸
        """
        self.data_dir = Path(data_dir)
        self.use_images = use_images
        self.image_size = image_size
        
        # 查找所有episode
        self.episodes = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith('episode')
        ])
        
        print(f"找到 {len(self.episodes)} 个episodes")
        
        # 构建样本索引
        self.samples = []
        for ep_dir in tqdm(self.episodes, desc="索引数据集"):
            actions = np.load(ep_dir / "actions.npy")
            states = np.load(ep_dir / "states.npy")
            
            # 检查是否有图像数据
            has_images = (ep_dir / "images").exists() if use_images else False
            
            for t in range(len(actions)):
                self.samples.append((ep_dir, t, has_images))
        
        print(f"总共 {len(self.samples)} 个样本")
        print(f"使用图像: {use_images}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ep_dir, t, has_images = self.samples[idx]
        
        # 加载状态和动作
        states = np.load(ep_dir / "states.npy")
        actions = np.load(ep_dir / "actions.npy")
        
        state = states[t]
        action = actions[t]
        
        # 加载图像（如果有）
        images = None
        if self.use_images and has_images:
            try:
                images = {}
                image_dir = ep_dir / "images" / f"step_{t:06d}"
                if image_dir.exists():
                    for img_file in image_dir.glob("*.npy"):
                        camera_name = img_file.stem
                        img = np.load(img_file)
                        images[camera_name] = img
            except Exception as e:
                print(f"加载图像失败: {e}")
                images = None
        
        return {
            'state': torch.FloatTensor(state),
            'action': torch.FloatTensor(action),
            'images': images
        }


class GR00TFineTuner:
    """GR00T微调训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化微调训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        print("="*60)
        print("GR00T微调训练器初始化")
        print("="*60)
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据集
        self.train_dataset = GR00TDataset(
            data_dir=Path(config['data_dir']) / 'train',
            use_images=config.get('use_images', True),
            image_size=config.get('image_size', (224, 224))
        )
        
        self.val_dataset = GR00TDataset(
            data_dir=Path(config['data_dir']) / 'val',
            use_images=config.get('use_images', True),
            image_size=config.get('image_size', (224, 224))
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        # 获取数据维度
        sample = self.train_dataset[0]
        self.state_dim = sample['state'].shape[0]
        self.action_dim = sample['action'].shape[0]
        
        print(f"状态维度: {self.state_dim}")
        print(f"动作维度: {self.action_dim}")
        
        # 加载GR00T预训练模型
        print("\n加载GR00T预训练模型...")
        self.model = load_gr00t_pretrained(
            model_name=config.get('model_name', 'gr00t_n1.5'),
            checkpoint_path=config.get('pretrained_checkpoint'),
            device=str(self.device),
            freeze_backbone=config.get('freeze_backbone', False)
        )
        
        # 如果动作维度不匹配，替换输出层
        if hasattr(self.model, 'output_head'):
            output_dim = self.model.output_head[-1].out_features
            if output_dim != self.action_dim:
                print(f"动作维度不匹配 ({output_dim} vs {self.action_dim})，替换输出层...")
                self.model._replace_output_layer(self.action_dim)
        
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"可训练参数量: {sum(p.numel() for p in self.model.get_trainable_parameters()):,}")
        
        # 优化器
        trainable_params = self.model.get_trainable_parameters()
        self.optimizer = optim.AdamW(
            trainable_params,
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
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="训练")
        for batch in pbar:
            states = batch['state'].to(self.device)
            actions = batch['action'].to(self.device)
            images = batch.get('images', None)
            
            # 准备图像输入
            images_tensor = None
            if images is not None and len(images) > 0:
                # 处理batch中的图像
                # 这里需要根据实际的数据格式进行调整
                pass
            
            # 前向传播
            pred_actions = self.model(states, images_tensor)
            loss = self.criterion(pred_actions, actions)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(),
                max_norm=self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate_epoch(self) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                states = batch['state'].to(self.device)
                actions = batch['action'].to(self.device)
                images = batch.get('images', None)
                
                # 准备图像输入
                images_tensor = None
                if images is not None:
                    pass
                
                pred_actions = self.model(states, images_tensor)
                loss = self.criterion(pred_actions, actions)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # 保存最新的检查点
        checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最佳模型: {best_path}")
    
    def train(self):
        """执行完整的训练过程"""
        print("\n" + "="*60)
        print("开始微调训练")
        print("="*60)
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印统计信息
            lr = self.optimizer.param_groups[0]['lr']
            print(f"训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"学习率: {lr:.2e}")
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "="*60)
        print("微调训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"模型保存在: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="GR00T模型微调")
    parser.add_argument("--config", type=str, required=True,
                       help="配置文件路径")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="数据目录")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                       help="预训练模型checkpoint路径")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="冻结backbone参数")
    
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    config.update({
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'pretrained_checkpoint': args.pretrained_checkpoint,
        'freeze_backbone': args.freeze_backbone
    })
    
    # 创建训练器并训练
    trainer = GR00TFineTuner(config)
    trainer.train()


if __name__ == "__main__":
    main()
