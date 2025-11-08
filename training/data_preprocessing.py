#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
数据预处理脚本
功能：
1. 数据加载和验证
2. 数据标准化和归一化
3. 数据增强
4. 训练/测试集划分
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from tqdm import tqdm
import pickle


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, 
                 data_dirs: List[str],
                 output_dir: str,
                 split_ratio: float = 0.9,
                 normalize: bool = True,
                 augment: bool = False):
        """
        初始化数据预处理器
        
        Args:
            data_dirs: 数据目录列表
            output_dir: 输出目录
            split_ratio: 训练集比例
            normalize: 是否标准化
            augment: 是否数据增强
        """
        self.data_dirs = [Path(d) for d in data_dirs]
        self.output_dir = Path(output_dir)
        self.split_ratio = split_ratio
        self.normalize = normalize
        self.augment = augment
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)
        
        # 统计信息
        self.stats = {
            'action_mean': None,
            'action_std': None,
            'state_mean': None,
            'state_std': None
        }
    
    def load_episode(self, episode_dir: Path) -> Dict:
        """加载单个episode的数据"""
        try:
            # 加载metadata
            with open(episode_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # 加载数组数据
            actions = np.load(episode_dir / "actions.npy")
            observations = np.load(episode_dir / "observations.npy")
            states = np.load(episode_dir / "states.npy")
            
            # 加载图像（如果存在）
            images = {}
            image_dir = episode_dir / "images"
            if image_dir.exists():
                for cam_name in os.listdir(image_dir):
                    cam_dir = image_dir / cam_name
                    if cam_dir.is_dir():
                        image_files = sorted(list(cam_dir.glob("*.png")) + list(cam_dir.glob("*.jpg")))
                        images[cam_name] = [cv2.imread(str(f)) for f in image_files]
            
            return {
                'metadata': metadata,
                'actions': actions,
                'observations': observations,
                'states': states,
                'images': images
            }
        except Exception as e:
            print(f"加载episode失败 {episode_dir}: {e}")
            return None
    
    def collect_all_episodes(self) -> List[Dict]:
        """收集所有episode数据"""
        all_episodes = []
        
        for data_dir in self.data_dirs:
            if not data_dir.exists():
                print(f"警告: 数据目录不存在 {data_dir}")
                continue
            
            # 查找所有episode目录
            episode_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('episode')])
            
            print(f"从 {data_dir} 加载 {len(episode_dirs)} 个episodes...")
            for ep_dir in tqdm(episode_dirs):
                episode_data = self.load_episode(ep_dir)
                if episode_data is not None:
                    all_episodes.append(episode_data)
        
        print(f"总共加载了 {len(all_episodes)} 个有效episodes")
        return all_episodes
    
    def compute_statistics(self, episodes: List[Dict]):
        """计算数据统计信息（用于标准化）"""
        if not self.normalize:
            return
        
        all_actions = []
        all_states = []
        
        for ep in episodes:
            all_actions.append(ep['actions'])
            all_states.append(ep['states'])
        
        all_actions = np.concatenate(all_actions, axis=0)
        all_states = np.concatenate(all_states, axis=0)
        
        self.stats['action_mean'] = np.mean(all_actions, axis=0)
        self.stats['action_std'] = np.std(all_actions, axis=0) + 1e-6
        self.stats['state_mean'] = np.mean(all_states, axis=0)
        self.stats['state_std'] = np.std(all_states, axis=0) + 1e-6
        
        print(f"数据统计信息:")
        print(f"  动作均值: {self.stats['action_mean'][:5]}...")
        print(f"  动作标准差: {self.stats['action_std'][:5]}...")
        print(f"  状态均值: {self.stats['state_mean'][:5]}...")
        print(f"  状态标准差: {self.stats['state_std'][:5]}...")
    
    def normalize_episode(self, episode: Dict) -> Dict:
        """标准化单个episode"""
        if not self.normalize:
            return episode
        
        normalized = episode.copy()
        normalized['actions'] = (episode['actions'] - self.stats['action_mean']) / self.stats['action_std']
        normalized['states'] = (episode['states'] - self.stats['state_mean']) / self.stats['state_std']
        
        return normalized
    
    def augment_episode(self, episode: Dict) -> List[Dict]:
        """数据增强（可选）"""
        if not self.augment:
            return [episode]
        
        augmented_episodes = [episode]
        
        # 示例：图像增强（亮度、对比度等）
        if episode['images']:
            aug_episode = episode.copy()
            aug_episode['images'] = {}
            
            for cam_name, images in episode['images'].items():
                aug_images = []
                for img in images:
                    # 随机调整亮度
                    alpha = np.random.uniform(0.8, 1.2)  # 亮度调整系数
                    aug_img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
                    aug_images.append(aug_img)
                
                aug_episode['images'][cam_name] = aug_images
            
            augmented_episodes.append(aug_episode)
        
        return augmented_episodes
    
    def split_dataset(self, episodes: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """划分训练集和测试集"""
        np.random.shuffle(episodes)
        split_idx = int(len(episodes) * self.split_ratio)
        
        train_episodes = episodes[:split_idx]
        test_episodes = episodes[split_idx:]
        
        print(f"数据集划分:")
        print(f"  训练集: {len(train_episodes)} episodes")
        print(f"  测试集: {len(test_episodes)} episodes")
        
        return train_episodes, test_episodes
    
    def save_episodes(self, episodes: List[Dict], split: str):
        """保存处理后的episodes"""
        output_split_dir = self.output_dir / split
        
        for idx, episode in enumerate(tqdm(episodes, desc=f"保存{split}数据")):
            episode_dir = output_split_dir / f"episode_{idx:05d}"
            episode_dir.mkdir(exist_ok=True)
            
            # 保存metadata
            with open(episode_dir / "metadata.json", 'w') as f:
                json.dump(episode['metadata'], f, indent=2)
            
            # 保存数组数据
            np.save(episode_dir / "actions.npy", episode['actions'])
            np.save(episode_dir / "observations.npy", episode['observations'])
            np.save(episode_dir / "states.npy", episode['states'])
            
            # 保存图像
            if episode['images']:
                image_dir = episode_dir / "images"
                image_dir.mkdir(exist_ok=True)
                
                for cam_name, images in episode['images'].items():
                    cam_dir = image_dir / cam_name
                    cam_dir.mkdir(exist_ok=True)
                    
                    for img_idx, img in enumerate(images):
                        img_path = cam_dir / f"{img_idx:06d}.png"
                        cv2.imwrite(str(img_path), img)
    
    def save_statistics(self):
        """保存统计信息"""
        stats_file = self.output_dir / "statistics.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(self.stats, f)
        
        print(f"统计信息已保存到: {stats_file}")
    
    def process(self):
        """执行完整的预处理流程"""
        print("="*60)
        print("开始数据预处理")
        print("="*60)
        
        # 1. 收集所有episodes
        episodes = self.collect_all_episodes()
        if not episodes:
            print("错误: 没有找到有效的episode数据")
            return
        
        # 2. 计算统计信息
        print("\n计算数据统计信息...")
        self.compute_statistics(episodes)
        
        # 3. 标准化和增强
        print("\n处理episodes...")
        processed_episodes = []
        for ep in tqdm(episodes):
            normalized_ep = self.normalize_episode(ep)
            augmented_eps = self.augment_episode(normalized_ep)
            processed_episodes.extend(augmented_eps)
        
        print(f"处理后共有 {len(processed_episodes)} episodes")
        
        # 4. 划分数据集
        print("\n划分数据集...")
        train_episodes, test_episodes = self.split_dataset(processed_episodes)
        
        # 5. 保存数据
        print("\n保存处理后的数据...")
        self.save_episodes(train_episodes, "train")
        self.save_episodes(test_episodes, "test")
        
        # 6. 保存统计信息
        self.save_statistics()
        
        # 7. 保存数据集信息
        dataset_info = {
            'num_train_episodes': len(train_episodes),
            'num_test_episodes': len(test_episodes),
            'total_episodes': len(processed_episodes),
            'split_ratio': self.split_ratio,
            'normalized': self.normalize,
            'augmented': self.augment,
            'source_dirs': [str(d) for d in self.data_dirs]
        }
        
        with open(self.output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print("\n"+"="*60)
        print("数据预处理完成!")
        print(f"输出目录: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="机器人数据预处理")
    parser.add_argument("--data_dirs", type=str, required=True,
                       help="数据目录，多个目录用逗号分隔")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--split_ratio", type=float, default=0.9,
                       help="训练集比例 (default: 0.9)")
    parser.add_argument("--normalize", action="store_true",
                       help="是否标准化数据")
    parser.add_argument("--augment", action="store_true",
                       help="是否进行数据增强")
    
    args = parser.parse_args()
    
    # 解析数据目录
    data_dirs = [d.strip() for d in args.data_dirs.split(',')]
    
    # 创建预处理器
    preprocessor = DataPreprocessor(
        data_dirs=data_dirs,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        normalize=args.normalize,
        augment=args.augment
    )
    
    # 执行预处理
    preprocessor.process()


if __name__ == "__main__":
    main()
