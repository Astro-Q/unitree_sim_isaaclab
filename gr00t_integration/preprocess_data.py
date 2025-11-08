# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
数据预处理模块
处理遥操作采集的数据，准备用于GR00T模型训练
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from tqdm import tqdm
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        train_split: float = 0.9,
        normalize: bool = True
    ):
        """
        初始化预处理器
        
        Args:
            input_dir: 输入数据目录（遥操作数据）
            output_dir: 输出数据目录（处理后数据）
            train_split: 训练集比例
            normalize: 是否标准化数据
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_split = train_split
        self.normalize = normalize
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)
        
        logger.info(f"输入目录: {self.input_dir}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def process(self):
        """执行数据处理"""
        logger.info("开始数据处理...")
        
        # 查找所有episode
        episodes = self._find_episodes()
        logger.info(f"找到 {len(episodes)} 个episodes")
        
        if len(episodes) == 0:
            logger.error("未找到任何episode数据!")
            return
        
        # 分割训练集和测试集
        np.random.seed(42)
        np.random.shuffle(episodes)
        split_idx = int(len(episodes) * self.train_split)
        train_episodes = episodes[:split_idx]
        test_episodes = episodes[split_idx:]
        
        logger.info(f"训练集: {len(train_episodes)} episodes")
        logger.info(f"测试集: {len(test_episodes)} episodes")
        
        # 处理训练集
        logger.info("\n处理训练集...")
        train_stats = self._process_episodes(train_episodes, "train")
        
        # 处理测试集
        logger.info("\n处理测试集...")
        test_stats = self._process_episodes(test_episodes, "test")
        
        # 计算并保存统计信息
        if self.normalize:
            logger.info("\n计算统计信息...")
            stats = self._compute_statistics()
            stats_path = self.output_dir / "statistics.pkl"
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)
            logger.info(f"统计信息已保存: {stats_path}")
        
        # 保存元数据
        metadata = {
            'num_train_episodes': len(train_episodes),
            'num_test_episodes': len(test_episodes),
            'train_stats': train_stats,
            'test_stats': test_stats,
            'normalize': self.normalize
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"元数据已保存: {metadata_path}")
        
        logger.info("\n数据处理完成!")
    
    def _find_episodes(self) -> List[Path]:
        """查找所有episode目录"""
        episodes = []
        
        # 查找所有可能的episode目录
        for item in self.input_dir.iterdir():
            if item.is_dir():
                # 检查是否包含必要的数据文件
                actions_file = item / "actions.npy"
                states_file = item / "actions.npy"
                
                if actions_file.exists() and states_file.exists():
                    episodes.append(item)
        
        return sorted(episodes)
    
    def _process_episodes(self, episodes: List[Path], split: str) -> Dict:
        """处理episode列表"""
        stats = {
            'total_samples': 0,
            'total_episodes': len(episodes),
            'episode_lengths': []
        }
        
        for idx, ep_dir in enumerate(tqdm(episodes, desc=f"处理{split}集")):
            try:
                # 加载原始数据
                actions = np.load(ep_dir / "actions.npy")
                states = np.load(ep_dir / "states.npy")
                
                # 检查数据一致性
                if len(actions) != len(states):
                    logger.warning(f"Episode {ep_dir.name}: actions和states长度不一致，跳过")
                    continue
                
                # 创建输出episode目录
                output_ep_dir = self.output_dir / split / f"episode_{idx:05d}"
                output_ep_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存处理后的数据
                np.save(output_ep_dir / "actions.npy", actions)
                np.save(output_ep_dir / "states.npy", states)
                
                # 复制图像数据（如果存在）
                images_file = ep_dir / "images.npy"
                if images_file.exists():
                    shutil.copy(images_file, output_ep_dir / "images.npy")
                
                # 更新统计信息
                stats['total_samples'] += len(actions)
                stats['episode_lengths'].append(len(actions))
                
            except Exception as e:
                logger.warning(f"处理episode {ep_dir.name}失败: {e}")
                continue
        
        stats['avg_episode_length'] = np.mean(stats['episode_lengths']) if stats['episode_lengths'] else 0
        stats['min_episode_length'] = np.min(stats['episode_lengths']) if stats['episode_lengths'] else 0
        stats['max_episode_length'] = np.max(stats['episode_lengths']) if stats['episode_lengths'] else 0
        
        return stats
    
    def _compute_statistics(self) -> Dict:
        """计算数据统计信息"""
        logger.info("计算统计信息...")
        
        all_states = []
        all_actions = []
        
        # 从训练集采样计算统计信息
        train_dir = self.output_dir / "train"
        episodes = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        
        # 采样计算（避免内存过大）
        sample_size = min(10000, sum(len(np.load(ep / "states.npy")) for ep in episodes))
        samples_per_episode = max(1, sample_size // len(episodes)) if episodes else 0
        
        for ep_dir in tqdm(episodes, desc="采样数据"):
            try:
                states = np.load(ep_dir / "states.npy")
                actions = np.load(ep_dir / "actions.npy")
                
                # 均匀采样
                indices = np.linspace(0, len(states) - 1, min(samples_per_episode, len(states)), dtype=int)
                
                all_states.append(states[indices])
                all_actions.append(actions[indices])
            except Exception as e:
                logger.warning(f"采样episode {ep_dir.name}失败: {e}")
                continue
        
        if len(all_states) == 0:
            logger.warning("无法计算统计信息：没有有效数据")
            return {}
        
        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        stats = {
            'state_mean': all_states.mean(axis=0).astype(np.float32),
            'state_std': (all_states.std(axis=0) + 1e-8).astype(np.float32),
            'action_mean': all_actions.mean(axis=0).astype(np.float32),
            'action_std': (all_actions.std(axis=0) + 1e-8).astype(np.float32),
        }
        
        logger.info(f"状态维度: {stats['state_mean'].shape}")
        logger.info(f"动作维度: {stats['action_mean'].shape}")
        logger.info(f"状态均值范围: [{stats['state_mean'].min():.3f}, {stats['state_mean'].max():.3f}]")
        logger.info(f"动作均值范围: [{stats['action_mean'].min():.3f}, {stats['action_mean'].max():.3f}]")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="输入数据目录")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出数据目录")
    parser.add_argument("--train_split", type=float, default=0.9,
                       help="训练集比例")
    parser.add_argument("--normalize", action="store_true",
                       help="是否标准化数据")
    
    args = parser.parse_args()
    
    # 创建预处理器并处理
    preprocessor = DataPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        normalize=args.normalize
    )
    
    preprocessor.process()


if __name__ == "__main__":
    main()
