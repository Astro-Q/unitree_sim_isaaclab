#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unitree 双臂抓取数据预处理脚本
用于数据格式转换、数据分析和数据增强
"""

import json
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, data_root: str):
        """
        初始化数据处理器
        
        Args:
            data_root: 数据根目录路径
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"数据目录不存在: {data_root}")
    
    def list_episodes(self) -> List[Path]:
        """列出所有 episode 目录"""
        episodes = []
        for item in self.data_root.iterdir():
            if item.is_dir() and item.name.startswith("episode_"):
                episodes.append(item)
        return sorted(episodes)
    
    def load_episode(self, episode_path: Path) -> Dict:
        """加载单个 episode 数据"""
        json_path = episode_path / "data.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Episode 数据文件不存在: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        episodes = self.list_episodes()
        total_frames = 0
        total_duration = 0.0
        
        for episode_path in episodes:
            try:
                data = self.load_episode(episode_path)
                episode_frames = len(data.get("data", []))
                total_frames += episode_frames
                
                # 估算时长（假设 30 FPS）
                fps = data.get("info", {}).get("image", {}).get("fps", 30)
                total_duration += episode_frames / fps
            except Exception as e:
                print(f"警告: 无法加载 {episode_path}: {e}")
        
        stats = {
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_duration_hours": total_duration / 3600,
            "average_frames_per_episode": total_frames / len(episodes) if episodes else 0,
            "average_duration_per_episode": (total_duration / len(episodes)) if episodes else 0
        }
        
        return stats
    
    def convert_to_training_format(self, output_dir: str, 
                                   image_size: Tuple[int, int] = (224, 224)):
        """
        转换为训练格式
        
        Args:
            output_dir: 输出目录
            image_size: 目标图像尺寸 (width, height)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        episodes = self.list_episodes()
        print(f"找到 {len(episodes)} 个 episode")
        
        all_observations = []
        all_actions = []
        all_rewards = []
        
        for idx, episode_path in enumerate(episodes):
            print(f"处理 episode {idx+1}/{len(episodes)}: {episode_path.name}")
            
            try:
                data = self.load_episode(episode_path)
                episode_data = data.get("data", [])
                
                for frame_data in episode_data:
                    # 加载图像
                    observations = {}
                    colors = frame_data.get("colors", {})
                    for cam_name, img_path in colors.items():
                        full_path = episode_path / img_path
                        if full_path.exists():
                            img = cv2.imread(str(full_path))
                            img_resized = cv2.resize(img, image_size)
                            observations[f"image_{cam_name}"] = img_resized
                    
                    # 提取状态
                    states = frame_data.get("states", {})
                    if states:
                        observations["joint_positions"] = states.get("joint_positions", [])
                        observations["joint_velocities"] = states.get("joint_velocities", [])
                    
                    # 提取动作
                    actions = frame_data.get("actions", {})
                    
                    all_observations.append(observations)
                    all_actions.append(actions)
                    
            except Exception as e:
                print(f"错误: 处理 {episode_path} 时出错: {e}")
                continue
        
        # 保存为 numpy 格式
        np.save(output_path / "observations.npy", all_observations)
        np.save(output_path / "actions.npy", all_actions)
        
        print(f"转换完成，保存到: {output_path}")
        print(f"观察数据: {len(all_observations)} 帧")
        print(f"动作数据: {len(all_actions)} 帧")
    
    def filter_episodes(self, min_frames: int = 10, 
                        max_frames: int = 10000) -> List[Path]:
        """
        过滤 episode
        
        Args:
            min_frames: 最小帧数
            max_frames: 最大帧数
        
        Returns:
            过滤后的 episode 列表
        """
        episodes = self.list_episodes()
        filtered = []
        
        for episode_path in episodes:
            try:
                data = self.load_episode(episode_path)
                num_frames = len(data.get("data", []))
                
                if min_frames <= num_frames <= max_frames:
                    filtered.append(episode_path)
            except Exception as e:
                print(f"警告: 无法检查 {episode_path}: {e}")
        
        return filtered
    
    def visualize_episode(self, episode_idx: int, output_video: str = None):
        """
        可视化 episode
        
        Args:
            episode_idx: Episode 索引
            output_video: 输出视频路径（可选）
        """
        episodes = self.list_episodes()
        if episode_idx >= len(episodes):
            raise ValueError(f"Episode 索引超出范围: {episode_idx} >= {len(episodes)}")
        
        episode_path = episodes[episode_idx]
        data = self.load_episode(episode_path)
        episode_data = data.get("data", [])
        
        print(f"可视化 episode: {episode_path.name}")
        print(f"总帧数: {len(episode_data)}")
        
        if output_video:
            # 创建视频写入器
            fps = data.get("info", {}).get("image", {}).get("fps", 30)
            first_frame = episode_data[0]
            colors = first_frame.get("colors", {})
            
            if colors:
                # 获取第一张图像的尺寸
                first_img_path = episode_path / list(colors.values())[0]
                first_img = cv2.imread(str(first_img_path))
                h, w = first_img.shape[:2]
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
                
                for frame_data in episode_data:
                    colors = frame_data.get("colors", {})
                    if colors:
                        # 使用第一台相机
                        img_path = episode_path / list(colors.values())[0]
                        img = cv2.imread(str(img_path))
                        video_writer.write(img)
                
                video_writer.release()
                print(f"视频已保存到: {output_video}")
        else:
            # 显示前几帧
            for i, frame_data in enumerate(episode_data[:10]):
                colors = frame_data.get("colors", {})
                if colors:
                    img_path = episode_path / list(colors.values())[0]
                    img = cv2.imread(str(img_path))
                    cv2.imshow(f"Frame {i}", img)
                    cv2.waitKey(500)  # 显示 0.5 秒
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="Unitree 双臂抓取数据预处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 查看数据集统计信息
  python data_processor.py --data_root /path/to/data --stats
  
  # 转换为训练格式
  python data_processor.py --data_root /path/to/data --convert --output_dir ./training_data
  
  # 可视化 episode
  python data_processor.py --data_root /path/to/data --visualize --episode_idx 0
  
  # 导出为视频
  python data_processor.py --data_root /path/to/data --visualize --episode_idx 0 --output_video episode_0.mp4
        """
    )
    
    parser.add_argument("--data_root", type=str, required=True,
                       help="数据根目录路径")
    parser.add_argument("--stats", action="store_true",
                       help="显示数据集统计信息")
    parser.add_argument("--convert", action="store_true",
                       help="转换为训练格式")
    parser.add_argument("--output_dir", type=str, default="./training_data",
                       help="输出目录（用于转换）")
    parser.add_argument("--visualize", action="store_true",
                       help="可视化 episode")
    parser.add_argument("--episode_idx", type=int, default=0,
                       help="Episode 索引（用于可视化）")
    parser.add_argument("--output_video", type=str, default=None,
                       help="输出视频路径（用于可视化）")
    parser.add_argument("--filter_min", type=int, default=10,
                       help="最小帧数过滤")
    parser.add_argument("--filter_max", type=int, default=10000,
                       help="最大帧数过滤")
    
    args = parser.parse_args()
    
    # 创建数据处理器
    try:
        processor = DataProcessor(args.data_root)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 执行操作
    if args.stats:
        print("=" * 60)
        print("数据集统计信息")
        print("=" * 60)
        stats = processor.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    if args.convert:
        print("=" * 60)
        print("转换为训练格式")
        print("=" * 60)
        processor.convert_to_training_format(args.output_dir)
    
    if args.visualize:
        print("=" * 60)
        print("可视化 Episode")
        print("=" * 60)
        try:
            processor.visualize_episode(args.episode_idx, args.output_video)
        except Exception as e:
            print(f"错误: {e}")
            sys.exit(1)
    
    if not (args.stats or args.convert or args.visualize):
        print("请指定操作: --stats, --convert, 或 --visualize")
        parser.print_help()

if __name__ == "__main__":
    import sys
    main()
