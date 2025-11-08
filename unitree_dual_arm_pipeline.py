#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unitree 双臂5指灵巧手全流程工程主入口
整合：抓取仿真、数据采集、模型微调、部署等功能
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# 设置项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

class UnitreeDualArmPipeline:
    """Unitree 双臂抓取全流程管道"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化管道
        
        Args:
            config_path: 配置文件路径（可选）
        """
        self.config_path = config_path
        self.project_root = Path(project_root)
        
    def run_simulation(self, args):
        """运行仿真"""
        print("=" * 60)
        print("🚀 启动抓取仿真环境")
        print("=" * 60)
        
        from sim_main import main as sim_main
        # 将参数传递给 sim_main
        sys.argv = ['sim_main.py'] + self._build_sim_args(args)
        sim_main()
    
    def collect_data(self, args):
        """数据采集"""
        print("=" * 60)
        print("📊 启动数据采集")
        print("=" * 60)
        
        # 数据采集模式：遥操作或自动采集
        if args.collection_mode == "teleop":
            print("使用遥操作模式采集数据")
            print("请配合 xr_teleoperate 项目使用")
            # 启动仿真环境，等待遥操作数据
            self.run_simulation(args)
        elif args.collection_mode == "auto":
            print("使用自动采集模式")
            # TODO: 实现自动采集逻辑
            print("自动采集功能开发中...")
        else:
            print(f"未知的采集模式: {args.collection_mode}")
    
    def train_model(self, args):
        """模型训练/微调"""
        print("=" * 60)
        print("🎓 启动模型训练/微调")
        print("=" * 60)
        
        from training.trainer import ModelTrainer
        trainer = ModelTrainer(
            data_path=args.data_path,
            output_dir=args.output_dir,
            config_path=args.config_path
        )
        trainer.train()
    
    def fine_tune_model(self, args):
        """模型微调"""
        print("=" * 60)
        print("🔧 启动模型微调")
        print("=" * 60)
        
        from training.trainer import ModelTrainer
        trainer = ModelTrainer(
            data_path=args.data_path,
            output_dir=args.output_dir,
            config_path=args.config_path,
            pretrained_model=args.pretrained_model
        )
        trainer.fine_tune()
    
    def deploy_model(self, args):
        """模型部署"""
        print("=" * 60)
        print("🚢 启动模型部署")
        print("=" * 60)
        
        from deployment.deployer import ModelDeployer
        deployer = ModelDeployer(
            model_path=args.model_path,
            deployment_target=args.target,
            config_path=args.config_path
        )
        deployer.deploy()
    
    def process_data(self, args):
        """数据处理"""
        print("=" * 60)
        print("🔄 启动数据处理")
        print("=" * 60)
        
        from data_processor import DataProcessor
        processor = DataProcessor(args.data_path)
        
        if args.stats:
            stats = processor.get_statistics()
            print("\n数据集统计信息:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        if args.convert:
            processor.convert_to_training_format(
                args.output_dir,
                image_size=tuple(args.image_size)
            )
        
        if args.visualize:
            processor.visualize_episode(
                args.episode_idx,
                args.output_video
            )
    
    def _build_sim_args(self, args):
        """构建仿真参数"""
        sim_args = []
        
        if hasattr(args, 'device'):
            sim_args.extend(['--device', args.device])
        if hasattr(args, 'task'):
            sim_args.extend(['--task', args.task])
        if hasattr(args, 'robot_type'):
            sim_args.extend(['--robot_type', args.robot_type])
        if hasattr(args, 'enable_cameras') and args.enable_cameras:
            sim_args.append('--enable_cameras')
        if hasattr(args, 'enable_dex1_dds') and args.enable_dex1_dds:
            sim_args.append('--enable_dex1_dds')
        if hasattr(args, 'enable_dex3_dds') and args.enable_dex3_dds:
            sim_args.append('--enable_dex3_dds')
        if hasattr(args, 'enable_inspire_dds') and args.enable_inspire_dds:
            sim_args.append('--enable_inspire_dds')
        if hasattr(args, 'headless') and args.headless:
            sim_args.append('--headless')
        if hasattr(args, 'replay_data') and args.replay_data:
            sim_args.append('--replay_data')
        if hasattr(args, 'file_path'):
            sim_args.extend(['--file_path', args.file_path])
        
        return sim_args

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Unitree 双臂5指灵巧手全流程工程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 运行仿真:
   python unitree_dual_arm_pipeline.py simulation --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint --robot_type g129 --enable_dex3_dds

2. 数据采集（遥操作）:
   python unitree_dual_arm_pipeline.py collect --mode teleop --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint --robot_type g129 --enable_dex3_dds

3. 数据处理:
   python unitree_dual_arm_pipeline.py process --data_path /path/to/data --stats --convert --output_dir ./training_data

4. 模型训练:
   python unitree_dual_arm_pipeline.py train --data_path ./training_data --output_dir ./models

5. 模型微调:
   python unitree_dual_arm_pipeline.py fine_tune --data_path ./training_data --pretrained_model ./models/checkpoint.pth --output_dir ./models_finetuned

6. 模型部署:
   python unitree_dual_arm_pipeline.py deploy --model_path ./models/best_model.pth --target simulation
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 仿真命令
    sim_parser = subparsers.add_parser('simulation', help='运行仿真')
    sim_parser.add_argument('--task', type=str, required=True, help='任务名称')
    sim_parser.add_argument('--robot_type', type=str, default='g129', choices=['g129', 'h1_2'], help='机器人类型')
    sim_parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='计算设备')
    sim_parser.add_argument('--enable_cameras', action='store_true', help='启用相机')
    sim_parser.add_argument('--enable_dex1_dds', action='store_true', help='启用Dex1 DDS')
    sim_parser.add_argument('--enable_dex3_dds', action='store_true', help='启用Dex3 DDS')
    sim_parser.add_argument('--enable_inspire_dds', action='store_true', help='启用Inspire DDS')
    sim_parser.add_argument('--headless', action='store_true', help='无头模式')
    
    # 数据采集命令
    collect_parser = subparsers.add_parser('collect', help='数据采集')
    collect_parser.add_argument('--mode', type=str, default='teleop', choices=['teleop', 'auto'], 
                               dest='collection_mode', help='采集模式')
    collect_parser.add_argument('--task', type=str, required=True, help='任务名称')
    collect_parser.add_argument('--robot_type', type=str, default='g129', help='机器人类型')
    collect_parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    collect_parser.add_argument('--enable_cameras', action='store_true', help='启用相机')
    collect_parser.add_argument('--enable_dex1_dds', action='store_true', help='启用Dex1 DDS')
    collect_parser.add_argument('--enable_dex3_dds', action='store_true', help='启用Dex3 DDS')
    collect_parser.add_argument('--enable_inspire_dds', action='store_true', help='启用Inspire DDS')
    collect_parser.add_argument('--output_dir', type=str, default='./collected_data', help='输出目录')
    
    # 数据处理命令
    process_parser = subparsers.add_parser('process', help='数据处理')
    process_parser.add_argument('--data_path', type=str, required=True, help='数据路径')
    process_parser.add_argument('--stats', action='store_true', help='显示统计信息')
    process_parser.add_argument('--convert', action='store_true', help='转换为训练格式')
    process_parser.add_argument('--output_dir', type=str, default='./training_data', help='输出目录')
    process_parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='图像尺寸')
    process_parser.add_argument('--visualize', action='store_true', help='可视化')
    process_parser.add_argument('--episode_idx', type=int, default=0, help='Episode索引')
    process_parser.add_argument('--output_video', type=str, default=None, help='输出视频路径')
    
    # 模型训练命令
    train_parser = subparsers.add_parser('train', help='模型训练')
    train_parser.add_argument('--data_path', type=str, required=True, help='训练数据路径')
    train_parser.add_argument('--output_dir', type=str, default='./models', help='模型输出目录')
    train_parser.add_argument('--config_path', type=str, default=None, help='配置文件路径')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    train_parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    
    # 模型微调命令
    finetune_parser = subparsers.add_parser('fine_tune', help='模型微调')
    finetune_parser.add_argument('--data_path', type=str, required=True, help='微调数据路径')
    finetune_parser.add_argument('--pretrained_model', type=str, required=True, help='预训练模型路径')
    finetune_parser.add_argument('--output_dir', type=str, default='./models_finetuned', help='模型输出目录')
    finetune_parser.add_argument('--config_path', type=str, default=None, help='配置文件路径')
    finetune_parser.add_argument('--epochs', type=int, default=50, help='微调轮数')
    finetune_parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    
    # 模型部署命令
    deploy_parser = subparsers.add_parser('deploy', help='模型部署')
    deploy_parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    deploy_parser.add_argument('--target', type=str, default='simulation', 
                               choices=['simulation', 'real_robot'], help='部署目标')
    deploy_parser.add_argument('--config_path', type=str, default=None, help='配置文件路径')
    deploy_parser.add_argument('--task', type=str, default=None, help='任务名称（仿真部署时需要）')
    deploy_parser.add_argument('--robot_type', type=str, default='g129', help='机器人类型')
    
    return parser

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 创建管道实例
    pipeline = UnitreeDualArmPipeline(config_path=getattr(args, 'config_path', None))
    
    # 执行对应命令
    try:
        if args.command == 'simulation':
            pipeline.run_simulation(args)
        elif args.command == 'collect':
            pipeline.collect_data(args)
        elif args.command == 'process':
            pipeline.process_data(args)
        elif args.command == 'train':
            pipeline.train_model(args)
        elif args.command == 'fine_tune':
            pipeline.fine_tune_model(args)
        elif args.command == 'deploy':
            pipeline.deploy_model(args)
        else:
            print(f"未知命令: {args.command}")
            parser.print_help()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
