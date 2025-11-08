#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unitree 双臂抓取仿真快速入门脚本
快速启动不同配置的仿真环境
"""

import argparse
import subprocess
import sys
import os

def run_simulation(config):
    """运行仿真"""
    cmd = [
        "python", "sim_main.py",
        "--device", config["device"],
        "--enable_cameras",
        "--task", config["task"],
        "--robot_type", config["robot_type"]
    ]
    
    # 添加末端执行器 DDS 选项
    if config["end_effector"] == "dex1":
        cmd.append("--enable_dex1_dds")
    elif config["end_effector"] == "dex3":
        cmd.append("--enable_dex3_dds")
    elif config["end_effector"] == "inspire":
        cmd.append("--enable_inspire_dds")
    
    # 添加其他选项
    if config.get("headless", False):
        cmd.append("--headless")
    
    if config.get("replay_data", False):
        cmd.extend(["--replay_data", "--file_path", config.get("file_path", "")])
    
    if config.get("generate_data", False):
        cmd.extend([
            "--generate_data",
            "--generate_data_dir", config.get("generate_data_dir", "./data")
        ])
    
    print("=" * 60)
    print("启动仿真环境...")
    print(f"命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n用户中断")
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Unitree 双臂抓取仿真快速启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # G1 + Dex1 抓取圆柱体
  python quick_start.py --robot g1 --effector dex1 --task cylinder
  
  # G1 + Dex3 抓取红色方块
  python quick_start.py --robot g1 --effector dex3 --task redblock
  
  # G1 + Inspire 堆叠方块
  python quick_start.py --robot g1 --effector inspire --task stack
  
  # 数据回放
  python quick_start.py --robot g1 --effector dex1 --task cylinder --replay --data_path /path/to/data
  
  # 数据生成
  python quick_start.py --robot g1 --effector dex1 --task cylinder --generate --data_path /path/to/data
        """
    )
    
    parser.add_argument("--robot", choices=["g1", "h1-2"], default="g1",
                       help="机器人型号")
    parser.add_argument("--effector", choices=["dex1", "dex3", "inspire"], 
                       default="dex1", help="末端执行器类型")
    parser.add_argument("--task", choices=["cylinder", "redblock", "stack", "move"],
                       default="cylinder", help="任务类型")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                       help="计算设备")
    parser.add_argument("--headless", action="store_true",
                       help="无头模式运行（无 GUI）")
    parser.add_argument("--replay", action="store_true",
                       help="数据回放模式")
    parser.add_argument("--generate", action="store_true",
                       help="数据生成模式")
    parser.add_argument("--data_path", type=str, default="",
                       help="数据路径（用于回放或生成）")
    
    args = parser.parse_args()
    
    # 构建任务名称
    robot_code = "G129" if args.robot == "g1" else "H12-27dof"
    effector_code = {
        "dex1": "Dex1",
        "dex3": "Dex3", 
        "inspire": "Inspire"
    }[args.effector]
    
    task_code = {
        "cylinder": "PickPlace-Cylinder",
        "redblock": "PickPlace-RedBlock",
        "stack": "Stack-RgyBlock",
        "move": "Move-Cylinder"
    }[args.task]
    
    # 构建完整任务名称
    if args.task == "move":
        task_name = f"Isaac-{task_code}-{robot_code}-{effector_code}-Wholebody"
    else:
        task_name = f"Isaac-{task_code}-{robot_code}-{effector_code}-Joint"
    
    # 构建配置
    config = {
        "device": args.device,
        "task": task_name,
        "robot_type": "g129" if args.robot == "g1" else "h1_2",
        "end_effector": args.effector,
        "headless": args.headless,
        "replay_data": args.replay,
        "file_path": args.data_path,
        "generate_data": args.generate,
        "generate_data_dir": args.data_path.replace("data", "augmented_data") if args.data_path else "./augmented_data"
    }
    
    # 验证配置
    if args.replay or args.generate:
        if not args.data_path:
            print("错误: 回放或生成模式需要指定 --data_path")
            sys.exit(1)
        if not os.path.exists(args.data_path):
            print(f"错误: 数据路径不存在: {args.data_path}")
            sys.exit(1)
    
    # 运行仿真
    run_simulation(config)

if __name__ == "__main__":
    main()
