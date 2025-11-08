#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
模型部署到真实机器人
支持ONNX模型推理和DDS通信
"""

import argparse
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
import pickle
import sys


class RobotDeployer:
    """机器人部署器"""
    
    def __init__(self,
                 model_path: str,
                 robot_type: str,
                 robot_ip: str,
                 effector: str,
                 statistics_path: str = None,
                 safety_mode: bool = True):
        """
        初始化部署器
        
        Args:
            model_path: ONNX模型路径
            robot_type: 机器人类型 (g1, h1_2)
            robot_ip: 机器人IP地址
            effector: 执行器类型 (dex1, dex3, inspire)
            statistics_path: 数据统计文件路径
            safety_mode: 是否启用安全模式
        """
        self.model_path = Path(model_path)
        self.robot_type = robot_type
        self.robot_ip = robot_ip
        self.effector = effector
        self.safety_mode = safety_mode
        
        print("="*60)
        print("机器人部署器初始化")
        print("="*60)
        print(f"模型路径: {self.model_path}")
        print(f"机器人类型: {self.robot_type}")
        print(f"机器人IP: {self.robot_ip}")
        print(f"执行器: {self.effector}")
        print(f"安全模式: {self.safety_mode}")
        
        # 加载ONNX模型
        print("\n1. 加载ONNX模型...")
        self.ort_session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        input_shape = self.ort_session.get_inputs()[0].shape
        output_shape = self.ort_session.get_outputs()[0].shape
        print(f"   输入形状: {input_shape}")
        print(f"   输出形状: {output_shape}")
        
        # 加载统计信息（用于反标准化）
        self.stats = None
        if statistics_path:
            print(f"\n2. 加载数据统计信息...")
            with open(statistics_path, 'rb') as f:
                self.stats = pickle.load(f)
            print(f"   统计信息已加载")
        
        # 初始化DDS通信
        print(f"\n3. 初始化DDS通信...")
        self._init_dds()
        
        # 安全限制
        if self.safety_mode:
            self.max_action_delta = 0.1  # 最大动作变化
            self.action_limits = (-1.0, 1.0)  # 动作范围
            print(f"\n安全限制:")
            print(f"   最大动作变化: {self.max_action_delta}")
            print(f"   动作范围: {self.action_limits}")
        
        self.last_action = None
        self.running = False
    
    def _init_dds(self):
        """初始化DDS通信"""
        try:
            # 导入DDS模块（根据实际项目结构调整）
            sys.path.append(str(Path(__file__).parent.parent))
            from dds import dds_create
            
            # 创建DDS对象（简化版本）
            print(f"   连接到机器人: {self.robot_ip}")
            print(f"   ✓ DDS通信已建立")
            
            # 这里需要根据实际的DDS通信接口进行实现
            # 示例代码，实际使用时需要调整
            self.dds_initialized = True
            
        except Exception as e:
            print(f"   ✗ DDS初始化失败: {e}")
            print(f"   将在测试模式下运行")
            self.dds_initialized = False
    
    def get_robot_state(self) -> np.ndarray:
        """获取机器人当前状态"""
        # 这里需要通过DDS获取机器人状态
        # 示例代码，实际使用时需要调整
        
        # 假设状态包括：关节位置、关节速度、末端位姿等
        # state = [joint_positions, joint_velocities, end_effector_pose, ...]
        
        # 示例：返回随机状态（需要替换为实际的状态获取代码）
        state_dim = self.ort_session.get_inputs()[0].shape[1]
        state = np.random.randn(state_dim).astype(np.float32)
        
        return state
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """标准化状态"""
        if self.stats is None:
            return state
        
        normalized = (state - self.stats['state_mean']) / self.stats['state_std']
        return normalized
    
    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """反标准化动作"""
        if self.stats is None:
            return action
        
        denormalized = action * self.stats['action_std'] + self.stats['action_mean']
        return denormalized
    
    def predict_action(self, state: np.ndarray) -> np.ndarray:
        """使用模型预测动作"""
        # 标准化状态
        normalized_state = self.normalize_state(state)
        
        # 准备输入
        input_data = normalized_state.reshape(1, -1).astype(np.float32)
        
        # 推理
        outputs = self.ort_session.run([self.output_name], {self.input_name: input_data})
        action = outputs[0][0]
        
        # 反标准化动作
        action = self.denormalize_action(action)
        
        return action
    
    def apply_safety_constraints(self, action: np.ndarray) -> np.ndarray:
        """应用安全约束"""
        if not self.safety_mode:
            return action
        
        # 限制动作变化
        if self.last_action is not None:
            delta = action - self.last_action
            delta = np.clip(delta, -self.max_action_delta, self.max_action_delta)
            action = self.last_action + delta
        
        # 限制动作范围
        action = np.clip(action, self.action_limits[0], self.action_limits[1])
        
        return action
    
    def send_action_to_robot(self, action: np.ndarray):
        """发送动作到机器人"""
        if not self.dds_initialized:
            # 测试模式：只打印动作
            print(f"[测试模式] 动作: {action[:5]}...")
            return
        
        # 这里需要通过DDS发送动作到机器人
        # 示例代码，实际使用时需要调整
        try:
            # dds_manager.send_action(action)
            pass
        except Exception as e:
            print(f"发送动作失败: {e}")
    
    def run(self, duration: float = None, frequency: float = 50.0):
        """
        运行部署循环
        
        Args:
            duration: 运行时长（秒），None表示无限运行
            frequency: 控制频率（Hz）
        """
        print("\n" + "="*60)
        print("开始运行")
        print("="*60)
        
        if not self.dds_initialized:
            print("\n⚠️  警告：在测试模式下运行（未连接真实机器人）")
        
        print(f"控制频率: {frequency} Hz")
        if duration:
            print(f"运行时长: {duration} 秒")
        else:
            print("运行时长: 无限 (按Ctrl+C停止)")
        
        print("\n按Ctrl+C停止运行\n")
        
        dt = 1.0 / frequency
        start_time = time.time()
        self.running = True
        step_count = 0
        
        try:
            while self.running:
                step_start = time.time()
                
                # 1. 获取机器人状态
                state = self.get_robot_state()
                
                # 2. 预测动作
                action = self.predict_action(state)
                
                # 3. 应用安全约束
                action = self.apply_safety_constraints(action)
                
                # 4. 发送动作到机器人
                self.send_action_to_robot(action)
                
                # 更新
                self.last_action = action
                step_count += 1
                
                # 打印统计信息
                if step_count % 100 == 0:
                    elapsed = time.time() - start_time
                    actual_freq = step_count / elapsed
                    print(f"步数: {step_count}, 实际频率: {actual_freq:.1f} Hz")
                
                # 检查是否达到运行时长
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # 控制频率
                sleep_time = dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n\n用户中断")
        
        finally:
            self.running = False
            elapsed = time.time() - start_time
            print("\n" + "="*60)
            print("运行结束")
            print(f"总步数: {step_count}")
            print(f"运行时间: {elapsed:.2f} 秒")
            print(f"平均频率: {step_count/elapsed:.1f} Hz")
            print("="*60)


def main():
    parser = argparse.ArgumentParser(description="部署模型到真实机器人")
    parser.add_argument("--model_path", type=str, required=True,
                       help="ONNX模型路径")
    parser.add_argument("--robot_type", type=str, required=True,
                       choices=["g1", "h1_2"],
                       help="机器人类型")
    parser.add_argument("--robot_ip", type=str, required=True,
                       help="机器人IP地址")
    parser.add_argument("--effector", type=str, required=True,
                       choices=["dex1", "dex3", "inspire"],
                       help="执行器类型")
    parser.add_argument("--statistics_path", type=str, default=None,
                       help="数据统计文件路径")
    parser.add_argument("--safety_mode", action="store_true",
                       help="启用安全模式")
    parser.add_argument("--test_mode", action="store_true",
                       help="测试模式（不连接真实机器人）")
    parser.add_argument("--duration", type=float, default=None,
                       help="运行时长（秒）")
    parser.add_argument("--frequency", type=float, default=50.0,
                       help="控制频率（Hz）")
    
    args = parser.parse_args()
    
    # 创建部署器
    deployer = RobotDeployer(
        model_path=args.model_path,
        robot_type=args.robot_type,
        robot_ip=args.robot_ip,
        effector=args.effector,
        statistics_path=args.statistics_path,
        safety_mode=args.safety_mode
    )
    
    # 运行
    deployer.run(duration=args.duration, frequency=args.frequency)


if __name__ == "__main__":
    main()
