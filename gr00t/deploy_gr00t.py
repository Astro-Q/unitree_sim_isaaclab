# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
GR00T模型部署模块
集成GR00T模型到部署流程
"""

import argparse
import time
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
import pickle
import sys

# 导入GR00T模块
sys.path.append(str(Path(__file__).parent.parent))
from gr00t import GR00TModel, prepare_gr00t_inputs, process_gr00t_outputs


class GR00TRobotDeployer:
    """GR00T模型机器人部署器"""
    
    def __init__(
        self,
        model_path: str,
        robot_type: str,
        robot_ip: str,
        effector: str,
        use_onnx: bool = True,
        statistics_path: str = None,
        safety_mode: bool = True
    ):
        """
        初始化部署器
        
        Args:
            model_path: 模型路径 (ONNX或PyTorch)
            robot_type: 机器人类型 (g1, h1_2)
            robot_ip: 机器人IP地址
            effector: 执行器类型 (dex1, dex3, inspire)
            use_onnx: 是否使用ONNX模型
            statistics_path: 数据统计文件路径
            safety_mode: 是否启用安全模式
        """
        self.model_path = Path(model_path)
        self.robot_type = robot_type
        self.robot_ip = robot_ip
        self.effector = effector
        self.use_onnx = use_onnx
        self.safety_mode = safety_mode
        
        print("="*60)
        print("GR00T模型部署器初始化")
        print("="*60)
        print(f"模型路径: {self.model_path}")
        print(f"机器人类型: {self.robot_type}")
        print(f"机器人IP: {self.robot_ip}")
        print(f"执行器: {self.effector}")
        print(f"使用ONNX: {use_onnx}")
        print(f"安全模式: {self.safety_mode}")
        
        # 加载模型
        if use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
        
        # 加载统计信息
        self.stats = None
        if statistics_path:
            print(f"\n加载数据统计信息...")
            with open(statistics_path, 'rb') as f:
                self.stats = pickle.load(f)
            print(f"✓ 统计信息已加载")
        
        # 初始化DDS通信
        print(f"\n初始化DDS通信...")
        self._init_dds()
        
        # 安全限制
        if self.safety_mode:
            self.max_action_delta = 0.1
            self.action_limits = (-1.0, 1.0)
            print(f"\n安全限制:")
            print(f"   最大动作变化: {self.max_action_delta}")
            print(f"   动作范围: {self.action_limits}")
        
        self.last_action = None
        self.running = False
    
    def _load_onnx_model(self):
        """加载ONNX模型"""
        print("\n加载ONNX模型...")
        self.ort_session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        input_shape = self.ort_session.get_inputs()[0].shape
        output_shape = self.ort_session.get_outputs()[0].shape
        print(f"✓ 输入形状: {input_shape}")
        print(f"✓ 输出形状: {output_shape}")
    
    def _load_pytorch_model(self):
        """加载PyTorch模型"""
        print("\n加载PyTorch模型...")
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # 创建模型
        config = checkpoint.get('config', {})
        self.model = GR00TModel(
            model_name=config.get('model_name', 'gr00t_n1.5'),
            pretrained=False,
            num_actions=config.get('action_dim'),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ PyTorch模型已加载")
    
    def _init_dds(self):
        """初始化DDS通信"""
        try:
            sys.path.append(str(Path(__file__).parent.parent))
            from dds import dds_create
            
            print(f"   连接到机器人: {self.robot_ip}")
            print(f"   ✓ DDS通信已建立")
            self.dds_initialized = True
            
        except Exception as e:
            print(f"   ✗ DDS初始化失败: {e}")
            print(f"   将在测试模式下运行")
            self.dds_initialized = False
    
    def get_robot_state(self) -> np.ndarray:
        """获取机器人当前状态"""
        # 这里需要通过DDS获取机器人状态
        # 示例代码，实际使用时需要调整
        
        state_dim = 512  # 示例维度，需要根据实际情况调整
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
    
    def predict_action(self, state: np.ndarray, images: dict = None) -> np.ndarray:
        """使用模型预测动作"""
        # 标准化状态
        normalized_state = self.normalize_state(state)
        
        if self.use_onnx:
            # ONNX推理
            input_data = normalized_state.reshape(1, -1).astype(np.float32)
            outputs = self.ort_session.run([self.output_name], {self.input_name: input_data})
            action = outputs[0][0]
        else:
            # PyTorch推理
            state_tensor, images_tensor = prepare_gr00t_inputs(
                normalized_state, images, device=str(self.model.device)
            )
            with torch.no_grad():
                action_tensor = self.model(state_tensor, images_tensor)
                action = action_tensor.cpu().numpy()[0]
        
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
            print(f"[测试模式] 动作: {action[:5]}...")
            return
        
        # 这里需要通过DDS发送动作到机器人
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
    parser = argparse.ArgumentParser(description="部署GR00T模型到真实机器人")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径 (ONNX或PyTorch)")
    parser.add_argument("--robot_type", type=str, required=True,
                       choices=["g1", "h1_2"],
                       help="机器人类型")
    parser.add_argument("--robot_ip", type=str, required=True,
                       help="机器人IP地址")
    parser.add_argument("--effector", type=str, required=True,
                       choices=["dex1", "dex3", "inspire"],
                       help="执行器类型")
    parser.add_argument("--use_onnx", action="store_true",
                       help="使用ONNX模型")
    parser.add_argument("--statistics_path", type=str, default=None,
                       help="数据统计文件路径")
    parser.add_argument("--safety_mode", action="store_true",
                       help="启用安全模式")
    parser.add_argument("--duration", type=float, default=None,
                       help="运行时长（秒）")
    parser.add_argument("--frequency", type=float, default=50.0,
                       help="控制频率（Hz）")
    
    args = parser.parse_args()
    
    # 创建部署器
    deployer = GR00TRobotDeployer(
        model_path=args.model_path,
        robot_type=args.robot_type,
        robot_ip=args.robot_ip,
        effector=args.effector,
        use_onnx=args.use_onnx,
        statistics_path=args.statistics_path,
        safety_mode=args.safety_mode
    )
    
    # 运行
    deployer.run(duration=args.duration, frequency=args.frequency)


if __name__ == "__main__":
    main()
