#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型部署模块
支持将训练好的模型部署到仿真环境或真实机器人
"""

import os
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

class ModelDeployer:
    """模型部署器"""
    
    def __init__(self,
                 model_path: str,
                 deployment_target: str = 'simulation',
                 config_path: Optional[str] = None):
        """
        初始化部署器
        
        Args:
            model_path: 模型路径
            deployment_target: 部署目标 ('simulation' 或 'real_robot')
            config_path: 配置文件路径（可选）
        """
        self.model_path = Path(model_path)
        self.deployment_target = deployment_target
        self.config_path = config_path
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 根据部署目标选择部署方式
        if deployment_target == 'simulation':
            self._deploy_to_simulation()
        elif deployment_target == 'real_robot':
            self._deploy_to_real_robot()
        else:
            raise ValueError(f"未知的部署目标: {deployment_target}")
    
    def _deploy_to_simulation(self):
        """部署到仿真环境"""
        print("=" * 60)
        print("🚀 部署模型到仿真环境")
        print("=" * 60)
        
        # 转换为ONNX格式（便于在仿真中使用）
        onnx_path = self.model_path.parent / f"{self.model_path.stem}.onnx"
        self._convert_to_onnx(onnx_path)
        
        print(f"✅ 模型已转换为ONNX格式: {onnx_path}")
        print("\n使用方法:")
        print("1. 在 sim_main.py 中使用 --model_path 参数指定ONNX模型路径")
        print("2. 设置 --action_source policy 以使用策略模型")
        print("\n示例命令:")
        print(f"python sim_main.py --task <TASK_NAME> --model_path {onnx_path} --action_source policy")
    
    def _deploy_to_real_robot(self):
        """部署到真实机器人"""
        print("=" * 60)
        print("🤖 部署模型到真实机器人")
        print("=" * 60)
        
        # 转换为ONNX格式（真实机器人通常使用ONNX Runtime）
        onnx_path = self.model_path.parent / f"{self.model_path.stem}.onnx"
        self._convert_to_onnx(onnx_path)
        
        # 创建部署包
        deploy_dir = self.model_path.parent / "deployment_package"
        deploy_dir.mkdir(exist_ok=True)
        
        # 复制模型文件
        import shutil
        shutil.copy(onnx_path, deploy_dir / "model.onnx")
        
        # 创建推理脚本
        self._create_inference_script(deploy_dir)
        
        # 创建配置文件
        self._create_deployment_config(deploy_dir)
        
        print(f"✅ 部署包已创建: {deploy_dir}")
        print("\n部署包包含:")
        print("  - model.onnx: ONNX模型文件")
        print("  - inference.py: 推理脚本")
        print("  - config.json: 配置文件")
        print("\n使用方法:")
        print(f"cd {deploy_dir}")
        print("python inference.py")
    
    def _convert_to_onnx(self, output_path: Path):
        """转换为ONNX格式"""
        print(f"正在转换模型为ONNX格式...")
        
        # 加载PyTorch模型
        from training.trainer import GraspingPolicy
        
        model = GraspingPolicy(
            image_channels=3,
            image_size=(224, 224),
            joint_dim=29,
            action_dim=29,
            hidden_dim=256
        )
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        
        # 创建示例输入
        dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_joint_pos = torch.randn(1, 29).to(self.device)
        dummy_joint_vel = torch.randn(1, 29).to(self.device)
        
        # 导出ONNX（注意：由于模型接受多个输入，需要特殊处理）
        # 这里简化处理，实际使用时可能需要调整模型结构
        try:
            torch.onnx.export(
                model,
                (dummy_image, dummy_joint_pos, dummy_joint_vel),
                str(output_path),
                input_names=['image', 'joint_positions', 'joint_velocities'],
                output_names=['action'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'joint_positions': {0: 'batch_size'},
                    'joint_velocities': {0: 'batch_size'},
                    'action': {0: 'batch_size'}
                },
                opset_version=11
            )
            print(f"✅ ONNX转换成功")
        except Exception as e:
            print(f"⚠️ ONNX转换失败: {e}")
            print("提示: 可能需要调整模型结构以支持ONNX导出")
    
    def _create_inference_script(self, deploy_dir: Path):
        """创建推理脚本"""
        script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理脚本
用于在真实机器人上运行训练好的模型
"""

import numpy as np
import onnxruntime as ort
import cv2
from typing import Dict, List

class ModelInference:
    """模型推理器"""
    
    def __init__(self, model_path: str):
        """
        初始化推理器
        
        Args:
            model_path: ONNX模型路径
        """
        self.session = ort.InferenceSession(model_path)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像 (H, W, C)
        
        Returns:
            预处理后的图像 (1, C, H, W)
        """
        # 调整大小
        image = cv2.resize(image, (224, 224))
        
        # 转换为CHW格式
        image = image.transpose(2, 0, 1)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 添加batch维度
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image: np.ndarray, 
                joint_positions: np.ndarray,
                joint_velocities: np.ndarray) -> np.ndarray:
        """
        预测动作
        
        Args:
            image: 输入图像
            joint_positions: 关节位置
            joint_velocities: 关节速度
        
        Returns:
            预测的动作
        """
        # 预处理
        image_processed = self.preprocess_image(image)
        joint_pos = np.expand_dims(joint_positions.astype(np.float32), axis=0)
        joint_vel = np.expand_dims(joint_velocities.astype(np.float32), axis=0)
        
        # 推理
        inputs = {
            self.input_names[0]: image_processed,
            self.input_names[1]: joint_pos,
            self.input_names[2]: joint_vel
        }
        
        outputs = self.session.run(self.output_names, inputs)
        
        return outputs[0][0]  # 返回第一个batch的结果

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型推理")
    parser.add_argument("--model_path", type=str, default="model.onnx",
                       help="模型路径")
    parser.add_argument("--image_path", type=str, required=True,
                       help="输入图像路径")
    parser.add_argument("--joint_positions", type=str, required=True,
                       help="关节位置（逗号分隔）")
    parser.add_argument("--joint_velocities", type=str, required=True,
                       help="关节速度（逗号分隔）")
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = ModelInference(args.model_path)
    
    # 加载输入
    image = cv2.imread(args.image_path)
    joint_pos = np.array([float(x) for x in args.joint_positions.split(',')])
    joint_vel = np.array([float(x) for x in args.joint_velocities.split(',')])
    
    # 预测
    action = inference.predict(image, joint_pos, joint_vel)
    
    print("预测的动作:")
    print(action)

if __name__ == "__main__":
    main()
'''
        
        script_path = deploy_dir / "inference.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 添加执行权限
        os.chmod(script_path, 0o755)
    
    def _create_deployment_config(self, deploy_dir: Path):
        """创建部署配置文件"""
        import json
        
        config = {
            "model_path": "model.onnx",
            "input_image_size": [224, 224],
            "joint_dim": 29,
            "action_dim": 29,
            "device": "cpu",  # 真实机器人通常使用CPU
            "inference_fps": 30
        }
        
        config_path = deploy_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    def deploy(self):
        """执行部署"""
        print("部署完成！")
        print(f"部署目标: {self.deployment_target}")
        print(f"模型路径: {self.model_path}")
