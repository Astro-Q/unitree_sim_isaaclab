# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
GR00T模型转换工具
将PyTorch模型转换为ONNX格式用于部署
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from gr00t import GR00TModel, prepare_gr00t_inputs


def convert_to_onnx(
    checkpoint_path: str,
    output_path: str,
    state_dim: int = 512,
    batch_size: int = 1,
    opset_version: int = 13
):
    """
    将PyTorch模型转换为ONNX格式
    
    Args:
        checkpoint_path: PyTorch checkpoint路径
        output_path: 输出ONNX文件路径
        state_dim: 状态维度
        batch_size: 批次大小
        opset_version: ONNX opset版本
    """
    print("="*60)
    print("GR00T模型转换为ONNX")
    print("="*60)
    
    # 加载checkpoint
    print(f"\n1. 加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建模型
    print("2. 创建模型...")
    config = checkpoint.get('config', {})
    model = GR00TModel(
        model_name=config.get('model_name', 'gr00t_n1.5'),
        pretrained=False,
        num_actions=config.get('action_dim', 53),  # G1 29DOF + Inspire 24DOF
        device='cpu'
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ 模型已加载")
    
    # 创建示例输入
    print("3. 准备示例输入...")
    dummy_state = torch.randn(batch_size, state_dim)
    print(f"   状态形状: {dummy_state.shape}")
    
    # 导出ONNX
    print(f"\n4. 导出ONNX模型到: {output_path}")
    torch.onnx.export(
        model,
        dummy_state,
        output_path,
        input_names=['state'],
        output_names=['action'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        },
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )
    
    print("✓ ONNX模型导出成功!")
    
    # 验证ONNX模型
    print("\n5. 验证ONNX模型...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证通过")
    except ImportError:
        print("⚠️ 未安装onnx库，跳过验证")
    except Exception as e:
        print(f"⚠️ ONNX模型验证失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="将GR00T模型转换为ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="PyTorch checkpoint路径")
    parser.add_argument("--output", type=str, required=True,
                       help="输出ONNX文件路径")
    parser.add_argument("--state_dim", type=int, default=512,
                       help="状态维度")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--opset_version", type=int, default=13,
                       help="ONNX opset版本")
    
    args = parser.parse_args()
    
    convert_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        state_dim=args.state_dim,
        batch_size=args.batch_size,
        opset_version=args.opset_version
    )


if __name__ == "__main__":
    main()
