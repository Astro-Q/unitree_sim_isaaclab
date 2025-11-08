#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
将PyTorch模型转换为ONNX格式
用于加速推理和跨平台部署
"""

import argparse
import torch
import torch.onnx
from pathlib import Path
import sys
import numpy as np

# 添加训练模块路径
sys.path.append(str(Path(__file__).parent.parent))
from training.train_bc import BCPolicy


def convert_to_onnx(model_path: str,
                   output_path: str,
                   state_dim: int,
                   action_dim: int,
                   opset_version: int = 14,
                   simplify: bool = True):
    """
    将PyTorch模型转换为ONNX
    
    Args:
        model_path: PyTorch模型路径
        output_path: ONNX输出路径
        state_dim: 状态维度
        action_dim: 动作维度
        opset_version: ONNX opset版本
        simplify: 是否简化ONNX模型
    """
    print("="*60)
    print("PyTorch to ONNX 转换")
    print("="*60)
    
    # 加载PyTorch模型
    print(f"\n1. 加载PyTorch模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 从checkpoint中获取配置
    config = checkpoint.get('config', {})
    hidden_dims = config.get('hidden_dims', [512, 512, 256])
    
    # 创建模型
    model = BCPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   状态维度: {state_dim}")
    print(f"   动作维度: {action_dim}")
    print(f"   隐藏层: {hidden_dims}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例输入
    print(f"\n2. 创建示例输入")
    dummy_input = torch.randn(1, state_dim)
    
    # 测试模型
    print(f"\n3. 测试PyTorch模型")
    with torch.no_grad():
        pytorch_output = model(dummy_input)
    print(f"   输出形状: {pytorch_output.shape}")
    
    # 转换为ONNX
    print(f"\n4. 转换为ONNX格式")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    
    print(f"   ONNX模型已保存: {output_path}")
    
    # 验证ONNX模型
    print(f"\n5. 验证ONNX模型")
    try:
        import onnx
        import onnxruntime as ort
        
        # 检查ONNX模型
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("   ONNX模型检查通过")
        
        # 使用ONNX Runtime测试
        ort_session = ort.InferenceSession(str(output_path))
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # 比较输出
        max_diff = np.abs(pytorch_output.numpy() - ort_outputs[0]).max()
        print(f"   PyTorch vs ONNX 最大差异: {max_diff:.2e}")
        
        if max_diff < 1e-5:
            print("   ✓ 转换成功！输出一致")
        else:
            print("   ⚠ 警告：输出存在较大差异")
        
    except ImportError:
        print("   ⚠ 未安装onnx/onnxruntime，跳过验证")
    
    # 简化ONNX模型（可选）
    if simplify:
        try:
            import onnxsim
            print(f"\n6. 简化ONNX模型")
            
            simplified_model, check = onnxsim.simplify(onnx_model)
            
            if check:
                simplified_path = output_path.parent / f"{output_path.stem}_simplified.onnx"
                onnx.save(simplified_model, str(simplified_path))
                print(f"   简化模型已保存: {simplified_path}")
                
                # 比较文件大小
                original_size = output_path.stat().st_size / 1024 / 1024
                simplified_size = simplified_path.stat().st_size / 1024 / 1024
                print(f"   原始大小: {original_size:.2f} MB")
                print(f"   简化后大小: {simplified_size:.2f} MB")
                print(f"   压缩率: {(1 - simplified_size/original_size)*100:.1f}%")
            else:
                print("   简化失败")
                
        except ImportError:
            print("   ⚠ 未安装onnx-simplifier，跳过简化")
    
    print("\n" + "="*60)
    print("转换完成!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="PyTorch模型转ONNX")
    parser.add_argument("--model_path", type=str, required=True,
                       help="PyTorch模型路径 (.pth)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="ONNX输出路径 (.onnx)")
    parser.add_argument("--state_dim", type=int, required=True,
                       help="状态维度")
    parser.add_argument("--action_dim", type=int, required=True,
                       help="动作维度")
    parser.add_argument("--opset_version", type=int, default=14,
                       help="ONNX opset版本")
    parser.add_argument("--no_simplify", action="store_true",
                       help="不简化ONNX模型")
    
    args = parser.parse_args()
    
    convert_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        opset_version=args.opset_version,
        simplify=not args.no_simplify
    )


if __name__ == "__main__":
    main()
