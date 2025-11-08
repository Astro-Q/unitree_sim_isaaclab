# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
模型转换和部署模块
将训练好的GR00T模型转换为ONNX格式并部署到机器人
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import onnx
import onnxruntime as ort
import logging
from typing import Optional, Dict

# 导入GR00T模块
import sys
sys.path.append(str(Path(__file__).parent.parent))
from gr00t_integration.config import GR00TConfig, load_config
from gr00t_integration.gr00t_model import GR00TModelWrapper, load_gr00t_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConverter:
    """模型转换器"""
    
    def __init__(self, config: GR00TConfig):
        """
        初始化转换器
        
        Args:
            config: GR00TConfig配置对象
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def convert_to_onnx(
        self,
        checkpoint_path: str,
        output_path: str,
        optimize: bool = True,
        quantize: bool = False
    ):
        """
        将PyTorch模型转换为ONNX格式
        
        Args:
            checkpoint_path: PyTorch检查点路径
            output_path: 输出ONNX模型路径
            optimize: 是否优化ONNX模型
            quantize: 是否量化模型
        """
        logger.info("="*60)
        logger.info("开始模型转换")
        logger.info("="*60)
        
        # 加载模型
        logger.info(f"加载检查点: {checkpoint_path}")
        model = load_gr00t_model(self.config, checkpoint_path, device=str(self.device))
        model.eval()
        
        # 准备示例输入
        state_dim = model.state_dim
        sample_state = torch.randn(1, state_dim).to(self.device)
        sample_images = None  # 如果使用图像，需要准备示例图像
        
        logger.info(f"输入形状: {sample_state.shape}")
        
        # 导出ONNX模型
        logger.info("导出ONNX模型...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                (sample_state, sample_images),
                str(output_path),
                input_names=['state', 'images'],
                output_names=['action'],
                dynamic_axes={
                    'state': {0: 'batch_size'},
                    'images': {0: 'batch_size'},
                    'action': {0: 'batch_size'}
                },
                opset_version=13,
                do_constant_folding=True,
                verbose=False
            )
        
        logger.info(f"ONNX模型已保存: {output_path}")
        
        # 验证ONNX模型
        logger.info("验证ONNX模型...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX模型验证通过")
        
        # 优化ONNX模型
        if optimize:
            logger.info("优化ONNX模型...")
            try:
                from onnxruntime.transformers import optimizer
                optimized_model = optimizer.optimize_model(
                    str(output_path),
                    model_type='bert',  # 使用bert优化器（适用于Transformer）
                    num_heads=8,
                    hidden_size=512
                )
                optimized_path = output_path.parent / f"{output_path.stem}_optimized.onnx"
                optimized_model.save_model_to_file(str(optimized_path))
                logger.info(f"优化后的模型已保存: {optimized_path}")
            except Exception as e:
                logger.warning(f"ONNX优化失败: {e}")
        
        # 量化模型
        if quantize:
            logger.info("量化ONNX模型...")
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                quantized_path = output_path.parent / f"{output_path.stem}_quantized.onnx"
                quantize_dynamic(
                    str(output_path),
                    str(quantized_path),
                    weight_type=QuantType.QUInt8
                )
                logger.info(f"量化后的模型已保存: {quantized_path}")
            except Exception as e:
                logger.warning(f"模型量化失败: {e}")
        
        logger.info("="*60)
        logger.info("模型转换完成!")
        logger.info("="*60)
    
    def test_onnx_model(self, onnx_path: str, num_tests: int = 10):
        """
        测试ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
            num_tests: 测试次数
        """
        logger.info(f"测试ONNX模型: {onnx_path}")
        
        # 创建ONNX运行时会话
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # 获取输入输出信息
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        output_name = ort_session.get_outputs()[0].name
        output_shape = ort_session.get_outputs()[0].shape
        
        logger.info(f"输入名称: {input_name}, 形状: {input_shape}")
        logger.info(f"输出名称: {output_name}, 形状: {output_shape}")
        
        # 运行测试
        logger.info(f"运行 {num_tests} 次推理测试...")
        inference_times = []
        
        for i in range(num_tests):
            # 准备输入
            state_dim = input_shape[1] if input_shape[1] is not None else 256
            sample_state = np.random.randn(1, state_dim).astype(np.float32)
            
            # 推理
            import time
            start_time = time.time()
            outputs = ort_session.run([output_name], {input_name: sample_state})
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            inference_times.append(inference_time)
            
            logger.info(f"测试 {i+1}/{num_tests}: 推理时间 {inference_time:.2f}ms, 输出形状 {outputs[0].shape}")
        
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        logger.info(f"平均推理时间: {avg_time:.2f}ms ± {std_time:.2f}ms")


class ModelDeployer:
    """模型部署器"""
    
    def __init__(
        self,
        onnx_path: str,
        config: GR00TConfig,
        statistics_path: Optional[str] = None
    ):
        """
        初始化部署器
        
        Args:
            onnx_path: ONNX模型路径
            config: GR00TConfig配置对象
            statistics_path: 数据统计文件路径（用于反标准化）
        """
        self.onnx_path = Path(onnx_path)
        self.config = config
        self.statistics_path = statistics_path
        
        # 加载ONNX模型
        logger.info(f"加载ONNX模型: {onnx_path}")
        self.ort_session = ort.InferenceSession(str(self.onnx_path))
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        # 加载统计信息
        self.stats = None
        if statistics_path and Path(statistics_path).exists():
            import pickle
            with open(statistics_path, 'rb') as f:
                self.stats = pickle.load(f)
            logger.info("统计信息已加载")
    
    def predict(self, state: np.ndarray, images: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预测动作
        
        Args:
            state: 状态数组
            images: 图像数组或None
        
        Returns:
            action: 动作数组
        """
        # 标准化状态
        if self.stats is not None:
            state = (state - self.stats['state_mean']) / self.stats['state_std']
        
        # 准备输入
        state_input = state.reshape(1, -1).astype(np.float32)
        
        # 推理
        outputs = self.ort_session.run([self.output_name], {self.input_name: state_input})
        action = outputs[0][0]
        
        # 反标准化动作
        if self.stats is not None:
            action = action * self.stats['action_std'] + self.stats['action_mean']
        
        return action


def main():
    parser = argparse.ArgumentParser(description="模型转换和部署")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["convert", "test", "deploy"],
                       help="运行模式")
    parser.add_argument("--config", type=str, required=True,
                       help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="PyTorch检查点路径（convert模式）")
    parser.add_argument("--onnx_path", type=str, default=None,
                       help="ONNX模型路径（test/deploy模式）")
    parser.add_argument("--output_path", type=str, default=None,
                       help="输出路径")
    parser.add_argument("--optimize", action="store_true",
                       help="优化ONNX模型")
    parser.add_argument("--quantize", action="store_true",
                       help="量化ONNX模型")
    parser.add_argument("--statistics_path", type=str, default=None,
                       help="数据统计文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    if args.mode == "convert":
        if not args.checkpoint or not args.output_path:
            logger.error("convert模式需要--checkpoint和--output_path参数")
            return
        
        converter = ModelConverter(config)
        converter.convert_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output_path,
            optimize=args.optimize,
            quantize=args.quantize
        )
    
    elif args.mode == "test":
        if not args.onnx_path:
            logger.error("test模式需要--onnx_path参数")
            return
        
        converter = ModelConverter(config)
        converter.test_onnx_model(args.onnx_path)
    
    elif args.mode == "deploy":
        if not args.onnx_path:
            logger.error("deploy模式需要--onnx_path参数")
            return
        
        deployer = ModelDeployer(
            onnx_path=args.onnx_path,
            config=config,
            statistics_path=args.statistics_path
        )
        
        # 示例：运行部署
        logger.info("部署器已初始化，可以开始使用predict方法进行推理")


if __name__ == "__main__":
    main()
