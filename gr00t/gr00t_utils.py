# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
GR00T工具函数
用于数据预处理和输出后处理
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2


def prepare_gr00t_inputs(
    state: np.ndarray,
    images: Optional[Dict[str, np.ndarray]] = None,
    device: str = "cuda"
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    准备GR00T模型输入
    
    Args:
        state: 机器人状态 [state_dim]
        images: 图像字典 {camera_name: image_array}
        device: 设备
    
    Returns:
        state_tensor: 状态张量 [1, state_dim]
        images_tensor: 图像张量 [1, num_cameras, C, H, W] 或 None
    """
    # 转换状态
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    # 处理图像
    images_tensor = None
    if images is not None and len(images) > 0:
        # 将图像字典转换为张量
        image_list = []
        for camera_name in sorted(images.keys()):
            img = images[camera_name]
            # 确保图像格式正确 (H, W, C) -> (C, H, W)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img.transpose(2, 0, 1)
            image_list.append(img)
        
        # 堆叠图像 [num_cameras, C, H, W]
        images_tensor = torch.FloatTensor(np.stack(image_list)).unsqueeze(0).to(device)
        # 归一化到[0, 1]
        images_tensor = images_tensor / 255.0
    
    return state_tensor, images_tensor


def process_gr00t_outputs(
    actions: torch.Tensor,
    action_limits: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    处理GR00T模型输出
    
    Args:
        actions: 动作张量 [batch_size, action_dim]
        action_limits: 动作限制字典 {joint_name: (min, max)}
    
    Returns:
        actions_array: 动作数组 [action_dim]
    """
    # 转换为numpy
    actions_array = actions.detach().cpu().numpy()
    
    # 如果是batch，取第一个
    if len(actions_array.shape) > 1:
        actions_array = actions_array[0]
    
    # 应用动作限制
    if action_limits is not None:
        for i, (joint_name, (min_val, max_val)) in enumerate(action_limits.items()):
            if i < len(actions_array):
                actions_array[i] = np.clip(actions_array[i], min_val, max_val)
    
    return actions_array


def extract_robot_state(
    env_state: Dict,
    robot_type: str = "g1",
    effector: str = "inspire"
) -> np.ndarray:
    """
    从环境状态中提取机器人状态向量
    
    Args:
        env_state: 环境状态字典
        robot_type: 机器人类型 (g1, h1_2)
        effector: 执行器类型 (dex1, dex3, inspire)
    
    Returns:
        state_vector: 状态向量
    """
    state_parts = []
    
    # 提取关节位置
    if 'joint_positions' in env_state:
        state_parts.append(env_state['joint_positions'])
    
    # 提取关节速度
    if 'joint_velocities' in env_state:
        state_parts.append(env_state['joint_velocities'])
    
    # 提取末端执行器位姿
    if 'end_effector_pose' in env_state:
        pose = env_state['end_effector_pose']
        if isinstance(pose, dict):
            # 提取位置和四元数
            state_parts.append(pose.get('position', []))
            state_parts.append(pose.get('orientation', []))
        else:
            state_parts.append(pose)
    
    # 提取物体状态（如果有）
    if 'object_states' in env_state:
        obj_states = env_state['object_states']
        if isinstance(obj_states, list):
            for obj_state in obj_states:
                if 'position' in obj_state:
                    state_parts.append(obj_state['position'])
                if 'orientation' in obj_state:
                    state_parts.append(obj_state['orientation'])
    
    # 拼接所有状态
    if state_parts:
        state_vector = np.concatenate([np.array(s).flatten() for s in state_parts])
    else:
        # 默认状态维度（需要根据实际情况调整）
        state_vector = np.zeros(512)  # 示例维度
    
    return state_vector


def prepare_images_from_env(
    camera_images: Dict[str, np.ndarray],
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, np.ndarray]:
    """
    从环境相机图像准备模型输入图像
    
    Args:
        camera_images: 相机图像字典 {camera_name: image}
        target_size: 目标图像尺寸 (H, W)
    
    Returns:
        processed_images: 处理后的图像字典
    """
    processed_images = {}
    
    for camera_name, image in camera_images.items():
        # 确保是numpy数组
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # 调整大小
        if image.shape[:2] != target_size:
            image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # 确保数据类型正确
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        processed_images[camera_name] = image
    
    return processed_images


def compute_action_statistics(actions: np.ndarray) -> Dict[str, float]:
    """
    计算动作统计信息
    
    Args:
        actions: 动作数组 [num_samples, action_dim]
    
    Returns:
        stats: 统计信息字典
    """
    stats = {
        'mean': np.mean(actions, axis=0),
        'std': np.std(actions, axis=0),
        'min': np.min(actions, axis=0),
        'max': np.max(actions, axis=0),
    }
    return stats
