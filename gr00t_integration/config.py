# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Isaac-GR00T配置模块
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml


class GR00TConfig:
    """Isaac-GR00T配置类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent))
        
        # 默认配置
        self.default_config = {
            # GR00T模型配置
            "gr00t": {
                "model_name": "GR00T-N1.5",
                "model_path": None,  # 将从HuggingFace或本地加载
                "use_pretrained": True,
                "checkpoint_path": None,
            },
            
            # 机器人配置
            "robot": {
                "type": "g1",  # g1, h1_2
                "effector": "inspire",  # dex1, dex3, inspire
                "dof": 29,  # 自由度
                "hand_dof": 24,  # 双手总自由度 (12 per hand)
            },
            
            # 任务配置
            "task": {
                "name": "Isaac-PickPlace-Cylinder-G129-Inspire-Joint",
                "observation_space": {
                    "joint_positions": True,
                    "joint_velocities": True,
                    "end_effector_pose": True,
                    "object_pose": True,
                    "images": True,
                    "image_cameras": ["front_camera", "left_wrist_camera", "right_wrist_camera"],
                },
                "action_space": {
                    "arm_joints": 14,  # 每只手臂7个关节
                    "hand_joints": 24,  # 双手24个关节
                    "total": 38,  # 14 + 24
                },
            },
            
            # 训练配置
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-5,
                "weight_decay": 1e-4,
                "epochs": 50,
                "warmup_epochs": 5,
                "gradient_clip": 1.0,
                "save_interval": 10,
                "eval_interval": 5,
            },
            
            # 数据配置
            "data": {
                "data_dir": "./data/teleoperate",
                "processed_dir": "./data/processed",
                "train_split": 0.9,
                "normalize": True,
                "augment": True,
            },
            
            # 部署配置
            "deployment": {
                "onnx_export": True,
                "optimize": True,
                "quantize": False,
                "target_device": "cuda",
            },
        }
        
        # 加载配置文件
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            self._merge_config(self.default_config, user_config)
        
        self.config = self.default_config
    
    def _merge_config(self, base: Dict, override: Dict):
        """递归合并配置"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default=None):
        """
        获取配置值
        
        Args:
            key_path: 配置路径，如 "gr00t.model_name"
            default: 默认值
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def save(self, path: str):
        """保存配置到文件"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    @property
    def model_name(self) -> str:
        return self.config["gr00t"]["model_name"]
    
    @property
    def robot_type(self) -> str:
        return self.config["robot"]["type"]
    
    @property
    def effector_type(self) -> str:
        return self.config["robot"]["effector"]
    
    @property
    def task_name(self) -> str:
        return self.config["task"]["name"]


def load_config(config_path: Optional[str] = None) -> GR00TConfig:
    """加载配置"""
    return GR00TConfig(config_path)
