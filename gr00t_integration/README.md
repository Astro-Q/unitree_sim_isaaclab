# Isaac-GR00T集成 - README

## 概述

本模块整合了NVIDIA Isaac-GR00T N1.5模型，用于宇树机器人双臂5指灵巧手抓取任务的全流程开发。

## 目录结构

```
gr00t_integration/
├── __init__.py              # 模块初始化
├── config.py                # 配置管理
├── gr00t_model.py           # GR00T模型封装
├── train_gr00t.py           # 训练脚本
├── preprocess_data.py        # 数据预处理
└── convert_and_deploy.py    # 模型转换和部署
```

## 快速开始

### 1. 配置

编辑 `configs/gr00t_config.yaml` 设置你的任务和机器人配置。

### 2. 数据采集

使用遥操作工具采集数据，保存到 `./data/teleoperate`

### 3. 数据预处理

```bash
python gr00t_integration/preprocess_data.py \
    --input_dir ./data/teleoperate \
    --output_dir ./data/processed \
    --normalize
```

### 4. 训练模型

```bash
python gr00t_integration/train_gr00t.py \
    --config configs/gr00t_config.yaml \
    --data_dir ./data/processed \
    --output_dir ./outputs/gr00t_training
```

### 5. 转换模型

```bash
python gr00t_integration/convert_and_deploy.py \
    --mode convert \
    --config configs/gr00t_config.yaml \
    --checkpoint ./outputs/gr00t_training/best_model.pth \
    --output_path ./models/gr00t/gr00t_model.onnx \
    --optimize
```

### 6. 部署

参考 `docs/Isaac-GR00T集成完整指南.md` 进行部署。

## 完整工作流程

使用一键脚本：

```bash
./scripts/gr00t_workflow.sh \
    --robot-type g1 \
    --effector inspire \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint
```

## 文档

详细文档请参考：
- [完整指南](docs/Isaac-GR00T集成完整指南.md)
- [API文档](docs/API文档.md)

## 注意事项

1. **GR00T模型**: 需要从NVIDIA获取Isaac-GR00T模型，如果无法获取，代码会自动使用替代实现
2. **数据质量**: 确保采集的数据质量高，建议至少20-50个episode
3. **计算资源**: 训练需要GPU，推荐RTX 4090或更高
4. **真机安全**: 部署到真机前务必在仿真环境中充分测试

## 许可证

Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
License: Apache License, Version 2.0
