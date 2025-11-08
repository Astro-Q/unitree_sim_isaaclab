# Unitree 双臂5指灵巧手全流程工程使用指南

## 📋 目录

1. [项目概述](#项目概述)
2. [快速开始](#快速开始)
3. [功能模块](#功能模块)
4. [详细使用说明](#详细使用说明)
5. [常见问题](#常见问题)

---

## 📖 项目概述

本项目是一个完整的工程框架，用于实现宇树（Unitree）人形机器人双臂5指灵巧手的抓取仿真、数据采集、模型微调和部署全流程工作。

### 核心功能

- ✅ **抓取仿真**：基于Isaac Lab的物理仿真环境
- ✅ **数据采集**：支持遥操作和自动采集
- ✅ **数据处理**：数据格式转换、统计分析、可视化
- ✅ **模型训练**：基于采集数据的模型训练
- ✅ **模型微调**：在预训练模型基础上进行微调
- ✅ **模型部署**：部署到仿真环境或真实机器人

### 支持的机器人配置

| 机器人型号 | 自由度 | 支持的末端执行器 | 任务类型 |
|---------|--------|----------------|---------|
| G1 | 29dof | Dex1, Dex3, Inspire | 固定基座/全身移动 |
| H1-2 | 27dof | Inspire | 固定基座 |

---

## 🚀 快速开始

### 1. 环境准备

确保已安装以下依赖：

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装Isaac Sim和Isaac Lab（参考项目README）
# 下载资产文件
. fetch_assets.sh
```

### 2. 运行仿真

```bash
# G1 + Dex3 抓取圆柱体
python unitree_dual_arm_pipeline.py simulation \
    --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
    --robot_type g129 \
    --enable_dex3_dds \
    --enable_cameras
```

### 3. 数据采集

```bash
# 遥操作模式采集数据
python unitree_dual_arm_pipeline.py collect \
    --mode teleop \
    --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
    --robot_type g129 \
    --enable_dex3_dds \
    --enable_cameras \
    --output_dir ./collected_data
```

### 4. 数据处理

```bash
# 查看数据统计
python unitree_dual_arm_pipeline.py process \
    --data_path ./collected_data \
    --stats

# 转换为训练格式
python unitree_dual_arm_pipeline.py process \
    --data_path ./collected_data \
    --convert \
    --output_dir ./training_data
```

### 5. 模型训练

```bash
# 训练新模型
python unitree_dual_arm_pipeline.py train \
    --data_path ./training_data \
    --output_dir ./models \
    --epochs 100 \
    --batch_size 32
```

### 6. 模型微调

```bash
# 在预训练模型基础上微调
python unitree_dual_arm_pipeline.py fine_tune \
    --data_path ./training_data \
    --pretrained_model ./models/best_model.pth \
    --output_dir ./models_finetuned \
    --epochs 50
```

### 7. 模型部署

```bash
# 部署到仿真环境
python unitree_dual_arm_pipeline.py deploy \
    --model_path ./models/best_model.pth \
    --target simulation \
    --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint

# 部署到真实机器人
python unitree_dual_arm_pipeline.py deploy \
    --model_path ./models/best_model.pth \
    --target real_robot
```

---

## 🔧 功能模块

### 1. 仿真模块 (`sim_main.py`)

提供物理仿真环境，支持：
- 多种任务场景（抓取、堆叠、移动等）
- DDS通信协议（与真实机器人一致）
- 相机数据采集
- 数据回放和生成

### 2. 数据采集模块

**遥操作模式**：
- 配合 [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) 项目使用
- 通过VR/AR设备进行遥操作
- 自动记录动作和观察数据

**自动采集模式**（开发中）：
- 基于预设策略自动采集数据
- 支持数据增强（光照、相机参数等）

### 3. 数据处理模块 (`data_processor.py`)

功能：
- 数据格式转换（转换为训练格式）
- 数据集统计分析
- Episode可视化
- 数据过滤和清洗

### 4. 训练模块 (`training/trainer.py`)

提供：
- 数据集加载器 (`GraspingDataset`)
- 策略网络 (`GraspingPolicy`)
- 训练器 (`ModelTrainer`)
- 支持训练和微调

### 5. 部署模块 (`deployment/deployer.py`)

功能：
- 模型格式转换（PyTorch -> ONNX）
- 仿真环境部署
- 真实机器人部署
- 推理脚本生成

---

## 📚 详细使用说明

### 仿真环境配置

#### 任务列表

**G1机器人任务**：
- `Isaac-PickPlace-Cylinder-G129-Dex1-Joint`
- `Isaac-PickPlace-Cylinder-G129-Dex3-Joint`
- `Isaac-PickPlace-Cylinder-G129-Inspire-Joint`
- `Isaac-PickPlace-RedBlock-G129-Dex1-Joint`
- `Isaac-PickPlace-RedBlock-G129-Dex3-Joint`
- `Isaac-PickPlace-RedBlock-G129-Inspire-Joint`
- `Isaac-Stack-RgyBlock-G129-Dex1-Joint`
- `Isaac-Stack-RgyBlock-G129-Dex3-Joint`
- `Isaac-Stack-RgyBlock-G129-Inspire-Joint`
- `Isaac-Move-Cylinder-G129-Dex1-Wholebody`
- `Isaac-Move-Cylinder-G129-Dex3-Wholebody`
- `Isaac-Move-Cylinder-G129-Inspire-Wholebody`

**H1-2机器人任务**：
- `Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint`
- `Isaac-PickPlace-RedBlock-H12-27dof-Inspire-Joint`
- `Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint`

#### DDS通信配置

- `--enable_dex1_dds`: 启用Dex1（二指夹爪）DDS通信
- `--enable_dex3_dds`: 启用Dex3（三指灵巧手）DDS通信
- `--enable_inspire_dds`: 启用Inspire（多指灵巧手）DDS通信

**注意**：只能同时启用一种末端执行器的DDS。

### 数据采集流程

1. **启动仿真环境**：
   ```bash
   python unitree_dual_arm_pipeline.py collect \
       --mode teleop \
       --task <TASK_NAME> \
       --robot_type g129 \
       --enable_dex3_dds
   ```

2. **启动遥操作客户端**（使用xr_teleoperate项目）

3. **进行遥操作**，数据会自动保存到指定目录

4. **数据处理**：
   ```bash
   python unitree_dual_arm_pipeline.py process \
       --data_path ./collected_data \
       --stats \
       --convert \
       --output_dir ./training_data
   ```

### 模型训练流程

1. **准备训练数据**：
   - 确保数据已转换为训练格式（使用`process --convert`）

2. **配置训练参数**（可选）：
   - 编辑 `configs/default_config.json`
   - 或创建自定义配置文件

3. **开始训练**：
   ```bash
   python unitree_dual_arm_pipeline.py train \
       --data_path ./training_data \
       --output_dir ./models \
       --config_path configs/default_config.json \
       --epochs 100
   ```

4. **监控训练过程**：
   - 训练历史保存在 `models/training_history.json`
   - 最佳模型保存在 `models/best_model.pth`

### 模型微调流程

1. **准备预训练模型**：
   - 使用训练好的模型或下载的预训练模型

2. **准备微调数据**：
   - 可以是新的采集数据或特定场景的数据

3. **开始微调**：
   ```bash
   python unitree_dual_arm_pipeline.py fine_tune \
       --data_path ./fine_tune_data \
       --pretrained_model ./models/best_model.pth \
       --output_dir ./models_finetuned \
       --epochs 50 \
       --learning_rate 1e-5
   ```

### 模型部署流程

#### 部署到仿真环境

1. **转换模型**：
   ```bash
   python unitree_dual_arm_pipeline.py deploy \
       --model_path ./models/best_model.pth \
       --target simulation
   ```

2. **在仿真中使用**：
   ```bash
   python sim_main.py \
       --task <TASK_NAME> \
       --model_path ./models/best_model.onnx \
       --action_source policy
   ```

#### 部署到真实机器人

1. **创建部署包**：
   ```bash
   python unitree_dual_arm_pipeline.py deploy \
       --model_path ./models/best_model.pth \
       --target real_robot
   ```

2. **部署包内容**：
   - `model.onnx`: ONNX模型文件
   - `inference.py`: 推理脚本
   - `config.json`: 配置文件

3. **在机器人上运行**：
   ```bash
   cd deployment_package
   python inference.py \
       --model_path model.onnx \
       --image_path <IMAGE_PATH> \
       --joint_positions <JOINT_POSITIONS> \
       --joint_velocities <JOINT_VELOCITIES>
   ```

---

## ❓ 常见问题

### Q1: 如何选择末端执行器？

A: 根据您的机器人配置选择：
- Dex1: 二指夹爪，适合简单抓取任务
- Dex3: 三指灵巧手，适合复杂抓取任务
- Inspire: 多指灵巧手，适合精细操作

### Q2: 数据采集需要多长时间？

A: 取决于任务复杂度：
- 简单抓取任务：每个episode约30秒-2分钟
- 复杂任务：每个episode可能需要5-10分钟
- 建议采集至少100-200个episode用于训练

### Q3: 训练需要什么硬件？

A: 推荐配置：
- GPU: RTX 3080/3090/4090 或更高
- 内存: 16GB+
- 存储: 50GB+（用于数据和模型）

### Q4: 如何提高模型性能？

A: 建议：
1. 增加训练数据量和多样性
2. 使用数据增强（光照、相机参数等）
3. 调整模型架构（隐藏层维度等）
4. 使用预训练模型进行微调
5. 调整超参数（学习率、批次大小等）

### Q5: 模型部署到真实机器人需要注意什么？

A: 注意事项：
1. 确保模型输入输出格式与机器人接口匹配
2. 测试推理速度（建议>30 FPS）
3. 验证动作范围是否在机器人限制内
4. 添加安全检查和异常处理

---

## 📞 获取帮助

- 项目仓库: https://github.com/unitreerobotics
- 文档: 查看项目README和GUIDE文档
- 问题反馈: 提交Issue到项目仓库

---

## 📄 许可证

本项目基于 Apache License 2.0 许可证。
