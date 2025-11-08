# Unitree机器人双臂抓取仿真、数据采集与模型微调完整指南

## 📋 目录

1. [项目概述](#项目概述)
2. [环境搭建](#环境搭建)
3. [双臂抓取仿真配置](#双臂抓取仿真配置)
4. [数据采集方法](#数据采集方法)
5. [模型微调流程](#模型微调流程)
6. [实战示例](#实战示例)
7. [常见问题](#常见问题)

---

## 📖 项目概述

本项目基于 **Isaac Lab** 和 **Isaac Sim** 构建，为宇树（Unitree）G1/H1-2 人形机器人提供双臂抓取仿真环境。项目采用与真实机器人相同的 **DDS 通信协议**，确保仿真与实机代码的通用性。

### 核心特性

- ✅ **双臂协同抓取**：支持 G1-29dof 和 H1-2-27dof 机器人的双臂操作
- ✅ **多种末端执行器**：支持 Dex1（二指夹爪）、Dex3（三指灵巧手）、Inspire（多指灵巧手）
- ✅ **DDS 通信**：与真实机器人使用相同的通信协议
- ✅ **数据采集**：支持遥操作数据采集、数据回放和数据增强生成
- ✅ **模型验证**：可在仿真环境中验证训练好的模型

### 支持的机器人配置

| 机器人型号 | 自由度 | 支持的末端执行器 | 任务类型 |
|---------|--------|----------------|---------|
| G1 | 29dof | Dex1, Dex3, Inspire | 固定基座/全身移动 |
| H1-2 | 27dof | Inspire | 固定基座 |

---

## ⚙️ 环境搭建

### 1. 系统要求

**硬件要求：**
- GPU: RTX 3080/3090/4090 或更高（RTX 50系列需使用 Isaac Sim 5.0）
- 内存: 16GB+ 推荐
- 存储: 50GB+ 可用空间

**软件要求：**
- Ubuntu 20.04 / 22.04
- Python 3.8+
- CUDA 11.8+ / 12.2+

### 2. Isaac Sim 安装

根据您的 GPU 型号选择安装版本：

#### 2.1 RTX 4080 及以下（推荐 Isaac Sim 4.5.0）

```bash
# 参考安装文档
cat doc/isaacsim4.5_install_zh.md
```

#### 2.2 RTX 4080 及以上（推荐 Isaac Sim 5.0.0）

```bash
# 参考安装文档
cat doc/isaacsim5.0_install_zh.md
```

### 3. 项目依赖安装

```bash
# 克隆项目（如果还没有）
git clone <repository_url>
cd unitree_sim_isaaclab

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 git-lfs（用于下载资产文件）
sudo apt update
sudo apt install git-lfs

# 下载必要的资产文件
. fetch_assets.sh
```

### 4. Docker 环境（可选）

如果您使用 Docker 环境：

```bash
# 构建 Docker 镜像
sudo docker pull nvidia/cuda:12.2.0-runtime-ubuntu22.04
sudo docker build \
  --build-arg http_proxy=http://127.0.0.1:7890 \
  --build-arg https_proxy=http://127.0.0.1:7890 \
  -t unitree-sim:latest -f Dockerfile .

# 运行 Docker 容器
xhost +local:docker
sudo docker run --gpus all -it --rm \
  --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  unitree-sim /bin/bash
```

---

## 🤖 双臂抓取仿真配置

### 1. 理解任务结构

项目中的任务按以下结构组织：

```
tasks/
├── common_scene/          # 公共场景配置（如抓取圆柱体、红色方块等）
├── common_observations/   # 观测数据获取（相机、机器人状态、手部状态）
├── common_termination/    # 终止条件判断
├── g1_tasks/             # G1 机器人任务
│   ├── pick_place_cylinder_g1_29dof_dex1/
│   ├── pick_place_cylinder_g1_29dof_dex3/
│   ├── pick_place_cylinder_g1_29dof_inspire/
│   └── ...
└── h1-2_tasks/           # H1-2 机器人任务
```

### 2. 启动双臂抓取仿真

#### 2.1 G1 机器人 + Dex1 夹爪（二指）

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129
```

#### 2.2 G1 机器人 + Dex3 灵巧手（三指）

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds \
  --robot_type g129
```

#### 2.3 G1 机器人 + Inspire 灵巧手（多指）

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
  --enable_inspire_dds \
  --robot_type g129
```

#### 2.4 H1-2 机器人 + Inspire 灵巧手

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint \
  --enable_inspire_dds \
  --robot_type h1_2
```

### 3. 可用的双臂抓取任务

| 任务名称 | 描述 | 机器人 | 末端执行器 |
|---------|------|--------|-----------|
| `Isaac-PickPlace-Cylinder-G129-Dex1-Joint` | 抓取圆柱体 | G1 | Dex1 |
| `Isaac-PickPlace-Cylinder-G129-Dex3-Joint` | 抓取圆柱体 | G1 | Dex3 |
| `Isaac-PickPlace-Cylinder-G129-Inspire-Joint` | 抓取圆柱体 | G1 | Inspire |
| `Isaac-PickPlace-RedBlock-G129-Dex1-Joint` | 抓取红色方块 | G1 | Dex1 |
| `Isaac-PickPlace-RedBlock-G129-Dex3-Joint` | 抓取红色方块 | G1 | Dex3 |
| `Isaac-Stack-RgyBlock-G129-Dex1-Joint` | 堆叠方块 | G1 | Dex1 |
| `Isaac-Move-Cylinder-G129-Dex1-Wholebody` | 移动抓取（全身） | G1 | Dex1 |

### 4. 创建自定义双臂抓取任务

如果您想创建新的双臂抓取任务，按照以下步骤：

#### 步骤 1: 创建任务目录

```bash
cd tasks/g1_tasks
mkdir my_dual_arm_task_g1_29dof_dex1
cd my_dual_arm_task_g1_29dof_dex1
mkdir mdp
```

#### 步骤 2: 创建观测文件 (`mdp/observations.py`)

```python
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states
from tasks.common_observations.gripper_state import get_robot_gipper_joint_states
from tasks.common_observations.camera_state import get_camera_image

__all__ = [
    "get_robot_boy_joint_states",
    "get_robot_gipper_joint_states", 
    "get_camera_image"
]
```

#### 步骤 3: 创建终止条件文件 (`mdp/terminations.py`)

```python
from tasks.common_termination.base_termination_pick_place_cylinder import reset_object_estimate

__all__ = [
    "reset_object_estimate"
]
```

#### 步骤 4: 创建任务注册文件 (`__init__.py`)

```python
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym
from . import my_task_env_cfg

gym.register(
    id="Isaac-MyDualArmTask-G129-Dex1-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": my_task_env_cfg.MyDualArmTaskEnvCfg,
    },
    disable_env_checker=True,
)
```

#### 步骤 5: 创建环境配置文件 (`my_task_env_cfg.py`)

参考现有任务的配置文件，导入公共场景并配置机器人位置、相机等。

---

## 📊 数据采集方法

### 1. 遥操作数据采集

遥操作是最常用的数据采集方式，需要配合 [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) 项目使用。

#### 1.1 启动仿真环境

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129
```

#### 1.2 启动遥操作客户端

在另一个终端中启动遥操作客户端（参考 xr_teleoperate 项目文档）：

```bash
# 遥操作客户端会自动通过 DDS 接收仿真环境的数据
# 并发送控制命令回仿真环境
```

#### 1.3 数据保存

数据会自动保存到 `xr_teleoperate/teleop/utils/data` 目录下，格式为：

```
data/
├── episode_0000/
│   ├── colors/          # RGB 图像
│   ├── depths/          # 深度图像
│   ├── audios/          # 音频数据
│   └── data.json        # 元数据和状态信息
├── episode_0001/
└── ...
```

### 2. 数据回放

使用已采集的数据进行回放，用于验证或分析：

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --replay_data \
  --file_path "/path/to/your/data"
```

**注意事项：**
- 数据集格式需与 xr_teleoperate 录制的格式一致
- 任务名称必须与数据集中的任务名称匹配

### 3. 数据增强生成

通过在数据回放过程中修改光照条件和相机参数，生成多样化的增强数据：

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --replay_data \
  --file_path "/path/to/original/data" \
  --generate_data \
  --generate_data_dir "./augmented_data" \
  --modify_light \
  --modify_camera
```

**参数说明：**
- `--generate_data`: 启用数据生成
- `--generate_data_dir`: 新数据保存目录
- `--modify_light`: 修改光照条件（需在代码中调整 `update_light` 函数）
- `--modify_camera`: 修改相机参数（需在代码中调整 `batch_augment_cameras_by_name` 函数）

**自定义光照和相机参数：**

编辑 `sim_main.py` 中的相关函数：

```python
# 修改光照
if args_cli.modify_light:
    update_light(
        prim_path="/World/light",
        color=(0.75, 0.75, 0.75),  # RGB 颜色
        intensity=500.0,            # 光照强度
        radius=0.1,
        enabled=True,
        cast_shadows=True
    )

# 修改相机参数
if args_cli.modify_camera:
    batch_augment_cameras_by_name(
        names=["front_camera", "left_wrist_camera", "right_wrist_camera"],
        focal_length=3.0,          # 焦距
        horizontal_aperture=22.0,  # 水平孔径
        vertical_aperture=16.0,    # 垂直孔径
        exposure=0.8,              # 曝光
        focus_distance=1.2        # 对焦距离
    )
```

### 4. 数据格式说明

采集的数据以 JSON 格式存储，包含以下信息：

```json
{
  "info": {
    "version": "1.0.0",
    "date": "2025-01-01",
    "image": {"width": 640, "height": 480, "fps": 30},
    "joint_names": {
      "left_arm": [...],
      "right_arm": [...]
    },
    "sim_state": {...}
  },
  "text": {
    "goal": "任务目标描述",
    "desc": "任务详细描述",
    "steps": "步骤说明"
  },
  "data": [
    {
      "idx": 0,
      "colors": {"front_camera": "colors/000000_front_camera.jpg", ...},
      "depths": {...},
      "states": {...},      # 机器人关节状态
      "actions": {...},     # 动作指令
      "tactiles": {...},    # 触觉数据（如果有）
      "audios": {...}       # 音频数据（如果有）
    },
    ...
  ]
}
```

---

## 🎯 模型微调流程

### 1. 数据准备

#### 1.1 数据收集

使用遥操作收集足够的数据（建议至少 100-500 个 episode）：

```bash
# 收集多个任务的数据
# - 抓取圆柱体
# - 抓取红色方块
# - 堆叠方块
# 等
```

#### 1.2 数据预处理

将数据转换为模型训练所需的格式：

```python
# 示例：数据加载和预处理脚本
from tools.data_json_load import get_data_json_list, load_episode_data

# 加载数据列表
data_list = get_data_json_list("/path/to/data")

# 处理每个 episode
for episode_path in data_list:
    episode_data = load_episode_data(episode_path)
    # 进行数据预处理
    # - 图像归一化
    # - 动作归一化
    # - 数据增强
    # ...
```

### 2. 模型训练

#### 2.1 使用强化学习框架

本项目支持多种 RL 框架，推荐使用：

- **Isaac Lab 内置的 RL 训练器**
- **RLlib**
- **Stable Baselines3**

#### 2.2 训练配置示例

```python
# 训练配置示例（伪代码）
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import gymnasium as gym

# 解析环境配置
env_cfg = parse_env_cfg(
    "Isaac-PickPlace-Cylinder-G129-Dex1-Joint",
    device="cuda",
    num_envs=4096  # 并行环境数量
)

# 创建环境
env = gym.make("Isaac-PickPlace-Cylinder-G129-Dex1-Joint", cfg=env_cfg)

# 配置训练参数
train_cfg = {
    "total_timesteps": 10_000_000,
    "learning_rate": 3e-4,
    "batch_size": 16384,
    "gamma": 0.99,
    # ... 其他超参数
}

# 开始训练
trainer.train(env, train_cfg)
```

### 3. 模型验证

#### 3.1 在仿真环境中测试

```bash
# 使用训练好的模型进行测试
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --action_source policy \
  --model_path "path/to/trained/model.onnx"
```

#### 3.2 评估指标

- **成功率**：完成任务的 episode 比例
- **平均奖励**：每个 episode 的平均奖励值
- **执行时间**：完成任务的平均时间
- **稳定性**：多次运行的一致性

### 4. 模型微调技巧

#### 4.1 从预训练模型开始

```python
# 加载预训练模型
pretrained_model = load_model("path/to/pretrained/model.pth")

# 冻结部分层（可选）
for param in pretrained_model.backbone.parameters():
    param.requires_grad = False

# 只训练特定层
optimizer = torch.optim.Adam(
    pretrained_model.head.parameters(),
    lr=1e-5  # 较小的学习率
)
```

#### 4.2 课程学习（Curriculum Learning）

逐步增加任务难度：

1. **简单任务**：固定位置抓取
2. **中等任务**：随机位置抓取
3. **困难任务**：复杂场景抓取

#### 4.3 数据增强

使用数据增强提高模型泛化能力：

```python
# 图像增强
augmentations = [
    RandomBrightness(0.2),
    RandomContrast(0.2),
    RandomRotation(5),
    RandomCrop(0.9),
    # ...
]
```

#### 4.4 域随机化（Domain Randomization）

在仿真环境中随机化物理参数：

```python
# 随机化物体属性
object_mass_range = (0.1, 2.0)
object_friction_range = (0.3, 1.5)

# 随机化光照
light_intensity_range = (300.0, 800.0)

# 随机化相机参数
camera_noise_range = (0.0, 0.1)
```

---

## 💡 实战示例

### 示例 1: 完整的双臂抓取数据采集流程

```bash
# 步骤 1: 启动仿真环境
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129

# 步骤 2: 在另一个终端启动遥操作客户端
# （参考 xr_teleoperate 项目）

# 步骤 3: 进行遥操作，数据自动保存

# 步骤 4: 数据回放验证
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --replay_data \
  --file_path "/path/to/collected/data"
```

### 示例 2: 数据增强生成

```bash
# 基于原始数据生成增强数据
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --replay_data \
  --file_path "/path/to/original/data" \
  --generate_data \
  --generate_data_dir "./augmented_data_v1" \
  --modify_light \
  --modify_camera \
  --rerun_log
```

### 示例 3: 无头模式运行（用于大规模数据生成）

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --headless \
  --replay_data \
  --file_path "/path/to/data" \
  --generate_data \
  --generate_data_dir "./augmented_data_batch"
```

### 示例 4: 性能优化配置

```bash
# 优化性能的参数配置
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --physics_dt 0.005 \
  --render_interval 2 \
  --camera_write_interval 1 \
  --solver_iterations 4 \
  --step_hz 100
```

---

## ❓ 常见问题

### Q1: DDS 通信失败怎么办？

**A:** 确保：
1. DDS 使用相同的通道（Channel 1）
2. 网络配置正确
3. 没有其他 DDS 实例冲突

```python
# 在代码中设置 DDS 通道
ChannelFactoryInitialize(1)
```

### Q2: 仿真运行缓慢怎么办？

**A:** 尝试以下优化：
- 使用 `--headless` 模式（无 GUI）
- 增加 `--render_interval`（减少渲染频率）
- 调整 `--physics_dt`（增大物理时间步）
- 减少相机数量或分辨率

### Q3: 如何添加新的末端执行器？

**A:** 
1. 在 `robots/unitree.py` 中添加新的机器人配置
2. 创建对应的 DDS 通信模块（参考 `dds/` 目录）
3. 更新任务配置文件

### Q4: 数据采集时图像质量不佳？

**A:** 
- 检查相机配置（`tasks/common_config/camera_configs.py`）
- 调整光照条件
- 使用 `--camera_jpeg_quality` 参数调整 JPEG 质量

### Q5: 如何将仿真模型部署到真实机器人？

**A:** 
1. 确保使用相同的 DDS 通信协议
2. 进行域适应（Domain Adaptation）
3. 在真实环境中进行少量微调
4. 逐步验证和调整

---

## 📚 相关资源

- **Isaac Lab 官方文档**: https://isaac-sim.github.io/IsaacLab/
- **Unitree SDK2**: https://github.com/unitreerobotics/unitree_sdk2_python
- **xr_teleoperate**: https://github.com/unitreerobotics/xr_teleoperate
- **Discord 社区**: https://discord.gg/ZwcVwxv5rq

---

## 🎓 总结

本指南涵盖了使用 Unitree 机器人和开源平台进行双臂抓取仿真、数据采集和模型微调的完整流程。关键步骤包括：

1. ✅ **环境搭建**：安装 Isaac Sim 和项目依赖
2. ✅ **仿真配置**：选择合适的任务和机器人配置
3. ✅ **数据采集**：通过遥操作收集高质量数据
4. ✅ **数据增强**：生成多样化的训练数据
5. ✅ **模型训练**：使用 RL 框架训练策略
6. ✅ **模型验证**：在仿真环境中测试模型性能

通过遵循本指南，您应该能够成功搭建双臂抓取仿真环境，收集数据，并训练出高性能的抓取策略模型。

如有问题，请参考项目文档或联系社区支持。

---

**最后更新**: 2025-01-XX
**版本**: 1.0.0
