# Isaac-GR00T集成完整指南

> **使用NVIDIA Isaac-GR00T N1.5模型进行宇树机器人双臂5指灵巧手抓取任务的全流程指南**

## 📖 目录

1. [概述](#概述)
2. [环境准备](#环境准备)
3. [快速开始](#快速开始)
4. [详细流程](#详细流程)
5. [API参考](#api参考)
6. [故障排查](#故障排查)

---

## 概述

本项目整合了NVIDIA Isaac-GR00T N1.5模型，用于宇树机器人（G1/H1-2）的双臂5指灵巧手（Inspire）抓取任务。提供了从数据采集、模型微调、到真机部署的完整解决方案。

### 主要特性

- ✅ **基于GR00T N1.5**: 使用NVIDIA最新的通用机器人基础模型
- ✅ **双臂5指灵巧手**: 支持Inspire灵巧手的精细操作
- ✅ **完整工作流程**: 数据采集 → 预处理 → 训练 → 转换 → 部署
- ✅ **仿真到真机**: 无缝从仿真环境迁移到真实机器人
- ✅ **ONNX部署**: 高效的模型推理和部署

---

## 环境准备

### 1. 系统要求

- **操作系统**: Ubuntu 20.04 / 22.04
- **GPU**: NVIDIA RTX 3080 或更高（推荐 RTX 4090）
- **CUDA**: 11.8 / 12.2
- **Python**: 3.8+

### 2. 安装依赖

```bash
# 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnx onnxruntime onnxruntime-gpu
pip install numpy tqdm pyyaml

# 安装Isaac-GR00T（如果可用）
# 注意: Isaac-GR00T可能需要从NVIDIA官方获取
# pip install isaac-groot  # 根据实际情况调整
```

### 3. 下载GR00T模型

```bash
# 从HuggingFace下载GR00T N1.5模型
# 或使用本地模型路径
# 具体方法取决于NVIDIA的发布方式
```

---

## 快速开始

### 一键运行完整流程

```bash
cd /workspace
./scripts/gr00t_workflow.sh \
    --robot-type g1 \
    --effector inspire \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint
```

### 分步执行

#### 1. 数据采集

使用遥操作工具采集专家演示数据：

```bash
# 终端1: 启动仿真
python sim_main.py \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
    --enable_inspire_dds \
    --robot_type g129

# 终端2: 启动遥操作（需要xr_teleoperate项目）
# 数据将自动保存到 ./data/teleoperate
```

#### 2. 数据预处理

```bash
python gr00t_integration/preprocess_data.py \
    --input_dir ./data/teleoperate \
    --output_dir ./data/processed \
    --train_split 0.9 \
    --normalize
```

#### 3. 模型训练

```bash
python gr00t_integration/train_gr00t.py \
    --config configs/gr00t_config.yaml \
    --data_dir ./data/processed \
    --output_dir ./outputs/gr00t_training \
    --device cuda
```

#### 4. 模型转换

```bash
python gr00t_integration/convert_and_deploy.py \
    --mode convert \
    --config configs/gr00t_config.yaml \
    --checkpoint ./outputs/gr00t_training/best_model.pth \
    --output_path ./models/gr00t/gr00t_model.onnx \
    --optimize
```

#### 5. 仿真测试

```bash
python sim_main.py \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
    --action_source policy \
    --model_path ./models/gr00t/gr00t_model.onnx
```

#### 6. 真机部署

```bash
python deployment/deploy_to_robot.py \
    --model_path ./models/gr00t/gr00t_model.onnx \
    --robot_type g1 \
    --robot_ip 192.168.123.10 \
    --effector inspire \
    --safety_mode
```

---

## 详细流程

### 数据采集

采集20-50个专家演示episode，每个episode包含：
- `states.npy`: 状态序列
- `actions.npy`: 动作序列
- `images.npy`: 图像序列（可选）

### 数据预处理

数据预处理包括：
1. **数据验证**: 检查数据完整性和一致性
2. **数据分割**: 按比例分割训练集和测试集
3. **数据标准化**: 计算均值和标准差，用于归一化
4. **数据保存**: 保存处理后的数据和统计信息

### 模型训练

GR00T微调训练特点：
- **小学习率**: 使用1e-5的学习率进行微调
- **冻结部分层**: 可以冻结GR00T的backbone，只训练适配层
- **梯度裁剪**: 防止梯度爆炸
- **学习率调度**: 使用warmup和cosine退火

### 模型转换

将PyTorch模型转换为ONNX格式：
- **动态轴**: 支持batch维度变化
- **优化**: 使用ONNX Runtime优化器
- **量化**: 可选INT8量化（可能影响精度）

### 模型部署

部署到真实机器人：
- **ONNX推理**: 使用ONNX Runtime进行高效推理
- **数据标准化**: 使用训练时的统计信息
- **安全约束**: 限制动作变化和范围

---

## API参考

### GR00TConfig

配置管理类：

```python
from gr00t_integration.config import load_config

config = load_config("configs/gr00t_config.yaml")
robot_type = config.robot_type
task_name = config.task_name
```

### GR00TModelWrapper

模型包装类：

```python
from gr00t_integration.gr00t_model import load_gr00t_model

model = load_gr00t_model(config, checkpoint_path="best_model.pth")
action = model.predict(state, images=None)
```

### ModelDeployer

模型部署类：

```python
from gr00t_integration.convert_and_deploy import ModelDeployer

deployer = ModelDeployer(
    onnx_path="model.onnx",
    config=config,
    statistics_path="statistics.pkl"
)
action = deployer.predict(state)
```

---

## 故障排查

### 1. GR00T模型加载失败

**问题**: 无法加载GR00T预训练模型

**解决方案**:
- 检查模型路径是否正确
- 确认Isaac-GR00T库已正确安装
- 如果无法获取GR00T，代码会自动使用替代实现

### 2. 训练损失不下降

**问题**: 训练过程中损失不下降或震荡

**解决方案**:
- 降低学习率（尝试1e-6）
- 检查数据质量和标注
- 增加warmup轮数
- 检查数据标准化是否正确

### 3. ONNX转换失败

**问题**: PyTorch模型无法转换为ONNX

**解决方案**:
- 检查模型是否包含不支持的操作
- 尝试不同的opset版本
- 简化模型结构（移除动态操作）

### 4. 推理速度慢

**问题**: ONNX模型推理速度慢

**解决方案**:
- 启用ONNX优化（--optimize）
- 使用TensorRT后端（如果可用）
- 量化模型（可能影响精度）

### 5. 真机部署失败

**问题**: 模型在真机上表现不佳

**解决方案**:
- 检查数据分布是否匹配
- 验证状态标准化是否正确
- 添加安全约束限制动作
- 在仿真环境中充分测试

---

## 参考资料

- [Isaac-GR00T官方文档](https://github.com/NVIDIA/Isaac-GR00T)
- [宇树机器人SDK](https://github.com/unitreerobotics)
- [ONNX Runtime文档](https://onnxruntime.ai/)

---

## 许可证

本项目采用 Apache License 2.0 开源许可证。

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues
- Discord社区
