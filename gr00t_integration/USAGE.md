# Isaac-GR00T集成 - 使用说明

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch onnx onnxruntime numpy tqdm pyyaml
```

### 2. 配置设置

编辑 `configs/gr00t_config.yaml`，设置你的任务和机器人配置。

### 3. 运行完整流程

```bash
./scripts/gr00t_workflow.sh \
    --robot-type g1 \
    --effector inspire \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint
```

## 详细步骤

### 步骤1: 数据采集

使用遥操作工具采集专家演示数据，数据应保存在 `./data/teleoperate` 目录下。

### 步骤2: 数据预处理

```bash
python gr00t_integration/preprocess_data.py \
    --input_dir ./data/teleoperate \
    --output_dir ./data/processed \
    --train_split 0.9 \
    --normalize
```

### 步骤3: 模型训练

```bash
python gr00t_integration/train_gr00t.py \
    --config configs/gr00t_config.yaml \
    --data_dir ./data/processed \
    --output_dir ./outputs/gr00t_training \
    --device cuda
```

### 步骤4: 模型转换

```bash
python gr00t_integration/convert_and_deploy.py \
    --mode convert \
    --config configs/gr00t_config.yaml \
    --checkpoint ./outputs/gr00t_training/best_model.pth \
    --output_path ./models/gr00t/gr00t_model.onnx \
    --optimize
```

### 步骤5: 模型测试

```bash
python gr00t_integration/convert_and_deploy.py \
    --mode test \
    --config configs/gr00t_config.yaml \
    --onnx_path ./models/gr00t/gr00t_model.onnx
```

### 步骤6: 仿真测试

```bash
python sim_main.py \
    --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
    --action_source policy \
    --model_path ./models/gr00t/gr00t_model.onnx
```

### 步骤7: 真机部署

```bash
python deployment/deploy_to_robot.py \
    --model_path ./models/gr00t/gr00t_model.onnx \
    --robot_type g1 \
    --robot_ip 192.168.123.10 \
    --effector inspire \
    --safety_mode
```

## 配置文件说明

主要配置文件：`configs/gr00t_config.yaml`

### 关键配置项

- `gr00t.model_name`: GR00T模型名称
- `robot.type`: 机器人类型（g1/h1_2）
- `robot.effector`: 执行器类型（dex1/dex3/inspire）
- `task.name`: 任务名称
- `training.learning_rate`: 学习率（建议1e-5用于微调）
- `data.normalize`: 是否标准化数据

## 常见问题

### Q: GR00T模型无法加载怎么办？

A: 代码会自动使用替代实现。如果需要使用真实的GR00T模型，请从NVIDIA获取并正确安装。

### Q: 训练损失不下降？

A: 尝试降低学习率、检查数据质量、增加warmup轮数。

### Q: ONNX转换失败？

A: 检查模型是否包含不支持的操作，尝试不同的opset版本。

## 更多信息

详细文档请参考：
- [完整指南](docs/Isaac-GR00T集成完整指南.md)
- [项目总览](项目总览.md)
