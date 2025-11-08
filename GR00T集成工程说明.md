# Isaac-GR00T 集成工程总览

> **使用Isaac-GR00T N1.5模型实现宇树机器人双臂5指灵巧手抓取全流程**

## 📦 项目结构

```
workspace/
├── gr00t/                          # GR00T集成模块
│   ├── __init__.py                 # 模块初始化
│   ├── gr00t_model.py             # GR00T模型封装
│   ├── gr00t_utils.py             # 工具函数
│   ├── train_gr00t.py             # 微调训练脚本
│   ├── deploy_gr00t.py            # 部署脚本
│   └── convert_to_onnx.py         # ONNX转换工具
│
├── configs/                        # 配置文件
│   ├── gr00t_finetune.yaml        # GR00T微调配置
│   └── ...
│
├── scripts/                        # 脚本
│   ├── gr00t_end_to_end.sh        # 端到端工作流程脚本
│   └── ...
│
├── docs/                           # 文档
│   ├── GR00T完整工作流程指南.md    # 详细工作流程
│   └── ...
│
├── GR00T快速开始.md                # 快速开始指南
└── ...
```

---

## 🎯 核心功能

### 1. GR00T模型集成 (`gr00t/`)

- **`gr00t_model.py`**: GR00T模型封装，支持加载预训练模型和微调
- **`gr00t_utils.py`**: 数据预处理和输出后处理工具
- **`train_gr00t.py`**: 微调训练脚本
- **`deploy_gr00t.py`**: 模型部署脚本
- **`convert_to_onnx.py`**: PyTorch到ONNX转换工具

### 2. 配置文件 (`configs/`)

- **`gr00t_finetune.yaml`**: GR00T微调训练配置

### 3. 工作流程脚本 (`scripts/`)

- **`gr00t_end_to_end.sh`**: 一键运行完整工作流程

---

## 🚀 快速开始

### 方式1: 使用端到端脚本（推荐）

```bash
./scripts/gr00t_end_to_end.sh \
  --robot-type g1 \
  --effector inspire \
  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
  --data-dir ./teleoperate_data \
  --epochs 50
```

### 方式2: 分步执行

#### 步骤1: 数据预处理
```bash
python training/data_preprocessing.py \
  --data_dirs ./teleoperate_data \
  --output_dir ./processed_data \
  --normalize
```

#### 步骤2: GR00T微调
```bash
python gr00t/train_gr00t.py \
  --config configs/gr00t_finetune.yaml \
  --data_dir ./processed_data \
  --output_dir ./models/gr00t_finetuned
```

#### 步骤3: 转换为ONNX
```bash
python gr00t/convert_to_onnx.py \
  --checkpoint ./models/gr00t_finetuned/best_model.pth \
  --output ./models/gr00t_finetuned/model.onnx
```

#### 步骤4: 部署
```bash
python gr00t/deploy_gr00t.py \
  --model_path ./models/gr00t_finetuned/model.onnx \
  --robot_type g1 \
  --robot_ip 192.168.123.10 \
  --effector inspire \
  --use_onnx \
  --safety_mode
```

---

## 📚 文档

- **[GR00T快速开始](GR00T快速开始.md)** - 快速上手指南
- **[完整工作流程指南](docs/GR00T完整工作流程指南.md)** - 详细步骤说明
- **[项目总览](项目总览.md)** - 项目整体介绍

---

## 🔧 核心API

### 加载GR00T模型

```python
from gr00t import load_gr00t_pretrained

model = load_gr00t_pretrained(
    model_name="gr00t_n1.5",
    checkpoint_path=None,
    device="cuda",
    freeze_backbone=False
)
```

### 微调训练

```python
from gr00t.train_gr00t import GR00TFineTuner
import yaml

with open("configs/gr00t_finetune.yaml") as f:
    config = yaml.safe_load(f)

trainer = GR00TFineTuner(config)
trainer.train()
```

### 模型部署

```python
from gr00t.deploy_gr00t import GR00TRobotDeployer

deployer = GR00TRobotDeployer(
    model_path="./models/model.onnx",
    robot_type="g1",
    robot_ip="192.168.123.10",
    effector="inspire",
    use_onnx=True,
    safety_mode=True
)

deployer.run(frequency=50.0)
```

---

## ⚙️ 配置说明

### 微调配置 (`configs/gr00t_finetune.yaml`)

**关键参数:**

- `freeze_backbone`: 
  - `false` - 端到端微调（推荐，数据充足时）
  - `true` - 只微调输出层（数据少时）

- `learning_rate`: 
  - 微调建议: `1e-4` ~ `5e-5`
  - 从头训练: `3e-4` ~ `1e-3`

- `epochs`: 
  - 微调: 30-50 epochs
  - 从头训练: 100+ epochs

---

## 🎓 使用建议

### 数据采集
- **数量**: 20-50个成功的演示episode
- **质量**: 确保数据多样性（不同物体位置、姿态）
- **格式**: 与xr_teleoperate项目兼容

### 微调策略
- **数据少 (<20 episodes)**: 
  - `freeze_backbone: true`
  - `learning_rate: 1e-4`
  - `epochs: 30`

- **数据多 (>30 episodes)**: 
  - `freeze_backbone: false`
  - `learning_rate: 5e-5`
  - `epochs: 50`

### 部署优化
- 使用ONNX格式（更快）
- 启用安全模式
- 控制频率: 50 Hz

---

## 🐛 故障排查

### 1. GR00T模型加载失败

```bash
# 检查Isaac-GR00T安装
pip list | grep gr00t

# 参考官方文档安装
# https://github.com/NVIDIA/Isaac-GR00T
```

### 2. 内存不足

- 减小`batch_size`
- 使用`freeze_backbone: true`
- 使用梯度累积

### 3. 动作维度不匹配

模型会自动替换输出层。确保数据预处理正确。

---

## 📖 参考资源

- [Isaac-GR00T官方仓库](https://github.com/NVIDIA/Isaac-GR00T)
- [宇树机器人SDK](https://github.com/unitreerobotics/unitree_sdk2_python)
- [项目完整工作流程指南](docs/GR00T完整工作流程指南.md)

---

## 📝 注意事项

1. **Isaac-GR00T安装**: 需要先安装Isaac-GR00T库，参考官方文档
2. **模型架构**: 当前实现使用示例架构，实际使用时需要根据Isaac-GR00T的真实架构进行调整
3. **数据格式**: 确保数据格式与GR00T模型输入格式兼容
4. **安全模式**: 部署到真实机器人时务必启用安全模式

---

## 🙏 致谢

本项目基于以下开源项目：

- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) - NVIDIA的通用机器人基础模型
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - NVIDIA的机器人学习框架
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2_python) - 宇树机器人SDK

---

<div align="center">
  <p><strong>开始使用Isaac-GR00T进行机器人学习！</strong></p>
  <p>如有问题，请参考文档或提交Issue</p>
</div>
