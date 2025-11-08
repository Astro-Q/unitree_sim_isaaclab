# Isaac-GR00T 集成使用指南

> **快速开始使用Isaac-GR00T N1.5模型进行宇树机器人双臂5指灵巧手抓取**

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装Isaac-GR00T (参考官方文档)
# https://github.com/NVIDIA/Isaac-GR00T

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 数据采集

配合遥操作进行数据采集（20-50个episode）：

```bash
# 终端1: 启动仿真
python sim_main.py \
  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
  --enable_inspire_dds \
  --robot_type g129

# 终端2: 启动遥操作
# (使用xr_teleoperate项目)
```

### 3. 完整工作流程（一键运行）

```bash
./scripts/gr00t_end_to_end.sh \
  --robot-type g1 \
  --effector inspire \
  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
  --data-dir ./teleoperate_data \
  --epochs 50
```

### 4. 仿真测试

```bash
python sim_main.py \
  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
  --action_source policy \
  --model_path ./models/gr00t_g1_inspire/model.onnx \
  --enable_inspire_dds \
  --robot_type g129
```

### 5. 真机部署

```bash
python gr00t/deploy_gr00t.py \
  --model_path ./models/gr00t_g1_inspire/model.onnx \
  --robot_type g1 \
  --robot_ip 192.168.123.10 \
  --effector inspire \
  --use_onnx \
  --safety_mode
```

---

## 📚 详细文档

- [完整工作流程指南](docs/GR00T完整工作流程指南.md) - 详细步骤说明
- [项目总览](项目总览.md) - 项目整体介绍
- [API文档](docs/API文档.md) - API参考

---

## 🔧 核心模块

### GR00T模型加载

```python
from gr00t import load_gr00t_pretrained

# 加载预训练模型
model = load_gr00t_pretrained(
    model_name="gr00t_n1.5",
    checkpoint_path=None,  # None表示从官方仓库下载
    device="cuda",
    freeze_backbone=False  # False=端到端微调
)
```

### 微调训练

```python
from gr00t.train_gr00t import GR00TFineTuner
import yaml

# 加载配置
with open("configs/gr00t_finetune.yaml") as f:
    config = yaml.safe_load(f)

# 创建训练器
trainer = GR00TFineTuner(config)
trainer.train()
```

### 模型部署

```python
from gr00t.deploy_gr00t import GR00TRobotDeployer

# 创建部署器
deployer = GR00TRobotDeployer(
    model_path="./models/model.onnx",
    robot_type="g1",
    robot_ip="192.168.123.10",
    effector="inspire",
    use_onnx=True,
    safety_mode=True
)

# 运行
deployer.run(frequency=50.0)
```

---

## ⚙️ 配置说明

### 微调配置 (`configs/gr00t_finetune.yaml`)

```yaml
model:
  model_name: "gr00t_n1.5"
  freeze_backbone: false  # false=端到端微调, true=只微调输出层

training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4  # 微调使用较小学习率
```

**微调策略:**
- **数据少 (<20 episodes)**: `freeze_backbone: true`
- **数据多 (>30 episodes)**: `freeze_backbone: false`

---

## 🐛 常见问题

### Q: 如何获取GR00T预训练模型？

A: 模型会自动从官方仓库下载。如果下载失败，可以：
1. 手动下载checkpoint并指定路径
2. 参考 [Isaac-GR00T官方文档](https://github.com/NVIDIA/Isaac-GR00T)

### Q: 内存不足怎么办？

A: 
- 减小`batch_size`（在配置文件中）
- 使用`freeze_backbone: true`只微调输出层
- 使用更小的模型

### Q: 动作维度不匹配？

A: 模型会自动替换输出层以适应新的动作空间。确保数据预处理正确。

---

## 📖 更多资源

- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [宇树机器人SDK](https://github.com/unitreerobotics/unitree_sdk2_python)
- [项目完整工作流程指南](docs/GR00T完整工作流程指南.md)

---

<div align="center">
  <p><strong>开始您的机器人学习之旅！</strong></p>
</div>
