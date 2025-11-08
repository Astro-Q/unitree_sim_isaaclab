# API 文档

## 训练模块 API

### 数据预处理 (`training/data_preprocessing.py`)

#### DataPreprocessor

数据预处理器类，负责加载、处理和保存机器人数据。

**初始化参数：**
```python
DataPreprocessor(
    data_dirs: List[str],          # 数据目录列表
    output_dir: str,                # 输出目录
    split_ratio: float = 0.9,       # 训练集比例
    normalize: bool = True,         # 是否标准化
    augment: bool = False           # 是否数据增强
)
```

**主要方法：**

- `load_episode(episode_dir: Path) -> Dict`: 加载单个episode数据
- `collect_all_episodes() -> List[Dict]`: 收集所有episode数据
- `compute_statistics(episodes: List[Dict])`: 计算数据统计信息
- `normalize_episode(episode: Dict) -> Dict`: 标准化单个episode
- `augment_episode(episode: Dict) -> List[Dict]`: 数据增强
- `process()`: 执行完整的预处理流程

**使用示例：**
```python
from training.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(
    data_dirs=["./data1", "./data2"],
    output_dir="./processed_data",
    split_ratio=0.9,
    normalize=True,
    augment=True
)

preprocessor.process()
```

---

### 行为克隆训练 (`training/train_bc.py`)

#### BCPolicy

策略网络模型。

**初始化参数：**
```python
BCPolicy(
    state_dim: int,                 # 状态维度
    action_dim: int,                # 动作维度
    hidden_dims: List[int] = [512, 512, 256]  # 隐藏层维度
)
```

**方法：**
- `forward(state: torch.Tensor) -> torch.Tensor`: 前向传播

---

#### BCTrainer

训练器类。

**初始化参数：**
```python
BCTrainer(config: Dict)
```

**配置字典必需字段：**
- `data_dir`: 数据目录
- `output_dir`: 输出目录
- `batch_size`: 批次大小
- `epochs`: 训练轮数
- `learning_rate`: 学习率

**主要方法：**
- `train_epoch() -> float`: 训练一个epoch
- `test_epoch() -> float`: 测试一个epoch
- `save_checkpoint(epoch, test_loss, is_best)`: 保存检查点
- `train()`: 执行完整训练过程

**使用示例：**
```python
from training.train_bc import BCTrainer

config = {
    'data_dir': './processed_data',
    'output_dir': './models/my_model',
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 3e-4,
    'device': 'cuda'
}

trainer = BCTrainer(config)
trainer.train()
```

---

## 部署模块 API

### 模型转换 (`deployment/convert_to_onnx.py`)

#### convert_to_onnx

将PyTorch模型转换为ONNX格式。

**参数：**
```python
convert_to_onnx(
    model_path: str,                # PyTorch模型路径
    output_path: str,               # ONNX输出路径
    state_dim: int,                 # 状态维度
    action_dim: int,                # 动作维度
    opset_version: int = 14,        # ONNX opset版本
    simplify: bool = True           # 是否简化模型
)
```

**使用示例：**
```python
from deployment.convert_to_onnx import convert_to_onnx

convert_to_onnx(
    model_path="./models/my_model/best_model.pth",
    output_path="./models/my_model/model.onnx",
    state_dim=60,
    action_dim=43,
    opset_version=14,
    simplify=True
)
```

---

### 机器人部署 (`deployment/deploy_to_robot.py`)

#### RobotDeployer

机器人部署器类。

**初始化参数：**
```python
RobotDeployer(
    model_path: str,                # ONNX模型路径
    robot_type: str,                # 机器人类型 (g1, h1_2)
    robot_ip: str,                  # 机器人IP地址
    effector: str,                  # 执行器类型 (dex1, dex3, inspire)
    statistics_path: str = None,    # 数据统计文件路径
    safety_mode: bool = True        # 是否启用安全模式
)
```

**主要方法：**
- `get_robot_state() -> np.ndarray`: 获取机器人状态
- `predict_action(state) -> np.ndarray`: 预测动作
- `apply_safety_constraints(action) -> np.ndarray`: 应用安全约束
- `send_action_to_robot(action)`: 发送动作到机器人
- `run(duration, frequency)`: 运行部署循环

**使用示例：**
```python
from deployment.deploy_to_robot import RobotDeployer

deployer = RobotDeployer(
    model_path="./models/my_model/model.onnx",
    robot_type="g1",
    robot_ip="192.168.123.10",
    effector="inspire",
    statistics_path="./processed_data/statistics.pkl",
    safety_mode=True
)

# 运行60秒，50Hz控制频率
deployer.run(duration=60.0, frequency=50.0)
```

---

## 数据结构

### Episode 数据格式

每个episode目录包含以下文件：

```
episode_00000/
├── metadata.json          # 元数据
├── actions.npy            # 动作数组 (T, action_dim)
├── observations.npy       # 观测数组 (T, obs_dim)
├── states.npy             # 状态数组 (T, state_dim)
└── images/                # 图像目录
    ├── front_camera/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── left_wrist_camera/
    └── right_wrist_camera/
```

**metadata.json 格式：**
```json
{
  "task_name": "Isaac-PickPlace-Cylinder-G129-Inspire-Joint",
  "robot_type": "g1",
  "effector": "inspire",
  "timestamp": "2025-01-08T10:30:00",
  "duration": 15.5,
  "num_steps": 775,
  "success": true,
  "episode_id": 0
}
```

---

### 统计信息格式 (`statistics.pkl`)

```python
{
    'action_mean': np.ndarray,      # 动作均值 (action_dim,)
    'action_std': np.ndarray,       # 动作标准差 (action_dim,)
    'state_mean': np.ndarray,       # 状态均值 (state_dim,)
    'state_std': np.ndarray         # 状态标准差 (state_dim,)
}
```

---

### 模型检查点格式 (`.pth`)

```python
{
    'epoch': int,                    # 训练轮数
    'model_state_dict': OrderedDict, # 模型权重
    'optimizer_state_dict': dict,    # 优化器状态
    'scheduler_state_dict': dict,    # 学习率调度器状态
    'test_loss': float,              # 测试损失
    'config': dict                   # 训练配置
}
```

---

## 命令行工具

### 数据预处理

```bash
python training/data_preprocessing.py \
  --data_dirs "dir1,dir2,dir3" \
  --output_dir "./processed_data" \
  --split_ratio 0.9 \
  --normalize \
  --augment
```

### 模型训练

```bash
python training/train_bc.py \
  --config configs/bc_inspire_hand.yaml \
  --data_dir "./processed_data" \
  --output_dir "./models/my_model" \
  --epochs 100 \
  --batch_size 64 \
  --learning_rate 3e-4 \
  --device cuda
```

### 模型转换

```bash
python deployment/convert_to_onnx.py \
  --model_path "./models/my_model/best_model.pth" \
  --output_path "./models/my_model/model.onnx" \
  --state_dim 60 \
  --action_dim 43 \
  --opset_version 14
```

### 真机部署

```bash
python deployment/deploy_to_robot.py \
  --model_path "./models/my_model/model.onnx" \
  --robot_type g1 \
  --robot_ip 192.168.123.10 \
  --effector inspire \
  --statistics_path "./processed_data/statistics.pkl" \
  --safety_mode \
  --frequency 50.0 \
  --duration 60.0
```

---

## 配置文件格式

### 训练配置 (`configs/bc_inspire_hand.yaml`)

```yaml
model:
  type: "BCPolicy"
  hidden_dims: [512, 512, 256]
  activation: "relu"
  dropout: 0.1

training:
  epochs: 100
  batch_size: 64
  learning_rate: 3e-4
  weight_decay: 1e-4
  
data:
  data_dir: "./processed_data"
  use_images: false
  num_workers: 4

device: "cuda"
seed: 42
```

### 部署配置 (`configs/deployment_config.yaml`)

```yaml
robot:
  type: "g1"
  ip: "192.168.123.10"
  effector: "inspire"

model:
  path: "./models/bc_inspire/model.onnx"
  type: "onnx"

control:
  frequency: 50.0
  control_mode: "position"
  
  safety:
    enabled: true
    max_action_delta: 0.1
    action_limits: [-1.0, 1.0]
```

---

## 常见问题

### Q: 如何确定状态维度和动作维度？

A: 可以通过以下方式获取：

```python
import numpy as np

# 加载一个episode
episode_dir = "./processed_data/train/episode_00000"
states = np.load(episode_dir + "/states.npy")
actions = np.load(episode_dir + "/actions.npy")

state_dim = states.shape[1]
action_dim = actions.shape[1]

print(f"状态维度: {state_dim}")
print(f"动作维度: {action_dim}")
```

### Q: 训练时GPU内存不足怎么办？

A: 尝试以下方法：
1. 减小批次大小：`--batch_size 32`
2. 减小模型大小：修改配置文件中的 `hidden_dims`
3. 关闭数据增强
4. 使用梯度累积

### Q: 如何监控训练过程？

A: 可以使用 Weights & Biases：

```python
# 在配置文件中启用
logging:
  use_wandb: true
  wandb_project: "my_project"
```

### Q: 模型在仿真中表现好，但真机效果差？

A: 这是Sim2Real问题，可以尝试：
1. 使用域随机化增强训练数据
2. 在真机上采集少量数据进行微调
3. 调整安全参数和控制频率
4. 使用更保守的动作限制

---

## 版本信息

- **当前版本**: 1.0.0
- **最后更新**: 2025-11-08
- **兼容性**: Python 3.8+, PyTorch 1.13+, ONNX 1.14+
