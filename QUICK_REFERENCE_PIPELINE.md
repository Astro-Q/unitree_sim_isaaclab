# Unitree 双臂5指灵巧手全流程工程 - 快速参考

## 🚀 常用命令速查

### 1. 仿真运行

```bash
# G1 + Dex3 抓取圆柱体
python unitree_dual_arm_pipeline.py simulation \
    --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
    --robot_type g129 \
    --enable_dex3_dds \
    --enable_cameras
```

### 2. 数据采集

```bash
python unitree_dual_arm_pipeline.py collect \
    --mode teleop \
    --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
    --robot_type g129 \
    --enable_dex3_dds \
    --output_dir ./collected_data
```

### 3. 数据处理

```bash
python unitree_dual_arm_pipeline.py process \
    --data_path ./collected_data \
    --stats \
    --convert \
    --output_dir ./training_data
```

### 4. 模型训练

```bash
python unitree_dual_arm_pipeline.py train \
    --data_path ./training_data \
    --output_dir ./models \
    --epochs 100
```

### 5. 模型部署

```bash
python unitree_dual_arm_pipeline.py deploy \
    --model_path ./models/best_model.pth \
    --target simulation
```

## 📋 任务名称速查表

### G1机器人任务

| 任务类型 | Dex1 | Dex3 | Inspire |
|---------|------|------|---------|
| 抓取圆柱体 | `Isaac-PickPlace-Cylinder-G129-Dex1-Joint` | `Isaac-PickPlace-Cylinder-G129-Dex3-Joint` | `Isaac-PickPlace-Cylinder-G129-Inspire-Joint` |
| 抓取红色方块 | `Isaac-PickPlace-RedBlock-G129-Dex1-Joint` | `Isaac-PickPlace-RedBlock-G129-Dex3-Joint` | `Isaac-PickPlace-RedBlock-G129-Inspire-Joint` |
| 堆叠方块 | `Isaac-Stack-RgyBlock-G129-Dex1-Joint` | `Isaac-Stack-RgyBlock-G129-Dex3-Joint` | `Isaac-Stack-RgyBlock-G129-Inspire-Joint` |

更多详细信息请参考 `PIPELINE_GUIDE.md`
