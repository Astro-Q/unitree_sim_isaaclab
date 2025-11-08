# Unitree 双臂抓取仿真快速参考

## 📁 文件说明

### 主要文档
- **`GUIDE_双臂抓取仿真与微调.md`** - 完整的使用指南，包含详细步骤和说明

### 实用脚本
- **`quick_start.py`** - 快速启动脚本，简化仿真启动命令
- **`data_processor.py`** - 数据处理工具，用于数据分析和格式转换

## 🚀 快速开始

### 1. 启动仿真（最简单方式）

```bash
# G1 + Dex1 抓取圆柱体
python quick_start.py --robot g1 --effector dex1 --task cylinder

# G1 + Dex3 抓取红色方块
python quick_start.py --robot g1 --effector dex3 --task redblock

# G1 + Inspire 堆叠方块
python quick_start.py --robot g1 --effector inspire --task stack
```

### 2. 数据回放

```bash
python quick_start.py \
  --robot g1 \
  --effector dex1 \
  --task cylinder \
  --replay \
  --data_path /path/to/your/data
```

### 3. 数据生成

```bash
python quick_start.py \
  --robot g1 \
  --effector dex1 \
  --task cylinder \
  --generate \
  --data_path /path/to/original/data
```

## 📊 数据处理

### 查看数据集统计

```bash
python data_processor.py --data_root /path/to/data --stats
```

### 转换为训练格式

```bash
python data_processor.py \
  --data_root /path/to/data \
  --convert \
  --output_dir ./training_data
```

### 可视化 Episode

```bash
python data_processor.py \
  --data_root /path/to/data \
  --visualize \
  --episode_idx 0 \
  --output_video episode_0.mp4
```

## 🎯 常用命令对照表

| 功能 | 快速脚本 | 原始命令 |
|------|---------|---------|
| G1+Dex1抓取圆柱体 | `python quick_start.py --robot g1 --effector dex1 --task cylinder` | `python sim_main.py --device cuda --enable_cameras --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint --enable_dex1_dds --robot_type g129` |
| G1+Dex3抓取红色方块 | `python quick_start.py --robot g1 --effector dex3 --task redblock` | `python sim_main.py --device cuda --enable_cameras --task Isaac-PickPlace-RedBlock-G129-Dex3-Joint --enable_dex3_dds --robot_type g129` |
| G1+Inspire堆叠方块 | `python quick_start.py --robot g1 --effector inspire --task stack` | `python sim_main.py --device cuda --enable_cameras --task Isaac-Stack-RgyBlock-G129-Inspire-Joint --enable_inspire_dds --robot_type g129` |
| H1-2+Inspire抓取圆柱体 | `python quick_start.py --robot h1-2 --effector inspire --task cylinder` | `python sim_main.py --device cuda --enable_cameras --task Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint --enable_inspire_dds --robot_type h1_2` |

## 📖 详细文档

查看完整指南：
```bash
cat GUIDE_双臂抓取仿真与微调.md
# 或
less GUIDE_双臂抓取仿真与微调.md
```

## 🔗 相关链接

- [完整指南](./GUIDE_双臂抓取仿真与微调.md)
- [项目 README](./README_zh-CN.md)
- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/)
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2_python)

## 💡 提示

1. **首次运行**：确保已安装所有依赖并下载资产文件
2. **性能优化**：使用 `--headless` 模式可提高运行速度
3. **数据采集**：需要配合 [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) 项目使用
4. **DDS 通信**：确保使用相同的 DDS 通道（Channel 1）

## ❓ 获取帮助

```bash
# 快速启动脚本帮助
python quick_start.py --help

# 数据处理脚本帮助
python data_processor.py --help

# 原始仿真脚本帮助
python sim_main.py --help
```
