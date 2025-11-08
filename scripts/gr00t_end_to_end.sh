#!/bin/bash
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

# GR00T端到端工作流程脚本
# 从数据采集到模型部署的完整流程

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
ROBOT_TYPE="g1"
EFFECTOR="inspire"
TASK="Isaac-PickPlace-Cylinder-G129-Inspire-Joint"
DATA_DIR="./teleoperate_data"
OUTPUT_DIR="./models/gr00t_${ROBOT_TYPE}_${EFFECTOR}"
EPOCHS=50
FREEZE_BACKBONE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --robot-type)
            ROBOT_TYPE="$2"
            shift 2
            ;;
        --effector)
            EFFECTOR="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --freeze-backbone)
            FREEZE_BACKBONE=true
            shift
            ;;
        -h|--help)
            echo "GR00T端到端工作流程脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --robot-type TYPE     机器人类型 (g1, h1_2) [默认: g1]"
            echo "  --effector TYPE        执行器类型 (dex1, dex3, inspire) [默认: inspire]"
            echo "  --task TASK            任务名称 [默认: Isaac-PickPlace-Cylinder-G129-Inspire-Joint]"
            echo "  --data-dir DIR         数据目录 [默认: ./teleoperate_data]"
            echo "  --output-dir DIR       输出目录 [默认: ./models/gr00t_<robot>_<effector>]"
            echo "  --epochs N             训练轮数 [默认: 50]"
            echo "  --freeze-backbone      冻结backbone（只微调输出层）"
            echo "  -h, --help             显示帮助信息"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "GR00T端到端工作流程"
echo "=========================================="
echo "机器人类型: $ROBOT_TYPE"
echo "执行器: $EFFECTOR"
echo "任务: $TASK"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "训练轮数: $EPOCHS"
echo "冻结backbone: $FREEZE_BACKBONE"
echo "=========================================="
echo ""

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}警告: 数据目录不存在: $DATA_DIR${NC}"
    echo "请先进行数据采集"
    exit 1
fi

# 步骤1: 数据预处理
echo -e "${GREEN}[步骤1] 数据预处理${NC}"
python training/data_preprocessing.py \
    --data_dirs "$DATA_DIR" \
    --output_dir "./processed_data" \
    --train_ratio 0.8 \
    --normalize

if [ $? -ne 0 ]; then
    echo -e "${RED}数据预处理失败${NC}"
    exit 1
fi

# 步骤2: GR00T微调
echo -e "${GREEN}[步骤2] GR00T模型微调${NC}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 准备配置文件
CONFIG_FILE="$OUTPUT_DIR/config.yaml"
cat > "$CONFIG_FILE" << EOF
model:
  model_name: "gr00t_n1.5"
  pretrained_checkpoint: null
  freeze_backbone: $FREEZE_BACKBONE

training:
  epochs: $EPOCHS
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-4
  min_lr: 1e-6
  grad_clip: 1.0

data:
  data_dir: "./processed_data"
  use_images: true
  image_size: [224, 224]
  num_workers: 4

device: "cuda"
seed: 42
EOF

# 运行训练
python gr00t/train_gr00t.py \
    --config "$CONFIG_FILE" \
    --data_dir "./processed_data" \
    --output_dir "$OUTPUT_DIR" \
    --freeze_backbone $FREEZE_BACKBONE

if [ $? -ne 0 ]; then
    echo -e "${RED}模型微调失败${NC}"
    exit 1
fi

# 步骤3: 转换为ONNX
echo -e "${GREEN}[步骤3] 转换为ONNX格式${NC}"
python gr00t/convert_to_onnx.py \
    --checkpoint "$OUTPUT_DIR/best_model.pth" \
    --output "$OUTPUT_DIR/model.onnx" \
    --state_dim 512 \
    --batch_size 1

if [ $? -ne 0 ]; then
    echo -e "${RED}ONNX转换失败${NC}"
    exit 1
fi

# 步骤4: 完成
echo ""
echo "=========================================="
echo -e "${GREEN}✓ 工作流程完成！${NC}"
echo "=========================================="
echo "模型文件:"
echo "  - PyTorch: $OUTPUT_DIR/best_model.pth"
echo "  - ONNX: $OUTPUT_DIR/model.onnx"
echo ""
echo "下一步:"
echo "  1. 在仿真中测试模型:"
echo "     python sim_main.py --task $TASK --action_source policy --model_path $OUTPUT_DIR/model.onnx"
echo ""
echo "  2. 部署到真实机器人:"
echo "     python gr00t/deploy_gr00t.py --model_path $OUTPUT_DIR/model.onnx --robot_type $ROBOT_TYPE --robot_ip <IP> --effector $EFFECTOR"
echo "=========================================="
