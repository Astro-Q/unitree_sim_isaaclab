#!/bin/bash
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

# 快速开始训练脚本 - 简化版

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}宇树机器人 - 快速训练脚本${NC}"
echo ""

# 默认参数
DATA_DIR=""
OUTPUT_DIR="./models/my_model"
EPOCHS=100
BATCH_SIZE=64
LR=3e-4

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 --data <数据目录> [选项]"
            echo ""
            echo "必需参数:"
            echo "  --data DIR              数据目录"
            echo ""
            echo "可选参数:"
            echo "  --output DIR            输出目录 (默认: ./models/my_model)"
            echo "  --epochs N              训练轮数 (默认: 100)"
            echo "  --batch-size N          批次大小 (默认: 64)"
            echo "  --lr RATE               学习率 (默认: 3e-4)"
            echo "  -h, --help              显示帮助"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$DATA_DIR" ]; then
    echo "错误: 必须指定数据目录 (--data)"
    echo "使用 --help 查看帮助"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    exit 1
fi

echo "配置:"
echo "  数据目录: $DATA_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  学习率: $LR"
echo ""

# 步骤1: 数据预处理
echo -e "${GREEN}[1/3] 数据预处理${NC}"
PROCESSED_DIR="${OUTPUT_DIR}_processed_data"
python training/data_preprocessing.py \
    --data_dirs "$DATA_DIR" \
    --output_dir "$PROCESSED_DIR" \
    --split_ratio 0.9 \
    --normalize

# 步骤2: 模型训练
echo ""
echo -e "${GREEN}[2/3] 模型训练${NC}"
python training/train_bc.py \
    --config configs/bc_inspire_hand.yaml \
    --data_dir "$PROCESSED_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --device cuda

# 步骤3: 模型转换
echo ""
echo -e "${GREEN}[3/3] 模型转换${NC}"
python deployment/convert_to_onnx.py \
    --model_path "$OUTPUT_DIR/best_model.pth" \
    --output_path "$OUTPUT_DIR/model.onnx" \
    --state_dim 60 \
    --action_dim 43

echo ""
echo -e "${GREEN}训练完成！${NC}"
echo "模型保存在: $OUTPUT_DIR"
echo ""
echo "下一步:"
echo "  1. 在仿真中测试模型:"
echo "     python sim_main.py --task <任务名> --action_source policy --model_path $OUTPUT_DIR/model.onnx"
echo ""
echo "  2. 部署到真实机器人:"
echo "     python deployment/deploy_to_robot.py --model_path $OUTPUT_DIR/model.onnx ..."
