#!/bin/bash
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

# Isaac-GR00T完整工作流程脚本
# 从数据采集到模型部署的全流程自动化

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/configs/gr00t_config.yaml"
DATA_DIR="${PROJECT_ROOT}/data/teleoperate"
PROCESSED_DATA_DIR="${PROJECT_ROOT}/data/processed"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/gr00t_training"
MODEL_DIR="${PROJECT_ROOT}/models/gr00t"

# 解析命令行参数
ROBOT_TYPE="g1"
EFFECTOR="inspire"
TASK_NAME="Isaac-PickPlace-Cylinder-G129-Inspire-Joint"
SKIP_DATA_COLLECTION=false
SKIP_TRAINING=false
SKIP_DEPLOYMENT=false

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --robot-type TYPE        机器人类型 (g1, h1_2) [默认: g1]"
    echo "  --effector TYPE          执行器类型 (dex1, dex3, inspire) [默认: inspire]"
    echo "  --task NAME              任务名称 [默认: Isaac-PickPlace-Cylinder-G129-Inspire-Joint]"
    echo "  --data-dir DIR           数据目录 [默认: ./data/teleoperate]"
    echo "  --skip-data-collection   跳过数据采集步骤"
    echo "  --skip-training          跳过训练步骤"
    echo "  --skip-deployment        跳过部署步骤"
    echo "  -h, --help               显示帮助信息"
    exit 1
}

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
            TASK_NAME="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --skip-data-collection)
            SKIP_DATA_COLLECTION=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-deployment)
            SKIP_DEPLOYMENT=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知选项: $1"
            usage
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Isaac-GR00T完整工作流程${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "配置:"
echo "  项目根目录: ${PROJECT_ROOT}"
echo "  配置文件: ${CONFIG_FILE}"
echo "  数据目录: ${DATA_DIR}"
echo "  机器人类型: ${ROBOT_TYPE}"
echo "  执行器: ${EFFECTOR}"
echo "  任务: ${TASK_NAME}"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到python3${NC}"
    exit 1
fi

# 步骤1: 数据采集
if [ "$SKIP_DATA_COLLECTION" = false ]; then
    echo -e "${YELLOW}[步骤1/5] 数据采集${NC}"
    echo "请使用遥操作工具采集数据，数据应保存在: ${DATA_DIR}"
    echo "按Enter继续（如果数据已准备好）..."
    read
else
    echo -e "${YELLOW}[步骤1/5] 数据采集 (已跳过)${NC}"
fi

# 步骤2: 数据预处理
echo -e "${YELLOW}[步骤2/5] 数据预处理${NC}"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR)" ]; then
    echo -e "${RED}错误: 数据目录不存在或为空: ${DATA_DIR}${NC}"
    exit 1
fi

python3 "${PROJECT_ROOT}/gr00t_integration/preprocess_data.py" \
    --input_dir "${DATA_DIR}" \
    --output_dir "${PROCESSED_DATA_DIR}" \
    --train_split 0.9 \
    --normalize

if [ $? -ne 0 ]; then
    echo -e "${RED}数据预处理失败${NC}"
    exit 1
fi
echo -e "${GREEN}数据预处理完成${NC}"
echo ""

# 步骤3: 模型训练
if [ "$SKIP_TRAINING" = false ]; then
    echo -e "${YELLOW}[步骤3/5] 模型训练${NC}"
    
    # 创建输出目录
    mkdir -p "${OUTPUT_DIR}"
    
    python3 "${PROJECT_ROOT}/gr00t_integration/train_gr00t.py" \
        --config "${CONFIG_FILE}" \
        --data_dir "${PROCESSED_DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}模型训练失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}模型训练完成${NC}"
    echo ""
else
    echo -e "${YELLOW}[步骤3/5] 模型训练 (已跳过)${NC}"
fi

# 步骤4: 模型转换
echo -e "${YELLOW}[步骤4/5] 模型转换${NC}"
CHECKPOINT_PATH="${OUTPUT_DIR}/best_model.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH="${OUTPUT_DIR}/latest_checkpoint.pth"
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}错误: 未找到模型检查点${NC}"
    exit 1
fi

mkdir -p "${MODEL_DIR}"
ONNX_PATH="${MODEL_DIR}/gr00t_model.onnx"

python3 "${PROJECT_ROOT}/gr00t_integration/convert_and_deploy.py" \
    --mode convert \
    --config "${CONFIG_FILE}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output_path "${ONNX_PATH}" \
    --optimize

if [ $? -ne 0 ]; then
    echo -e "${RED}模型转换失败${NC}"
    exit 1
fi
echo -e "${GREEN}模型转换完成${NC}"
echo ""

# 步骤5: 模型测试
echo -e "${YELLOW}[步骤5/5] 模型测试${NC}"
python3 "${PROJECT_ROOT}/gr00t_integration/convert_and_deploy.py" \
    --mode test \
    --config "${CONFIG_FILE}" \
    --onnx_path "${ONNX_PATH}"

if [ $? -ne 0 ]; then
    echo -e "${RED}模型测试失败${NC}"
    exit 1
fi
echo -e "${GREEN}模型测试完成${NC}"
echo ""

# 完成
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}工作流程完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "输出文件:"
echo "  训练模型: ${OUTPUT_DIR}"
echo "  ONNX模型: ${ONNX_PATH}"
echo "  统计信息: ${PROCESSED_DATA_DIR}/statistics.pkl"
echo ""
echo "下一步:"
echo "  1. 在仿真环境中测试模型:"
echo "     python sim_main.py --task ${TASK_NAME} --action_source policy --model_path ${ONNX_PATH}"
echo ""
echo "  2. 部署到真实机器人:"
echo "     python deployment/deploy_to_robot.py --model_path ${ONNX_PATH} --robot_type ${ROBOT_TYPE} --effector ${EFFECTOR}"
echo ""
