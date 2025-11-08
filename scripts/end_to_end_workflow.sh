#!/bin/bash
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

# 端到端工作流程脚本
# 完成从数据采集到模型部署的全流程

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo ""
    echo "================================"
    echo "$1"
    echo "================================"
}

# 配置参数（可根据需要修改）
ROBOT_TYPE="g1"
EFFECTOR="inspire"
TASK="Isaac-PickPlace-Cylinder-G129-Inspire-Joint"
TELEOPERATE_DATA_DIR="./teleoperate_data"
AUGMENTED_DATA_DIR="./augmented_data"
PROCESSED_DATA_DIR="./processed_data"
MODEL_OUTPUT_DIR="./models/bc_inspire"
CONFIG_FILE="./configs/bc_inspire_hand.yaml"

# 解析命令行参数
SKIP_DATA_COLLECTION=false
SKIP_TRAINING=false
SKIP_DEPLOYMENT=false

while [[ $# -gt 0 ]]; do
    case $1 in
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
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --skip-data-collection    跳过数据采集步骤"
            echo "  --skip-training           跳过训练步骤"
            echo "  --skip-deployment         跳过部署步骤"
            echo "  --robot-type TYPE         机器人类型 (默认: g1)"
            echo "  --effector TYPE           执行器类型 (默认: inspire)"
            echo "  --task TASK               任务名称"
            echo "  -h, --help                显示帮助信息"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

print_info "端到端工作流程开始"
print_info "机器人类型: $ROBOT_TYPE"
print_info "执行器: $EFFECTOR"
print_info "任务: $TASK"

# ========================================
# 步骤1: 数据采集（遥操作）
# ========================================
if [ "$SKIP_DATA_COLLECTION" = false ]; then
    print_step "步骤1: 数据采集"
    
    print_warning "请确保已启动遥操作系统 (xr_teleoperate)"
    print_info "启动仿真环境接收遥操作数据..."
    
    # 启动仿真（这里使用后台运行，实际使用时可能需要手动操作）
    # python sim_main.py \
    #     --device cuda \
    #     --enable_cameras \
    #     --task $TASK \
    #     --enable_${EFFECTOR}_dds \
    #     --robot_type $ROBOT_TYPE
    
    print_info "数据采集环境已准备好"
    print_warning "请在遥操作系统中采集20-50个成功的演示"
    read -p "按Enter键继续到下一步..." dummy
else
    print_info "跳过数据采集步骤"
fi

# ========================================
# 步骤2: 数据增强
# ========================================
print_step "步骤2: 数据增强"

if [ ! -d "$TELEOPERATE_DATA_DIR" ]; then
    print_error "未找到遥操作数据目录: $TELEOPERATE_DATA_DIR"
    exit 1
fi

print_info "生成增强数据..."
python sim_main.py \
    --device cuda \
    --enable_cameras \
    --task $TASK \
    --enable_${EFFECTOR}_dds \
    --robot_type $ROBOT_TYPE \
    --replay_data \
    --file_path "$TELEOPERATE_DATA_DIR" \
    --generate_data \
    --generate_data_dir "$AUGMENTED_DATA_DIR" \
    --modify_light \
    --modify_camera \
    --headless

print_info "数据增强完成"

# ========================================
# 步骤3: 数据预处理
# ========================================
print_step "步骤3: 数据预处理"

print_info "合并和预处理数据..."
python training/data_preprocessing.py \
    --data_dirs "$TELEOPERATE_DATA_DIR,$AUGMENTED_DATA_DIR" \
    --output_dir "$PROCESSED_DATA_DIR" \
    --split_ratio 0.9 \
    --normalize \
    --augment

print_info "数据预处理完成"

# ========================================
# 步骤4: 模型训练
# ========================================
if [ "$SKIP_TRAINING" = false ]; then
    print_step "步骤4: 模型训练"
    
    print_info "开始训练模型..."
    python training/train_bc.py \
        --config "$CONFIG_FILE" \
        --data_dir "$PROCESSED_DATA_DIR" \
        --output_dir "$MODEL_OUTPUT_DIR" \
        --epochs 100 \
        --batch_size 64 \
        --learning_rate 3e-4 \
        --device cuda
    
    print_info "模型训练完成"
else
    print_info "跳过训练步骤"
fi

# ========================================
# 步骤5: 模型评估（在仿真中）
# ========================================
print_step "步骤5: 模型评估"

print_info "在仿真环境中评估模型..."

# 首先需要知道状态和动作维度
STATE_DIM=60  # 根据实际情况调整
ACTION_DIM=43  # 根据实际情况调整

# 转换为ONNX
print_info "转换模型为ONNX格式..."
python deployment/convert_to_onnx.py \
    --model_path "$MODEL_OUTPUT_DIR/best_model.pth" \
    --output_path "$MODEL_OUTPUT_DIR/model.onnx" \
    --state_dim $STATE_DIM \
    --action_dim $ACTION_DIM \
    --opset_version 14

print_info "模型转换完成"

# 在仿真中测试
print_info "在仿真中测试模型..."
python sim_main.py \
    --device cuda \
    --enable_cameras \
    --task $TASK \
    --enable_${EFFECTOR}_dds \
    --robot_type $ROBOT_TYPE \
    --action_source policy \
    --model_path "$MODEL_OUTPUT_DIR/model.onnx"

read -p "评估完成。按Enter键继续..." dummy

# ========================================
# 步骤6: 真机部署（可选）
# ========================================
if [ "$SKIP_DEPLOYMENT" = false ]; then
    print_step "步骤6: 真机部署"
    
    print_warning "即将部署到真实机器人"
    print_warning "请确保："
    print_warning "  1. 机器人已开机并连接网络"
    print_warning "  2. 安全区域已清空"
    print_warning "  3. 紧急停止按钮可用"
    
    read -p "确认继续部署？(y/n): " confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        ROBOT_IP="192.168.123.10"  # 修改为实际机器人IP
        
        print_info "部署模型到真实机器人..."
        python deployment/deploy_to_robot.py \
            --model_path "$MODEL_OUTPUT_DIR/model.onnx" \
            --robot_type $ROBOT_TYPE \
            --robot_ip $ROBOT_IP \
            --effector $EFFECTOR \
            --statistics_path "$PROCESSED_DATA_DIR/statistics.pkl" \
            --safety_mode \
            --frequency 50.0
        
        print_info "部署完成"
    else
        print_info "跳过真机部署"
    fi
else
    print_info "跳过部署步骤"
fi

# ========================================
# 完成
# ========================================
print_step "工作流程完成！"

print_info "输出文件位置："
print_info "  原始数据: $TELEOPERATE_DATA_DIR"
print_info "  增强数据: $AUGMENTED_DATA_DIR"
print_info "  处理数据: $PROCESSED_DATA_DIR"
print_info "  训练模型: $MODEL_OUTPUT_DIR"

print_info "后续步骤："
print_info "  1. 查看训练日志和模型性能"
print_info "  2. 在仿真中进一步测试"
print_info "  3. 调整超参数重新训练"
print_info "  4. 部署到真实机器人"

print_info "感谢使用！"
