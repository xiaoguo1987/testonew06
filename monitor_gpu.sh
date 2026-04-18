#!/bin/bash
# monitor_gpu.sh - 监控GPU利用率和训练进度

echo "=========================================="
echo "GPU和训练监控"
echo "=========================================="
echo ""

# 检查训练进程
TRAIN_PID=$(pgrep -f "train.py")
if [ -z "$TRAIN_PID" ]; then
    echo "⚠️  未找到训练进程！"
    echo "   请先启动训练: python train.py --mode train --model-type ppo ..."
    exit 1
else
    echo "✓ 训练进程运行中 (PID: $TRAIN_PID)"
fi

echo ""
echo "=========================================="
echo "GPU状态 (每2秒刷新，按Ctrl+C停止)"
echo "=========================================="
echo ""

# 持续监控
while true; do
    clear
    echo "=========================================="
    echo "GPU监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # GPU利用率
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1)
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    GPU_PWR=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
    
    echo "GPU利用率: ${GPU_UTIL}%"
    echo "显存使用: ${GPU_MEM} MB"
    echo "GPU温度: ${GPU_TEMP}°C"
    echo "功耗: ${GPU_PWR} W"
    echo ""
    
    # 训练进程状态
    CPU_USAGE=$(top -b -n 1 -p $TRAIN_PID | grep $TRAIN_PID | awk '{print $9}')
    MEM_USAGE=$(top -b -n 1 -p $TRAIN_PID | grep $TRAIN_PID | awk '{print $10}')
    
    echo "训练进程 (PID: $TRAIN_PID)"
    echo "  CPU使用率: ${CPU_USAGE}%"
    echo "  内存使用: ${MEM_USAGE}%"
    echo ""
    
    # 判断状态
    if [ "$GPU_UTIL" -gt 50 ]; then
        echo "✅ GPU正在工作！(利用率: ${GPU_UTIL}%)"
    elif [ "$GPU_UTIL" -gt 10 ]; then
        echo "⚠️  GPU利用率较低 (${GPU_UTIL}%)，可能在准备数据"
    else
        echo "❌ GPU未使用 (0%)，可能卡住了或在初始化"
    fi
    
    echo ""
    echo "------------------------------------------"
    echo "最新的训练日志:"
    echo "------------------------------------------"
    
    # 显示最新日志
    LOG_FILE=$(ls -t logs/poker_ai_*.log 2>/dev/null | head -1)
    if [ -n "$LOG_FILE" ]; then
        tail -n 5 "$LOG_FILE"
    else
        echo "  (未找到日志文件)"
    fi
    
    echo ""
    echo "=========================================="
    echo "按 Ctrl+C 停止监控"
    echo "=========================================="
    
    sleep 2
done
