#!/bin/bash

# 定义服务启动命令
SERVICE_CMD="python app.py"

# 定义日志文件
LOG_FILE="log.out"

# 检查是否已有服务进程在运行（基于命令匹配）
PID=$(ps aux | grep "$SERVICE_CMD" | grep -v grep | awk '{print $2}')

if [ ! -z "$PID" ]; then
    echo "正在停止现有的服务进程 (PID: $PID)..."
    kill -9 $PID
    sleep 2  # 等待进程完全停止
fi

# 使用nohup后台启动服务，并重定向日志
echo "启动服务: $SERVICE_CMD"
nohup $SERVICE_CMD > $LOG_FILE 2>&1 &

# 输出启动信息
echo "服务已启动在后台。日志输出到: $LOG_FILE"
echo "使用 'tail -f $LOG_FILE' 查看实时日志。"