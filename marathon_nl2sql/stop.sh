#!/bin/bash

# 查找 app.py 进程的 PID（使用 [a]pp.py 过滤以避免匹配 grep 自身）
PID=$(ps aux | grep '[a]pp.py' | awk '{print $2}')

# 检查是否找到 PID
if [ ! -z "$PID" ]; then
    # 强制终止进程
    kill -9 $PID
    echo "进程（PID: $PID）已成功终止。"
else
    echo "未找到运行中的 app.py 进程。"
fi