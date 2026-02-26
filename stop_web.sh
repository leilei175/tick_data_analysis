#!/bin/bash

PID_FILE="flask.pid"

if [ ! -f "$PID_FILE" ]; then
	    echo "未找到PID文件，服务可能未运行"
	        exit 1
fi

PID=$(cat $PID_FILE)

if ps -p $PID > /dev/null 2>&1; then
	    kill $PID
	        echo "服务已停止 (PID=$PID)"
	else
		    echo "进程不存在"
fi

rm -f $PID_FILE
