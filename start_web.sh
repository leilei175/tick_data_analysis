#!/bin/bash
# 启动高频因子分析平台Web服务（后台模式 + 按日期日志）

cd "$(dirname "$0")"

# ==========================
# 初始化conda环境
# ==========================
__conda_setup="$('/home/zxx/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
	    eval "$__conda_setup"
	        conda activate quant 2>/dev/null || true
fi

# ==========================
# 基础变量
# ==========================
APP_MODULE="factor_dashboard.app"
PORT=9999
LOG_DIR="log"
TODAY=$(date +"%Y-%m-%d")
LOG_FILE="${LOG_DIR}/${TODAY}.log"
PID_FILE="flask.pid"

# ==========================
# 创建日志目录
# ==========================
mkdir -p ${LOG_DIR}

echo "========================================"
echo "高频因子分析平台 - Web服务后台启动"
echo "========================================"
echo "Python: $(which python 2>/dev/null || echo '未找到')"
echo "端口: ${PORT}"
echo "日志文件: ${LOG_FILE}"
echo ""

# ==========================
# 如果已运行
# ==========================
if [ -f "$PID_FILE" ]; then
	    OLD_PID=$(cat $PID_FILE)
	        if ps -p $OLD_PID > /dev/null 2>&1; then
			        echo "服务已在运行中 (PID=$OLD_PID)"
				        exit 1
					    else
						            echo "检测到旧PID文件，已清理"
							            rm -f $PID_FILE
								        fi
fi

# ==========================
# 后台启动
# ==========================
echo "正在后台启动服务..."

nohup python -m ${APP_MODULE} >> ${LOG_FILE} 2>&1 &

# 记录PID
echo $! > ${PID_FILE}

sleep 1

if ps -p $(cat $PID_FILE) > /dev/null 2>&1; then
	    echo "✅ 启动成功!"
	        echo "访问地址: http://localhost:${PORT}"
		    echo "PID: $(cat $PID_FILE)"
	    else
		        echo "❌ 启动失败，请查看日志 ${LOG_FILE}"
fi

echo "========================================"
