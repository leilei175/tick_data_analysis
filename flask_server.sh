#!/bin/bash
# 高频因子分析平台 - Web服务管理脚本

cd "$(dirname "$0")"

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
# 初始化conda环境
# ==========================
init_conda() {
    __conda_setup="$('/home/zxx/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
        conda activate quant 2>/dev/null || true
    fi
}

# ==========================
# 检查是否运行
# ==========================
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        if ps -p $PID > /dev/null 2>&1; then
            return 0
        else
            rm -f $PID_FILE
            return 1
        fi
    else
        return 1
    fi
}

# ==========================
# 启动服务
# ==========================
start() {
    if is_running; then
        echo "⚠ 服务已在运行 (PID=$(cat $PID_FILE))"
        exit 1
    fi

    mkdir -p ${LOG_DIR}
    init_conda

    echo "🚀 正在启动服务..."
    echo "日志文件: ${LOG_FILE}"

    nohup python -m ${APP_MODULE} >> ${LOG_FILE} 2>&1 &

    echo $! > ${PID_FILE}
    sleep 1

    if is_running; then
        echo "✅ 启动成功"
        echo "访问地址: http://localhost:${PORT}"
        echo "PID: $(cat $PID_FILE)"
    else
        echo "❌ 启动失败，请查看日志"
    fi
}

# ==========================
# 前台运行（供systemd等进程管理器使用）
# ==========================
run() {
    mkdir -p "${LOG_DIR}"
    init_conda

    echo "前台运行服务，端口: ${PORT}"
    echo "日志文件: ${LOG_FILE}"

    exec python -m "${APP_MODULE}" >> "${LOG_FILE}" 2>&1
}

# ==========================
# 停止服务
# ==========================
stop() {
    if is_running; then
        PID=$(cat $PID_FILE)
        echo "🛑 正在停止服务 (PID=$PID)..."
        kill $PID
        sleep 1
        rm -f $PID_FILE
        echo "✅ 已停止"
    else
        echo "⚠ 服务未运行"
    fi
}

# ==========================
# 重启服务
# ==========================
restart() {
    stop
    sleep 1
    start
}

# ==========================
# 状态检查
# ==========================
status() {
    if is_running; then
        echo "✅ 服务正在运行 (PID=$(cat $PID_FILE))"
    else
        echo "❌ 服务未运行"
    fi
}

# ==========================
# 主逻辑
# ==========================
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    run)
        run
        ;;
    status)
        status
        ;;
    *)
        echo "用法: $0 {start|stop|restart|run|status}"
        exit 1
        ;;
esac
