# Flask 服务启动与开机自启

本文说明本项目 Flask 服务的启动、关闭、状态查看，以及开机自动启动的管理方法。

## 1. 服务信息

- 启动脚本：`/data1/code_git/tick_data_analysis/flask_server.sh`
- 开机自启安装脚本：`/data1/code_git/tick_data_analysis/install_flask_autostart.sh`
- 默认端口：`9999`
- 默认访问地址：`http://127.0.0.1:9999`
- 仪表盘首页：`http://127.0.0.1:9999/dashboard`

## 2. 手动启动与关闭

先进入项目目录：

```bash
cd /data1/code_git/tick_data_analysis
```

启动 Flask 服务：

```bash
./flask_server.sh start
```

停止 Flask 服务：

```bash
./flask_server.sh stop
```

重启 Flask 服务：

```bash
./flask_server.sh restart
```

查看服务状态：

```bash
./flask_server.sh status
```

## 3. 开机自动启动

安装并启用用户级 `systemd` 自启动服务：

```bash
cd /data1/code_git/tick_data_analysis
./install_flask_autostart.sh --install
```

查看自启动服务状态：

```bash
./install_flask_autostart.sh --status
```

也可以直接用 `systemd` 命令查看：

```bash
systemctl --user status tick-data-flask.service
```

重启自启动服务：

```bash
systemctl --user restart tick-data-flask.service
```

停止自启动服务：

```bash
systemctl --user stop tick-data-flask.service
```

取消开机自启动：

```bash
cd /data1/code_git/tick_data_analysis
./install_flask_autostart.sh --remove
```

## 4. 日志

服务日志按天写入项目目录下的 `log/`：

```bash
log/YYYY-MM-DD.log
```

查看今日日志：

```bash
tail -f log/$(date +%F).log
```

## 5. 验证服务是否正常

检查端口是否监听：

```bash
ss -ltnp | grep 9999
```

检查首页是否可访问：

```bash
curl --noproxy '*' -I http://127.0.0.1:9999/
```

检查接口是否可访问：

```bash
curl --noproxy '*' http://127.0.0.1:9999/api/factors/list
```

说明：

- 访问 `/` 时返回 `302` 跳转到 `/dashboard` 属于正常行为。
- 如果本机配置了 HTTP 代理，访问 `127.0.0.1:9999` 时建议加 `--noproxy '*'`，避免被代理层误转发。
