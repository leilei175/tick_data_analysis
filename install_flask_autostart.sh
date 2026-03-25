#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data1/code_git/tick_data_analysis"
SERVICE_NAME="tick-data-flask.service"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
SERVICE_FILE="${SYSTEMD_USER_DIR}/${SERVICE_NAME}"

show_usage() {
  echo "Usage: $0 [--install|--remove|--show|--status]"
}

ACTION="--install"
if [[ $# -gt 0 ]]; then
  ACTION="$1"
fi

mkdir -p "${SYSTEMD_USER_DIR}"

generate_service_file() {
  cat <<EOF
[Unit]
Description=Tick Data Analysis Flask Server
After=network.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
ExecStart=/bin/bash ${PROJECT_DIR}/flask_server.sh run
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
}

case "${ACTION}" in
  --install)
    generate_service_file > "${SERVICE_FILE}"
    systemctl --user daemon-reload
    systemctl --user enable "${SERVICE_NAME}"
    systemctl --user restart "${SERVICE_NAME}"
    echo "Installed and started ${SERVICE_NAME}"
    systemctl --user --no-pager --full status "${SERVICE_NAME}" | sed -n '1,12p'
    ;;
  --remove)
    systemctl --user disable --now "${SERVICE_NAME}" 2>/dev/null || true
    rm -f "${SERVICE_FILE}"
    systemctl --user daemon-reload
    echo "Removed ${SERVICE_NAME}"
    ;;
  --show)
    if [[ -f "${SERVICE_FILE}" ]]; then
      cat "${SERVICE_FILE}"
    else
      echo "Service file not found: ${SERVICE_FILE}"
    fi
    ;;
  --status)
    systemctl --user --no-pager --full status "${SERVICE_NAME}"
    ;;
  *)
    show_usage
    exit 1
    ;;
esac
