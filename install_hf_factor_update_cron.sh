#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data1/code_git/tick_data_analysis"
PYTHON_BIN="/home/zxx/miniconda3/envs/quant/bin/python"
CRON_TAG="# HF_FACTOR_AUTO_UPDATE"
CRON_LINE="30 18 * * 1-5 cd ${PROJECT_DIR} && ${PYTHON_BIN} hf_factor_auto_update.py --include-today >> log/hf_factor_update_cron.log 2>&1 ${CRON_TAG}"

show_usage() {
  echo "Usage: $0 [--install|--remove|--show]"
}

ACTION="--show"
if [[ $# -gt 0 ]]; then
  ACTION="$1"
fi

current_cron="$(crontab -l 2>/dev/null || true)"

case "$ACTION" in
  --install)
    if echo "$current_cron" | grep -Fq "$CRON_TAG"; then
      echo "Cron job already exists:"
      echo "$current_cron" | grep -F "$CRON_TAG"
      exit 0
    fi
    {
      echo "$current_cron"
      echo "$CRON_LINE"
    } | awk 'NF' | crontab -
    echo "Installed cron job:"
    crontab -l | grep -F "$CRON_TAG"
    ;;
  --remove)
    if ! echo "$current_cron" | grep -Fq "$CRON_TAG"; then
      echo "No cron job found with tag: $CRON_TAG"
      exit 0
    fi
    echo "$current_cron" | grep -Fv "$CRON_TAG" | crontab -
    echo "Removed cron job with tag: $CRON_TAG"
    ;;
  --show)
    echo "Expected cron line:"
    echo "$CRON_LINE"
    echo
    echo "Current matched cron jobs:"
    echo "$current_cron" | grep -F "$CRON_TAG" || true
    ;;
  *)
    show_usage
    exit 1
    ;;
esac
