#!/usr/bin/env python3
"""可转债集合竞价成交额因子日更脚本。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from build_kzz_call_auction_factor import Config, run_update


def main() -> None:
    cfg = Config(
        tick_base=Path("/data1/quant-data/tick_2026"),
        daily_output_dir=Path("./factor/high_frequency/kzz_call_auction_amount"),
        wide_output_dir=Path("./factor/by_factor"),
    )
    # 默认更新到当天；若当日数据尚未落地，run_update 会自动因缺文件而跳过
    today = datetime.now().date()
    run_update(cfg=cfg, years=[2025, 2026], today_cutoff=today)


if __name__ == "__main__":
    main()
