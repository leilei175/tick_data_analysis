#!/usr/bin/env python3
"""每日高频因子自动更新。

功能:
1. 补齐缺失的高频因子日文件: factor/high_frequency/YYYY_MM_DD.parquet
2. 补齐缺失的可转债集合竞价成交额因子日文件:
   factor/high_frequency/kzz_call_auction_amount/kzz_call_auction_amount_YYYY_MM_DD.parquet
3. 自动重建可转债因子年度宽表: factor/by_factor/kzz_call_auction_amount_YYYY.parquet
4. 输出结构化日志到 log/hf_factor_updates
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from high_frequency_factors import calc_high_frequency
from build_kzz_call_auction_factor import (
    FACTOR_NAME as KZZ_FACTOR_NAME,
    build_wide_for_year,
    compute_day_factor,
    output_file_for_date,
    save_day_factor,
)


@dataclass
class UpdateConfig:
    tick_base: Path = Path('/data1/quant-data/tick_2026')
    hf_output_dir: Path = Path('./factor/high_frequency')
    kzz_daily_output_dir: Path = Path('./factor/high_frequency/kzz_call_auction_amount')
    kzz_wide_output_dir: Path = Path('./factor/by_factor')
    log_dir: Path = Path('./log/hf_factor_updates')


def _list_tick_dates(tick_base: Path, years: Optional[Sequence[int]] = None) -> List[Tuple[date, Path]]:
    pairs: List[Tuple[date, Path]] = []

    if years:
        year_dirs = [tick_base / str(y) for y in years]
    else:
        year_dirs = sorted([p for p in tick_base.iterdir() if p.is_dir() and p.name.isdigit()]) if tick_base.exists() else []

    for year_dir in year_dirs:
        if not year_dir.exists() or not year_dir.name.isdigit():
            continue
        y = int(year_dir.name)
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir() or not month_dir.name.isdigit():
                continue
            m = int(month_dir.name)
            for day_dir in sorted(month_dir.iterdir()):
                if not day_dir.is_dir() or not day_dir.name.isdigit():
                    continue
                d = int(day_dir.name)
                try:
                    td = date(y, m, d)
                except ValueError:
                    continue
                if not any(day_dir.glob('*.parquet')):
                    continue
                pairs.append((td, day_dir))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _parse_existing_hf_dates(hf_output_dir: Path) -> Set[date]:
    out: Set[date] = set()
    if not hf_output_dir.exists():
        return out

    pat = re.compile(r'^(\d{4})_(\d{2})_(\d{2})\.parquet$')
    for fp in hf_output_dir.glob('*.parquet'):
        m = pat.match(fp.name)
        if not m:
            continue
        y, mo, d = map(int, m.groups())
        try:
            out.add(date(y, mo, d))
        except ValueError:
            continue
    return out


def _parse_existing_kzz_dates(kzz_output_dir: Path) -> Set[date]:
    out: Set[date] = set()
    if not kzz_output_dir.exists():
        return out

    pat = re.compile(rf'^{KZZ_FACTOR_NAME}_(\d{{4}})_(\d{{2}})_(\d{{2}})\.parquet$')
    for fp in kzz_output_dir.glob(f'{KZZ_FACTOR_NAME}_*.parquet'):
        m = pat.match(fp.name)
        if not m:
            continue
        y, mo, d = map(int, m.groups())
        try:
            out.add(date(y, mo, d))
        except ValueError:
            continue
    return out


def _date_to_iso(d: date) -> str:
    return d.strftime('%Y-%m-%d')


def _date_to_cli(d: date) -> str:
    return d.strftime('%Y-%m-%d')


def _safe_message(exc: Exception) -> str:
    return f'{type(exc).__name__}: {exc}'


def _determine_cutoff(include_today: bool, now: Optional[datetime] = None) -> date:
    now = now or datetime.now()
    if include_today:
        return now.date()
    return (now - timedelta(days=1)).date()


def run_update_job(
    cfg: Optional[UpdateConfig] = None,
    years: Optional[Sequence[int]] = None,
    include_today: Optional[bool] = None,
    cutoff_date: Optional[date] = None,
    verbose: bool = True,
) -> Dict:
    cfg = cfg or UpdateConfig()
    started_at = datetime.now()

    if include_today is None:
        now = datetime.now()
        include_today = (now.hour > 18) or (now.hour == 18 and now.minute >= 30)

    cutoff = cutoff_date or _determine_cutoff(include_today=include_today)

    tick_dates = _list_tick_dates(cfg.tick_base, years=years)
    tick_dates = [(d, p) for d, p in tick_dates if d <= cutoff]
    tick_date_set = {d for d, _ in tick_dates}

    hf_existing = _parse_existing_hf_dates(cfg.hf_output_dir)
    kzz_existing = _parse_existing_kzz_dates(cfg.kzz_daily_output_dir)

    hf_missing = sorted(tick_date_set - hf_existing)
    kzz_missing = sorted(tick_date_set - kzz_existing)

    tick_dir_map = {d: p for d, p in tick_dates}

    hf_success: List[str] = []
    hf_failed: List[Dict] = []
    kzz_success: List[str] = []
    kzz_failed: List[Dict] = []
    kzz_rows: Dict[str, int] = {}

    if verbose:
        print(f'tick交易日(<=cutoff): {len(tick_dates)}')
        print(f'hf缺失天数: {len(hf_missing)}')
        print(f'kzz缺失天数: {len(kzz_missing)}')

    # 1) 更新常规高频因子
    for idx, d in enumerate(hf_missing, 1):
        date_str = _date_to_cli(d)
        try:
            result_df = calc_high_frequency(
                date=date_str,
                stock_code='all',
                base_dir=str(cfg.tick_base),
                output_dir=str(cfg.hf_output_dir),
            )
            hf_success.append(date_str)
            if verbose and (idx % 5 == 0 or idx == len(hf_missing)):
                print(f'[HF {idx}/{len(hf_missing)}] {date_str} ok rows={len(result_df)}')
        except Exception as exc:
            hf_failed.append({
                'date': date_str,
                'error': _safe_message(exc),
            })
            if verbose:
                print(f'[HF {idx}/{len(hf_missing)}] {date_str} failed: {_safe_message(exc)}')

    # 2) 更新可转债集合竞价成交因子
    touched_years: Set[int] = set()
    for idx, d in enumerate(kzz_missing, 1):
        date_str = _date_to_cli(d)
        day_dir = tick_dir_map.get(d)
        if day_dir is None:
            kzz_failed.append({'date': date_str, 'error': 'tick目录不存在'})
            continue

        try:
            day_df = compute_day_factor(day_dir=day_dir, trade_date=d)
            out_file = output_file_for_date(d=d, daily_output_dir=cfg.kzz_daily_output_dir)
            save_day_factor(day_df, out_file)
            kzz_success.append(date_str)
            kzz_rows[date_str] = int(len(day_df))
            touched_years.add(d.year)
            if verbose and (idx % 5 == 0 or idx == len(kzz_missing)):
                print(f'[KZZ {idx}/{len(kzz_missing)}] {date_str} ok rows={len(day_df)}')
        except Exception as exc:
            kzz_failed.append({
                'date': date_str,
                'error': _safe_message(exc),
            })
            if verbose:
                print(f'[KZZ {idx}/{len(kzz_missing)}] {date_str} failed: {_safe_message(exc)}')

    # 3) 重建kzz年度宽表
    kzz_wide_outputs: List[str] = []
    kzz_wide_failed: List[Dict] = []
    if touched_years:
        kcfg = SimpleNamespace(
            daily_output_dir=cfg.kzz_daily_output_dir,
            wide_output_dir=cfg.kzz_wide_output_dir,
        )

        for y in sorted(touched_years):
            try:
                out = build_wide_for_year(y, kcfg)
                if out:
                    kzz_wide_outputs.append(str(out))
            except Exception as exc:
                kzz_wide_failed.append({'year': y, 'error': _safe_message(exc)})

    finished_at = datetime.now()
    elapsed_sec = round((finished_at - started_at).total_seconds(), 2)

    summary = {
        'status': 'success' if (not hf_failed and not kzz_failed and not kzz_wide_failed) else 'partial_success',
        'started_at': started_at.strftime('%Y-%m-%d %H:%M:%S'),
        'finished_at': finished_at.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_sec': elapsed_sec,
        'config': {
            'tick_base': str(cfg.tick_base),
            'hf_output_dir': str(cfg.hf_output_dir),
            'kzz_daily_output_dir': str(cfg.kzz_daily_output_dir),
            'kzz_wide_output_dir': str(cfg.kzz_wide_output_dir),
            'include_today': bool(include_today),
            'cutoff_date': _date_to_iso(cutoff),
            'years': list(years) if years else None,
        },
        'stats': {
            'tick_trade_days': len(tick_dates),
            'hf_missing_days': len(hf_missing),
            'hf_updated_days': len(hf_success),
            'hf_failed_days': len(hf_failed),
            'kzz_missing_days': len(kzz_missing),
            'kzz_updated_days': len(kzz_success),
            'kzz_failed_days': len(kzz_failed),
        },
        'details': {
            'hf_success_dates': hf_success,
            'hf_failed': hf_failed,
            'kzz_success_dates': kzz_success,
            'kzz_rows': kzz_rows,
            'kzz_failed': kzz_failed,
            'kzz_wide_outputs': kzz_wide_outputs,
            'kzz_wide_failed': kzz_wide_failed,
        },
    }

    _write_update_log(cfg.log_dir, summary)
    return summary


def _write_update_log(log_dir: Path, summary: Dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = log_dir / f'hf_update_{ts}.json'
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    jsonl_path = log_dir / 'hf_update_history.jsonl'
    with jsonl_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(summary, ensure_ascii=False) + '\n')

    text_path = log_dir / 'hf_update_latest.log'
    lines = [
        f"[{summary.get('finished_at')}] status={summary.get('status')} elapsed={summary.get('elapsed_sec')}s",
        f"HF: missing={summary['stats']['hf_missing_days']}, updated={summary['stats']['hf_updated_days']}, failed={summary['stats']['hf_failed_days']}",
        f"KZZ: missing={summary['stats']['kzz_missing_days']}, updated={summary['stats']['kzz_updated_days']}, failed={summary['stats']['kzz_failed_days']}",
    ]
    text_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='每日高频因子自动更新（含可转债集合竞价因子）')
    parser.add_argument('--years', default='', help='限制年份，逗号分隔，如 2025,2026；默认全量目录年份')
    parser.add_argument('--include-today', action='store_true', help='包含当天（默认18:30后自动包含）')
    parser.add_argument('--exclude-today', action='store_true', help='强制不包含当天，仅到昨日')
    parser.add_argument('--cutoff-date', default='', help='截止日期 YYYY-MM-DD（覆盖 include_today）')
    parser.add_argument('--quiet', action='store_true', help='减少stdout输出')

    args = parser.parse_args()

    years: Optional[List[int]] = None
    if args.years.strip():
        years = [int(x.strip()) for x in args.years.split(',') if x.strip()]

    include_today: Optional[bool] = None
    if args.include_today:
        include_today = True
    if args.exclude_today:
        include_today = False

    cutoff_date: Optional[date] = None
    if args.cutoff_date.strip():
        cutoff_date = datetime.strptime(args.cutoff_date.strip(), '%Y-%m-%d').date()

    try:
        summary = run_update_job(
            cfg=UpdateConfig(),
            years=years,
            include_today=include_today,
            cutoff_date=cutoff_date,
            verbose=not args.quiet,
        )
        print(json.dumps(summary, ensure_ascii=False))
    except Exception as exc:
        error = {
            'status': 'error',
            'message': _safe_message(exc),
            'traceback': traceback.format_exc(),
            'finished_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        UpdateConfig().log_dir.mkdir(parents=True, exist_ok=True)
        err_path = UpdateConfig().log_dir / f"hf_update_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        err_path.write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(error, ensure_ascii=False))
        raise


if __name__ == '__main__':
    main()
