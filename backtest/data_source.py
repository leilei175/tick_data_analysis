from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _pick_yearly_files(data_dir: Path, start_year: int, end_year: int) -> List[Path]:
    files: List[Path] = []
    for year in range(start_year, end_year + 1):
        year_all = data_dir / f"{year}_all.parquet"
        year_full = data_dir / f"{year}_full.parquet"
        if year_all.exists():
            files.append(year_all)
        elif year_full.exists():
            files.append(year_full)
    return files


def load_daily_panel(
    daily_dir: str,
    start: str,
    end: str,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load local daily bar data from yearly parquet files.
    Expected columns: ts_code, trade_date, open, high, low, close, vol
    """
    start_year = int(start[:4])
    end_year = int(end[:4])
    data_dir = Path(daily_dir)
    yearly_files = _pick_yearly_files(data_dir, start_year, end_year)
    if not yearly_files:
        raise FileNotFoundError(f"No yearly parquet found under: {data_dir}")

    required_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    frames: List[pd.DataFrame] = []
    for file in yearly_files:
        df = pd.read_parquet(file, columns=required_cols)
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = pd.to_numeric(panel["trade_date"], errors="coerce").astype("Int64")
    panel = panel.dropna(subset=["trade_date"]).copy()
    panel["trade_date"] = panel["trade_date"].astype(int)

    start_int = int(start)
    end_int = int(end)
    panel = panel[(panel["trade_date"] >= start_int) & (panel["trade_date"] <= end_int)]

    if symbols:
        normalized = {s.strip().upper() for s in symbols if s and s.strip()}
        panel = panel[panel["ts_code"].str.upper().isin(normalized)]

    panel = panel.dropna(subset=["open", "high", "low", "close", "vol"])
    panel = panel.drop_duplicates(subset=["ts_code", "trade_date"], keep="first")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"].astype(str), format="%Y%m%d")
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return panel


def split_symbol_frames(panel: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Convert panel data to per-symbol OHLCV DataFrames indexed by datetime.
    """
    symbol_frames: Dict[str, pd.DataFrame] = {}
    for ts_code, df in panel.groupby("ts_code", sort=True):
        one = df[["trade_date", "open", "high", "low", "close", "vol"]].copy()
        one = one.rename(columns={"trade_date": "datetime", "vol": "volume"})
        one["openinterest"] = 0.0
        one = one.set_index("datetime").sort_index()
        symbol_frames[ts_code] = one
    return symbol_frames
