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

    required_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pre_close"]
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


def _pick_daily_basic_files(data_dir: Path, start_year: int, end_year: int) -> List[Path]:
    """Pick yearly daily_basic parquet files."""
    files: List[Path] = []
    for year in range(start_year, end_year + 1):
        year_full = data_dir / f"{year}_full.parquet"
        if year_full.exists():
            files.append(year_full)
    return files


def load_daily_basic_panel(
    daily_basic_dir: str,
    start: str,
    end: str,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load daily basic data (including total_mv, circ_mv).

    Expected columns: ts_code, trade_date, total_mv, circ_mv
    """
    start_year = int(start[:4])
    end_year = int(end[:4])
    data_dir = Path(daily_basic_dir)
    yearly_files = _pick_daily_basic_files(data_dir, start_year, end_year)
    if not yearly_files:
        raise FileNotFoundError(f"No daily_basic parquet found under: {data_dir}")

    required_cols = ["ts_code", "trade_date", "total_mv", "circ_mv"]
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

    panel = panel.dropna(subset=["total_mv"])
    panel = panel.drop_duplicates(subset=["ts_code", "trade_date"], keep="first")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"].astype(str), format="%Y%m%d")
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return panel


def load_combined_panel(
    daily_dir: str,
    daily_basic_dir: str,
    start: str,
    end: str,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load daily panel with amount and total_mv combined.

    Returns a panel with: ts_code, trade_date, open, high, low, close, vol, amount, total_mv, circ_mv
    """
    # Load daily panel (includes amount)
    daily_panel = load_daily_panel(daily_dir=daily_dir, start=start, end=end, symbols=symbols)

    # Load daily basic (includes total_mv, circ_mv)
    basic_panel = load_daily_basic_panel(
        daily_basic_dir=daily_basic_dir, start=start, end=end, symbols=symbols
    )

    # Select only needed columns from basic
    basic_subset = basic_panel[["ts_code", "trade_date", "total_mv", "circ_mv"]].copy()

    # Convert datetime to string for merge key (more efficient)
    daily_panel = daily_panel.copy()
    daily_panel["date_key"] = daily_panel["trade_date"].dt.strftime("%Y%m%d")
    basic_subset = basic_subset.copy()
    basic_subset["date_key"] = basic_subset["trade_date"].dt.strftime("%Y%m%d")

    # Drop original datetime columns before merge
    daily_panel = daily_panel.drop(columns=["trade_date"])
    basic_subset = basic_subset.drop(columns=["trade_date"])

    # Merge on ts_code and date_key
    combined = daily_panel.merge(
        basic_subset,
        left_on=["ts_code", "date_key"],
        right_on=["ts_code", "date_key"],
        how="left"
    )

    # Fill missing total_mv with forward fill then backward fill
    combined = combined.sort_values(["ts_code", "date_key"])

    combined["total_mv"] = combined.groupby("ts_code")["total_mv"].ffill()
    combined["total_mv"] = combined.groupby("ts_code")["total_mv"].bfill()
    combined["circ_mv"] = combined.groupby("ts_code")["circ_mv"].ffill()
    combined["circ_mv"] = combined.groupby("ts_code")["circ_mv"].bfill()

    # Convert date_key back to datetime
    combined["trade_date"] = pd.to_datetime(combined["date_key"], format="%Y%m%d")
    combined = combined.drop(columns=["date_key"])

    return combined
