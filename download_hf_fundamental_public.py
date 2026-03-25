from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import akshare as ak
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "daily_data" / "hf_raw"
STANDARD_ROOT = ROOT / "daily_data" / "hf_standard"
REPORT_ROOT = ROOT / "doc"


@dataclass
class DatasetResult:
    dataset_name: str
    status: str
    exact_match: bool
    rows: int = 0
    raw_path: Optional[str] = None
    standard_path: Optional[str] = None
    note: str = ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_frame(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def to_iso(ts: datetime) -> str:
    return ts.replace(microsecond=0).isoformat()


def fetch_repo_fixing_rates() -> Dict[str, pd.DataFrame]:
    return {
        "fr_fixing_rates": ak.repo_rate_query(symbol="回购定盘利率"),
        "fdr_fixing_rates": ak.repo_rate_query(symbol="银银间回购定盘利率"),
    }


def fetch_shibor() -> pd.DataFrame:
    return ak.macro_china_shibor_all()


def fetch_chinabond_yield() -> pd.DataFrame:
    start = date(2019, 1, 1)
    end = date.today()
    frames: List[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=360), end)
        df = ak.bond_china_yield(
            start_date=chunk_start.strftime("%Y%m%d"),
            end_date=chunk_end.strftime("%Y%m%d"),
        )
        frames.append(df)
        chunk_start = chunk_end + timedelta(days=1)
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    return out


def fetch_govt_futures_daily() -> Dict[str, pd.DataFrame]:
    return {
        "cffex_t0_daily": ak.futures_zh_daily_sina(symbol="T0"),
        "cffex_tf0_daily": ak.futures_zh_daily_sina(symbol="TF0"),
        "cffex_tl0_daily": ak.futures_zh_daily_sina(symbol="TL0"),
        "cffex_ts0_daily": ak.futures_zh_daily_sina(symbol="TS0"),
    }


def fetch_agri_product_index() -> pd.DataFrame:
    return ak.macro_china_agricultural_product()


def standardize_indicator_series(
    df: pd.DataFrame,
    dataset_name: str,
    time_col: str,
    value_cols: List[str],
    source: str,
    note: str = "",
) -> pd.DataFrame:
    sdf = df.copy()
    sdf[time_col] = pd.to_datetime(sdf[time_col])
    records: List[pd.DataFrame] = []
    for col in value_cols:
        part = sdf[[time_col, col]].copy()
        part = part.rename(columns={time_col: "observation_time", col: "value"})
        part["indicator_code"] = col
        part["dataset_name"] = dataset_name
        part["source"] = source
        part["freq"] = "day"
        part["ingest_time"] = to_iso(datetime.now())
        part["note"] = note
        records.append(part)
    out = pd.concat(records, ignore_index=True)
    out = out.sort_values(["indicator_code", "observation_time"]).reset_index(drop=True)
    return out


def standardize_curve(df: pd.DataFrame, dataset_name: str, source: str) -> pd.DataFrame:
    sdf = df.copy()
    sdf["日期"] = pd.to_datetime(sdf["日期"])
    tenor_cols = [c for c in sdf.columns if c not in ["曲线名称", "日期"]]
    records: List[pd.DataFrame] = []
    for tenor in tenor_cols:
        part = sdf[["曲线名称", "日期", tenor]].copy()
        part = part.rename(columns={"曲线名称": "curve_name", "日期": "observation_time", tenor: "value"})
        part["tenor"] = tenor
        part["dataset_name"] = dataset_name
        part["source"] = source
        part["freq"] = "day"
        part["ingest_time"] = to_iso(datetime.now())
        records.append(part)
    out = pd.concat(records, ignore_index=True)
    out = out.sort_values(["curve_name", "tenor", "observation_time"]).reset_index(drop=True)
    return out


def standardize_futures(df: pd.DataFrame, dataset_name: str, contract_code: str) -> pd.DataFrame:
    sdf = df.copy()
    sdf["date"] = pd.to_datetime(sdf["date"])
    out = sdf.rename(columns={"date": "observation_time"})
    out["dataset_name"] = dataset_name
    out["source"] = "akshare_sina"
    out["contract_code"] = contract_code
    out["freq"] = "day"
    out["ingest_time"] = to_iso(datetime.now())
    cols = [
        "dataset_name",
        "source",
        "contract_code",
        "observation_time",
        "freq",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "hold",
        "settle",
        "ingest_time",
    ]
    return out[cols].sort_values("observation_time").reset_index(drop=True)


def write_dataset(
    dataset_name: str,
    raw_df: pd.DataFrame,
    standard_df: pd.DataFrame,
    exact_match: bool,
    note: str,
) -> DatasetResult:
    raw_path = RAW_ROOT / dataset_name / f"{dataset_name}.parquet"
    standard_path = STANDARD_ROOT / f"dataset={dataset_name}" / "freq=day" / f"{dataset_name}.parquet"
    save_frame(raw_df, raw_path)
    save_frame(standard_df, standard_path)
    return DatasetResult(
        dataset_name=dataset_name,
        status="downloaded",
        exact_match=exact_match,
        rows=len(standard_df),
        raw_path=str(raw_path.relative_to(ROOT)),
        standard_path=str(standard_path.relative_to(ROOT)),
        note=note,
    )


def main() -> None:
    results: List[DatasetResult] = []

    now_note = f"downloaded_at={to_iso(datetime.now())}"

    repo_data = fetch_repo_fixing_rates()
    for dataset_name, raw_df in repo_data.items():
        standard_df = standardize_indicator_series(
            df=raw_df,
            dataset_name=dataset_name,
            time_col="date",
            value_cols=[c for c in raw_df.columns if c != "date"],
            source="akshare_chinamoney",
            note=f"{now_note}; related_repo_fixing_rate",
        )
        exact = False
        note = "公开可得的银行间回购定盘利率；不是需求文档中的 DR007 加权平均利率原始序列。"
        results.append(write_dataset(dataset_name, raw_df, standard_df, exact, note))

    shibor_raw = fetch_shibor()
    shibor_std = standardize_indicator_series(
        df=shibor_raw,
        dataset_name="shibor_all",
        time_col="日期",
        value_cols=[c for c in shibor_raw.columns if c != "日期"],
        source="akshare_jin10",
        note=f"{now_note}; exact_shibor",
    )
    results.append(
        write_dataset(
            "shibor_all",
            shibor_raw,
            shibor_std,
            exact_match=False,
            note="已下载 Shibor 全期限历史。可作为流动性补充，但不等于 MLF。",
        )
    )

    curve_raw = fetch_chinabond_yield()
    curve_std = standardize_curve(curve_raw, "chinabond_yield_curve", "akshare_chinabond")
    results.append(
        write_dataset(
            "chinabond_yield_curve",
            curve_raw,
            curve_std,
            exact_match=True,
            note="已下载中债收益率曲线历史数据，覆盖多个曲线名称和期限。",
        )
    )

    futures_data = fetch_govt_futures_daily()
    for dataset_name, raw_df in futures_data.items():
        contract_code = dataset_name.replace("cffex_", "").replace("_daily", "").upper()
        standard_df = standardize_futures(raw_df, dataset_name, contract_code)
        results.append(
            write_dataset(
                dataset_name,
                raw_df,
                standard_df,
                exact_match=True,
                note="已下载新浪连续合约日线。满足日频行情需求，不包含 Tick 数据。",
            )
        )

    agri_raw = fetch_agri_product_index()
    agri_std = standardize_indicator_series(
        df=agri_raw,
        dataset_name="agricultural_product_wholesale_index",
        time_col="日期",
        value_cols=["最新值", "涨跌幅", "近3月涨跌幅", "近6月涨跌幅", "近1年涨跌幅", "近2年涨跌幅", "近3年涨跌幅"],
        source="akshare_eastmoney",
        note=f"{now_note}; near_match_agri_wholesale_total_index",
    )
    results.append(
        write_dataset(
            "agricultural_product_wholesale_index",
            agri_raw,
            agri_std,
            exact_match=False,
            note="下载的是农产品批发价格总指数公开序列，不是需求文档中的 200 指数原始口径。",
        )
    )

    missing = [
        {
            "dataset_name": "dr007_weighted_avg",
            "status": "missing",
            "reason": "当前环境未找到稳定的免费公开接口；已下载 FR/FDR 定盘利率作为相关补充，不等于 DR007。",
        },
        {
            "dataset_name": "pboc_omo",
            "status": "missing",
            "reason": "需要对人民银行公开市场业务公告做专门抓取解析，当前未实现稳定列表抓取。",
        },
        {
            "dataset_name": "mlf_rate",
            "status": "missing",
            "reason": "需要抓取人民银行货币政策工具公告；Shibor 已下载但不等于 MLF。",
        },
        {
            "dataset_name": "aaa_1y_ncd_yield",
            "status": "missing",
            "reason": "公开免费接口不稳定，通常依赖中国货币网定制抓取或商业终端。",
        },
        {
            "dataset_name": "bill_rediscount_rate",
            "status": "missing",
            "reason": "通常需要上海票据交易所网页抓取或付费源。",
        },
        {
            "dataset_name": "steel_blast_furnace_rate",
            "status": "missing",
            "reason": "MySteel 产业数据通常需要付费接口。",
        },
        {
            "dataset_name": "asphalt_operating_rate",
            "status": "missing",
            "reason": "隆众/卓创数据通常需要付费接口。",
        },
        {
            "dataset_name": "coastal_coal_consumption",
            "status": "missing",
            "reason": "原始公开序列发布不稳定，目前未接替代口径。",
        },
        {
            "dataset_name": "housing_sales_30_cities",
            "status": "missing",
            "reason": "官方来源分散，通常需要 Wind/CREIS 或专门爬虫。",
        },
        {
            "dataset_name": "cement_shipments",
            "status": "missing",
            "reason": "百年建筑数据通常需要产业数据终端。",
        },
        {
            "dataset_name": "cpca_weekly_sales",
            "status": "missing",
            "reason": "乘联会公开网页接口当前不可稳定解析，需针对周报做专项抓取。",
        },
        {
            "dataset_name": "pig_grain_ratio",
            "status": "missing",
            "reason": "当前 AkShare 版本未提供稳定接口，需要另找公开源或自行抓取。",
        },
        {
            "dataset_name": "nh_industrial_index",
            "status": "missing",
            "reason": "南华期货公开 JSON 接口已失效，AkShare 当前版本对应接口返回解析错误。",
        },
    ]

    report = {
        "generated_at": to_iso(datetime.now()),
        "downloaded": [r.__dict__ for r in results],
        "missing": missing,
    }

    report_path = REPORT_ROOT / "hf_fundamental_download_status.json"
    ensure_dir(report_path.parent)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    downloaded = len(results)
    missing_count = len(missing)
    print(f"downloaded={downloaded} missing={missing_count} report={report_path.relative_to(ROOT)}")
    for item in results:
        print(f"{item.dataset_name}: rows={item.rows} exact={item.exact_match}")


if __name__ == "__main__":
    main()
