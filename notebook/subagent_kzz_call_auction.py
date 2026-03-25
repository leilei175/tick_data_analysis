#!/usr/bin/env python3
"""可转债集合竞价分析子任务脚本。

功能:
1. 扫描 tick_2026 中可转债 tick 数据，计算集合竞价成交额与价格走势指标。
2. 统计集合竞价成交额分布。
3. 计算开盘后 5 分钟走势与集合竞价指标关系。
4. 产出 CSV/PNG 结果并自动生成 notebook。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KZZ_PREFIXES = ("110", "111", "113", "118", "123", "127", "128")
AUCTION_START = "09:15:00"
AUCTION_END = "09:26:00"  # < 09:26:00
OPEN5_START = "09:30:00"
OPEN5_END = "09:35:00"    # <= 09:35:00


@dataclass
class Config:
    year: int = 2026
    tick_base: Path = Path("/data1/quant-data/tick_2026")
    output_base: Path = Path("/data1/code_git/tick_data_analysis/notebook")

    @property
    def tick_root(self) -> Path:
        return self.tick_base / str(self.year)

    @property
    def output_dir(self) -> Path:
        return self.output_base / f"kzz_call_auction_outputs_{self.year}"

    @property
    def notebook_path(self) -> Path:
        return self.output_base / f"集合竞价kzz分析结果_{self.year}.ipynb"


def is_kzz_code(code: str) -> bool:
    return code.startswith(KZZ_PREFIXES) and code.endswith((".SH", ".SZ"))


def to_local_datetime(time_series: pd.Series) -> pd.Series:
    # 原始 time 为 epoch 毫秒，转为北京时间
    return pd.to_datetime(time_series, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)


def parse_level1(level) -> float:
    if isinstance(level, (list, tuple, np.ndarray)):
        return float(level[0]) if len(level) else np.nan
    if isinstance(level, str):
        s = level.strip()
        if not s or s == "[]":
            return np.nan
        s = s.strip("[]")
        if not s:
            return np.nan
        first = s.split(",")[0].strip()
        try:
            return float(first)
        except ValueError:
            return np.nan
    return np.nan


def compute_file_metrics(parquet_file: Path) -> tuple[Dict, List[Dict]] | tuple[None, None]:
    code = parquet_file.stem
    if not is_kzz_code(code):
        return None, None

    try:
        df = pd.read_parquet(parquet_file, columns=["time", "lastPrice", "amount", "bidPrice", "askPrice"])
    except Exception:
        return None, None

    if df.empty:
        return None, None

    df = df.sort_values("time").reset_index(drop=True)
    dt = to_local_datetime(df["time"])
    amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    delta_amount = amount.diff()
    if len(delta_amount) > 0:
        delta_amount.iloc[0] = amount.iloc[0]
    delta_amount = delta_amount.clip(lower=0.0)

    trade_price = pd.to_numeric(df["lastPrice"], errors="coerce").where(lambda s: s > 0)
    bid1 = df["bidPrice"].apply(parse_level1)
    ask1 = df["askPrice"].apply(parse_level1)
    mid_price = ((bid1 + ask1) / 2.0).where((bid1 > 0) & (ask1 > 0))
    # 集合竞价阶段常见无成交，使用“成交价优先 + 中间价回填”刻画价格走势
    auction_ref_price = trade_price.fillna(mid_price)

    day = pd.Timestamp(dt.iloc[0]).date()
    auction_start = pd.Timestamp(f"{day} {AUCTION_START}")
    auction_end = pd.Timestamp(f"{day} {AUCTION_END}")
    open5_start = pd.Timestamp(f"{day} {OPEN5_START}")
    open5_end = pd.Timestamp(f"{day} {OPEN5_END}")

    auction_mask = (dt >= auction_start) & (dt < auction_end)
    open5_mask = (dt >= open5_start) & (dt <= open5_end)

    auction_amount = float(delta_amount[auction_mask].sum())

    auction_price = auction_ref_price[auction_mask].dropna()
    open5_price = trade_price[open5_mask].dropna()

    first_auction_price = float(auction_price.iloc[0]) if not auction_price.empty else np.nan
    last_auction_price = float(auction_price.iloc[-1]) if not auction_price.empty else np.nan
    auction_return = (
        (last_auction_price / first_auction_price - 1.0)
        if (not np.isnan(first_auction_price) and first_auction_price > 0 and not np.isnan(last_auction_price))
        else np.nan
    )
    auction_amplitude = (
        (float(auction_price.max()) / float(auction_price.min()) - 1.0)
        if len(auction_price) >= 2 and float(auction_price.min()) > 0
        else np.nan
    )

    open5_return = (
        (float(open5_price.iloc[-1]) / float(open5_price.iloc[0]) - 1.0)
        if len(open5_price) >= 2 and float(open5_price.iloc[0]) > 0
        else np.nan
    )

    open5_amplitude = (
        (float(open5_price.max()) / float(open5_price.min()) - 1.0)
        if len(open5_price) >= 2 and float(open5_price.min()) > 0
        else np.nan
    )

    open5_volatility = (
        float(open5_price.pct_change().dropna().std())
        if len(open5_price) >= 3
        else np.nan
    )

    minute_rows: List[Dict] = []
    if len(auction_price) >= 2 and first_auction_price > 0:
        ap_df = pd.DataFrame({"dt": dt[auction_mask], "price": auction_price})
        ap_df["minute"] = ap_df["dt"].dt.strftime("%H:%M")
        minute_close = ap_df.groupby("minute", as_index=False)["price"].last()
        minute_close["norm_price"] = minute_close["price"] / first_auction_price
        minute_rows = [
            {
                "date": str(day),
                "code": code,
                "minute": row.minute,
                "norm_price": float(row.norm_price),
            }
            for row in minute_close.itertuples(index=False)
        ]

    metrics = {
        "date": str(day),
        "code": code,
        "auction_amount": auction_amount,
        "auction_ticks": int(auction_mask.sum()),
        "auction_first_price": first_auction_price,
        "auction_last_price": last_auction_price,
        "auction_return": auction_return,
        "auction_amplitude": auction_amplitude,
        "open5_return": open5_return,
        "open5_amplitude": open5_amplitude,
        "open5_volatility": open5_volatility,
    }
    return metrics, minute_rows


def analyze_all(config: Config) -> Dict[str, pd.DataFrame]:
    detail_rows: List[Dict] = []
    minute_rows: List[Dict] = []

    parquet_files = sorted(config.tick_root.rglob("*.parquet"))
    for idx, file in enumerate(parquet_files, start=1):
        metrics, minutes = compute_file_metrics(file)
        if metrics is None:
            continue
        detail_rows.append(metrics)
        minute_rows.extend(minutes)
        if idx % 30000 == 0:
            print(f"processed {idx}/{len(parquet_files)} files, collected={len(detail_rows)}")

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        raise RuntimeError("未找到可用的可转债 tick 数据，无法完成分析。")

    detail_df["date"] = pd.to_datetime(detail_df["date"])
    daily_df = detail_df.groupby("date", as_index=False)["auction_amount"].sum().rename(
        columns={"auction_amount": "daily_auction_amount"}
    )

    minute_df = pd.DataFrame(minute_rows)
    if not minute_df.empty:
        minute_summary = (
            minute_df.groupby("minute")["norm_price"]
            .agg(
                mean_norm_price="mean",
                median_norm_price="median",
                sample_count="count",
            )
            .reset_index()
        )
    else:
        minute_summary = pd.DataFrame(columns=["minute", "mean_norm_price", "median_norm_price", "sample_count"])

    corr_cols = ["open5_return", "auction_amount", "auction_return", "auction_amplitude"]
    corr_df = detail_df[corr_cols].corr().reset_index().rename(columns={"index": "factor"})

    relation_rows: List[Dict] = []
    valid = detail_df.dropna(subset=["open5_return", "auction_amount", "auction_return"])
    if len(valid) >= 20:
        valid = valid.copy()
        valid["amount_q"] = pd.qcut(valid["auction_amount"], q=5, labels=False, duplicates="drop")
        valid["auction_ret_q"] = pd.qcut(valid["auction_return"], q=5, labels=False, duplicates="drop")

        for key in ["amount_q", "auction_ret_q"]:
            grp = valid.groupby(key, as_index=False).agg(
                sample_count=("open5_return", "count"),
                mean_open5_return=("open5_return", "mean"),
                median_open5_return=("open5_return", "median"),
                mean_auction_amount=("auction_amount", "mean"),
                mean_auction_return=("auction_return", "mean"),
            )
            grp["group_type"] = key
            relation_rows.extend(grp.to_dict("records"))

    relation_df = pd.DataFrame(relation_rows)

    return {
        "detail": detail_df,
        "daily": daily_df,
        "minute_summary": minute_summary,
        "correlation": corr_df,
        "relation": relation_df,
    }


def make_plots(data: Dict[str, pd.DataFrame], outdir: Path) -> None:
    detail_df = data["detail"]
    daily_df = data["daily"]
    minute_summary = data["minute_summary"]

    plt.style.use("ggplot")

    # 1) 成交额分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    amount = detail_df["auction_amount"].replace([np.inf, -np.inf], np.nan).dropna()
    amount = amount[amount >= 0]
    axes[0].hist(amount, bins=60, color="#2f6db0", alpha=0.85)
    axes[0].set_title("Auction Turnover Distribution")
    axes[0].set_xlabel("auction_amount")
    axes[0].set_ylabel("count")

    positive = amount[amount > 0]
    if not positive.empty:
        axes[1].hist(np.log10(positive), bins=60, color="#e07a5f", alpha=0.85)
    axes[1].set_title("log10(Auction Turnover), amount>0")
    axes[1].set_xlabel("log10(auction_amount)")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(outdir / "auction_amount_distribution.png", dpi=150)
    plt.close(fig)

    # 2) 日度总成交额
    fig, ax = plt.subplots(figsize=(12, 5))
    if not daily_df.empty:
        ax.plot(daily_df["date"], daily_df["daily_auction_amount"], color="#1f77b4", linewidth=1.2)
    ax.set_title("Daily Total Auction Turnover (All KZZ)")
    ax.set_xlabel("date")
    ax.set_ylabel("daily_auction_amount")
    fig.tight_layout()
    fig.savefig(outdir / "daily_auction_amount.png", dpi=150)
    plt.close(fig)

    # 3) 集合竞价价格走势（分钟均值）
    fig, ax = plt.subplots(figsize=(10, 5))
    if not minute_summary.empty:
        minute_summary = minute_summary.sort_values("minute")
        ax.plot(minute_summary["minute"], minute_summary["mean_norm_price"], marker="o", color="#264653")
        ax.axhline(1.0, color="#999", linestyle="--", linewidth=1)
    ax.set_title("Average Normalized Price Path During Auction")
    ax.set_xlabel("minute")
    ax.set_ylabel("mean normalized price")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(outdir / "auction_price_path.png", dpi=150)
    plt.close(fig)

    # 4) 关系散点
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample = detail_df.dropna(subset=["open5_return", "auction_amount", "auction_return"]).copy()
    if not sample.empty:
        sample = sample[sample["auction_amount"] > 0]
        axes[0].scatter(np.log10(sample["auction_amount"]), sample["open5_return"], s=6, alpha=0.35, color="#3a86ff")
        axes[0].set_xlabel("log10(auction_amount)")
        axes[0].set_ylabel("open5_return")
        axes[0].set_title("Open5 Return vs Auction Turnover")

        axes[1].scatter(sample["auction_return"], sample["open5_return"], s=6, alpha=0.35, color="#ff006e")
        axes[1].set_xlabel("auction_return")
        axes[1].set_ylabel("open5_return")
        axes[1].set_title("Open5 Return vs Auction Return")

    fig.tight_layout()
    fig.savefig(outdir / "auction_open5_relation.png", dpi=150)
    plt.close(fig)


def build_notebook(config: Config, data: Dict[str, pd.DataFrame]) -> None:
    import nbformat as nbf

    detail_df = data["detail"]
    daily_df = data["daily"]
    corr_df = data["correlation"]
    relation_df = data["relation"]

    total_auction_amount = detail_df["auction_amount"].sum()
    sample_count = len(detail_df)
    date_count = detail_df["date"].nunique()

    amount_desc = detail_df["auction_amount"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame("value")
    corr_md = corr_df.round(4).to_markdown(index=False)
    amount_md = amount_desc.round(2).to_markdown()

    relation_preview = relation_df.copy()
    if not relation_preview.empty:
        relation_preview = relation_preview.round(6)
        relation_md = relation_preview.to_markdown(index=False)
    else:
        relation_md = "无足够样本形成分组统计。"

    daily_preview = daily_df.tail(20).copy()
    daily_preview["date"] = daily_preview["date"].dt.strftime("%Y-%m-%d")
    daily_md = daily_preview.to_markdown(index=False)

    outdir = config.output_dir

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# 可转债集合竞价分析\n"
            "\n"
            f"- 数据源: `{str(config.tick_root)}`\n"
            "- 代码筛选: 代码前缀 `110/111/113/118/123/127/128` 且后缀 `.SH/.SZ`\n"
            "- 集合竞价窗口: `09:15:00` 到 `09:25:59`（实现为 `< 09:26:00`）\n"
            "- 开盘5分钟窗口: `09:30:00` 到 `09:35:00`\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1) 集合竞价总成交额\n"
            f"- 样本数（债券-交易日）: **{sample_count}**\n"
            f"- 覆盖交易日: **{date_count}**\n"
            f"- 全样本集合竞价总成交额: **{total_auction_amount:,.2f}**\n"
            "\n"
            "最近20个交易日的全市场（转债样本）日度集合竞价成交额:\n\n"
            f"{daily_md}"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2) 集合竞价价格走势变化\n"
            "基于每个样本在集合竞价阶段首个有效成交价归一化到 1，按分钟聚合后计算均值轨迹。"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3) 集合竞价成交额分布\n"
            "成交额描述统计如下：\n\n"
            f"{amount_md}"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4) 开盘5分钟走势与集合竞价指标关系\n"
            "相关性矩阵（`open5_return` 与核心集合竞价指标）：\n\n"
            f"{corr_md}\n\n"
            "分组统计（五分位）:\n\n"
            f"{relation_md}"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "from IPython.display import Image, display\n"
            f"outdir = Path(r'{str(outdir)}')\n"
            "for fn in [\n"
            "    'daily_auction_amount.png',\n"
            "    'auction_price_path.png',\n"
            "    'auction_amount_distribution.png',\n"
            "    'auction_open5_relation.png',\n"
            "]:\n"
            "    p = outdir / fn\n"
            "    if p.exists():\n"
            "        print(fn)\n"
            "        display(Image(filename=str(p)))\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            f"outdir = Path(r'{str(outdir)}')\n"
            "detail = pd.read_csv(outdir / 'kzz_auction_detail.csv')\n"
            "detail.head()"
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }

    config.notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with config.notebook_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def save_outputs(config: Config, data: Dict[str, pd.DataFrame]) -> None:
    outdir = config.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    data["detail"].to_csv(outdir / "kzz_auction_detail.csv", index=False)
    data["daily"].to_csv(outdir / "kzz_auction_daily.csv", index=False)
    data["minute_summary"].to_csv(outdir / "kzz_auction_minute_summary.csv", index=False)
    data["correlation"].to_csv(outdir / "kzz_auction_correlation.csv", index=False)
    data["relation"].to_csv(outdir / "kzz_auction_relation_quantiles.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="可转债集合竞价分析")
    parser.add_argument("--year", type=int, default=2026, help="分析年份，例如 2025")
    args = parser.parse_args()

    config = Config(year=args.year)
    if not config.tick_root.exists():
        raise FileNotFoundError(f"tick目录不存在: {config.tick_root}")

    print("Start analyzing KZZ auction data...")
    data = analyze_all(config)

    save_outputs(config, data)
    make_plots(data, config.output_dir)
    build_notebook(config, data)

    print("Done.")
    print(f"Notebook: {config.notebook_path}")
    print(f"Outputs : {config.output_dir}")


if __name__ == "__main__":
    main()
