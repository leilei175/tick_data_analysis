#!/usr/bin/env python3
"""
生成项目数据资产说明文档和元数据文件。

输出:
- doc/data_inventory.md
- doc/data_inventory.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
DOC_DIR = ROOT / "doc"
MD_OUT = DOC_DIR / "data_inventory.md"
JSON_OUT = DOC_DIR / "data_inventory.json"


@dataclass(frozen=True)
class ModuleSpec:
    path: str
    purpose: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    category: str
    path: str
    file_glob: str
    file_regex: Optional[str]
    frequency: str
    source: str
    update_method: str
    update_scripts: List[str]
    description: str
    notes: str = ""


MODULE_SPECS: List[ModuleSpec] = [
    ModuleSpec("mylib/analysis_engine.py", "封装因子分析主流程，连接读取、预处理、收益率与报告。"),
    ModuleSpec("mylib/constants.py", "集中管理数据目录、文件模式、字段名和项目常量。"),
    ModuleSpec("mylib/date_utils.py", "统一日期解析、格式转换与边界处理。"),
    ModuleSpec("mylib/factor_factory.py", "统一封装多类因子来源的访问入口。"),
    ModuleSpec("mylib/factor_preprocessor.py", "执行 winsorize、标准化等因子预处理。"),
    ModuleSpec("mylib/financial_column_mapper.py", "维护 Tushare 财务字段的中文映射。"),
    ModuleSpec("mylib/get_local_data.py", "读取本地日频、财务、衍生与中文财务 parquet 数据。"),
    ModuleSpec("mylib/get_remote_data.py", "通过 Flask parquet 接口远程读取本地数据。"),
    ModuleSpec("mylib/get_tick_data.py", "读取 tick_2026 目录中的逐股 tick parquet 数据。"),
    ModuleSpec("mylib/plotting_utils.py", "统一图表样式与可视化配置。"),
    ModuleSpec("mylib/returns_calculator.py", "计算远期收益率与相关收益指标。"),
    ModuleSpec("mylib/tushare_client.py", "集中初始化 Tushare 与交易日历访问。"),
]


DATASET_SPECS: List[DatasetSpec] = [
    DatasetSpec(
        name="tick_raw",
        category="tick",
        path="tick_2026",
        file_glob="*/*/*/*.parquet",
        file_regex=r"(?P<date>\d{4}/\d{2}/\d{2})/.+\.parquet$",
        frequency="tick",
        source="本地 tick 原始数据目录（当前为软链接到 /data1/quant-data/tick_2026）",
        update_method="外部落库；项目内脚本以读取为主，不负责原始 tick 下载。",
        update_scripts=["mylib/get_tick_data.py", "tick_reader.py", "high_frequency_factors.py"],
        description="按 年/月/日/股票代码 组织的原始逐笔数据，是所有高频因子的上游数据源。",
        notes="目录为软链接；本仓库主要消费该数据。",
    ),
    DatasetSpec(
        name="daily_market",
        category="daily_data",
        path="daily_data/daily",
        file_glob="**/daily_*.parquet",
        file_regex=r"daily_(?P<date>\d{8})\.parquet$",
        frequency="日频",
        source="Tushare 日线行情接口",
        update_method="通过 update_data.py 或 tushare_downloader.py 增量更新交易日文件。",
        update_scripts=["update_data.py", "tushare_downloader.py"],
        description="A 股日线行情，文件按交易日切分。",
    ),
    DatasetSpec(
        name="daily_basic",
        category="daily_data",
        path="daily_data/daily_basic",
        file_glob="**/daily_basic_*.parquet",
        file_regex=r"daily_basic_(?P<date>\d{8})\.parquet$",
        frequency="日频",
        source="Tushare daily_basic 接口",
        update_method="通过 update_data.py 或 tushare_downloader.py 增量更新。",
        update_scripts=["update_data.py", "tushare_downloader.py"],
        description="每日基本面、市值、换手率等横截面数据。",
    ),
    DatasetSpec(
        name="cashflow_quarter",
        category="financial_quarterly",
        path="daily_data/cashflow",
        file_glob="**/cashflow_*.parquet",
        file_regex=r"cashflow_(?P<date>\d{8})(?:_cn)?\.parquet$",
        frequency="季频/公告期",
        source="Tushare 现金流量表",
        update_method="通过 financial_downloader.py 或 update_data.py 下载季度文件，并合并 all 文件。",
        update_scripts=["financial_downloader.py", "update_data.py"],
        description="原始季度现金流数据，目录按 年/月 组织。",
    ),
    DatasetSpec(
        name="income_quarter",
        category="financial_quarterly",
        path="daily_data/income",
        file_glob="**/income_*.parquet",
        file_regex=r"income_(?P<date>\d{8})(?:_cn)?\.parquet$",
        frequency="季频/公告期",
        source="Tushare 利润表",
        update_method="通过 financial_downloader.py 或 update_data.py 下载和增量补齐。",
        update_scripts=["financial_downloader.py", "update_data.py"],
        description="原始季度利润表数据。",
    ),
    DatasetSpec(
        name="balance_quarter",
        category="financial_quarterly",
        path="daily_data/balance",
        file_glob="**/balance_*.parquet",
        file_regex=r"balance_(?P<date>\d{8})(?:_cn)?\.parquet$",
        frequency="季频/公告期",
        source="Tushare 资产负债表",
        update_method="通过 financial_downloader.py 或 update_data.py 下载和增量补齐。",
        update_scripts=["financial_downloader.py", "update_data.py"],
        description="原始季度资产负债表数据。",
    ),
    DatasetSpec(
        name="cashflow_daily",
        category="financial_daily",
        path="daily_data/cashflow_daily",
        file_glob="cashflow_daily_*.parquet",
        file_regex=r"cashflow_daily_(?P<date>\d{8})\.parquet$",
        frequency="日频",
        source="由 cashflow 季度财务数据按公告日展开得到",
        update_method="使用 financial_daily_converter.py 或 cashflow_daily_converter.py 从季度表生成；新财报后可增量更新。",
        update_scripts=["financial_daily_converter.py", "cashflow_daily_converter.py", "update_data.py"],
        description="英文列名版本的日频现金流数据。",
        notes="同目录还包含 yearly full 文件与 `_cn` 变体文件。",
    ),
    DatasetSpec(
        name="income_daily",
        category="financial_daily",
        path="daily_data/income_daily",
        file_glob="income_daily_*.parquet",
        file_regex=r"income_daily_(?P<date>\d{8})\.parquet$",
        frequency="日频",
        source="由 income 季度财务数据按公告日展开得到",
        update_method="使用 financial_daily_converter.py 从季度表生成；支持公告日后的增量更新。",
        update_scripts=["financial_daily_converter.py", "update_data.py"],
        description="英文列名版本的日频利润表数据。",
        notes="同目录还包含 yearly full 文件与 `_cn` 变体文件。",
    ),
    DatasetSpec(
        name="balance_daily",
        category="financial_daily",
        path="daily_data/balance_daily",
        file_glob="balance_daily_*.parquet",
        file_regex=r"balance_daily_(?P<date>\d{8})\.parquet$",
        frequency="日频",
        source="由 balance 季度财务数据按公告日展开得到",
        update_method="使用 financial_daily_converter.py 从季度表生成；支持公告日后的增量更新。",
        update_scripts=["financial_daily_converter.py", "update_data.py"],
        description="英文列名版本的日频资产负债表数据。",
        notes="同目录还包含 yearly full 文件与 `_cn` 变体文件。",
    ),
    DatasetSpec(
        name="cashflow_daily_cn",
        category="financial_daily_cn",
        path="daily_data/cashflow_daily_cn",
        file_glob="**/cashflow_daily_cn_*.parquet",
        file_regex=r"cashflow_daily_cn_(?P<date>\d{8})\.parquet$",
        frequency="日频",
        source="由英文版日频财务数据重组/转中文列名得到",
        update_method="使用 reorganize_financial_daily_cn.py 重新整理目录与年汇总文件。",
        update_scripts=["reorganize_financial_daily_cn.py"],
        description="中文字段版本的日频现金流数据，用于基本面因子计算。",
    ),
    DatasetSpec(
        name="income_daily_cn",
        category="financial_daily_cn",
        path="daily_data/income_daily_cn",
        file_glob="**/income_daily_cn_*.parquet",
        file_regex=r"income_daily_cn_(?P<date>\d{8})\.parquet$",
        frequency="日频",
        source="由英文版日频财务数据重组/转中文列名得到",
        update_method="使用 reorganize_financial_daily_cn.py 重新整理目录与年汇总文件。",
        update_scripts=["reorganize_financial_daily_cn.py"],
        description="中文字段版本的日频利润表数据。",
    ),
    DatasetSpec(
        name="balance_daily_cn",
        category="financial_daily_cn",
        path="daily_data/balance_daily_cn",
        file_glob="**/balance_daily_cn_*.parquet",
        file_regex=r"balance_daily_cn_(?P<date>\d{8})\.parquet$",
        frequency="日频",
        source="由英文版日频财务数据重组/转中文列名得到",
        update_method="使用 reorganize_financial_daily_cn.py 重新整理目录与年汇总文件。",
        update_scripts=["reorganize_financial_daily_cn.py"],
        description="中文字段版本的日频资产负债表数据。",
    ),
    DatasetSpec(
        name="derivative_financial_metrics",
        category="derived_daily",
        path="daily_data/derivative",
        file_glob="**/derivative_*.parquet",
        file_regex=r"derivative_(?P<date>\d{8})\.parquet$",
        frequency="日频",
        source="由 income_daily 和 balance_daily 派生计算",
        update_method="使用 build_derivative_financial_metrics.py 全量重建日文件和年度 full 文件。",
        update_scripts=["build_derivative_financial_metrics.py"],
        description="衍生财务指标，当前包含 roe、roa、gross_margin、roic。",
    ),
    DatasetSpec(
        name="wind_hub_imports",
        category="external_import",
        path="daily_data/wind_hub",
        file_glob="*.parquet",
        file_regex=None,
        frequency="季度/不规则",
        source="Wind Hub 外部导入数据",
        update_method="手工导入后，使用 convert_wind_hub.py 转为时间 x 股票宽表。",
        update_scripts=["convert_wind_hub.py"],
        description="当前包含 ROIC、销售毛利率、定期报告实际披露日期等外部数据。",
    ),
    DatasetSpec(
        name="high_frequency_daily_factors",
        category="factor_daily",
        path="factor/high_frequency",
        file_glob="*.parquet",
        file_regex=r"(?P<date>\d{4}_\d{2}_\d{2})\.parquet$",
        frequency="日频",
        source="由 tick_raw 计算得到",
        update_method="单日用 high_frequency_factors.py 计算，补齐缺口用 hf_factor_auto_update.py 自动更新。",
        update_scripts=["high_frequency_factors.py", "hf_factor_auto_update.py"],
        description="逐日保存的高频因子明细文件，每行通常对应单只股票当天的因子结果。",
    ),
    DatasetSpec(
        name="high_frequency_daily_panels",
        category="factor_daily",
        path="factor/daily",
        file_glob="*.parquet",
        file_regex=r"(?:zz1000_factors_|factors_)(?P<date>\d{8})\.parquet$",
        frequency="日频",
        source="由 tick_raw 计算得到的单日因子面板",
        update_method="主要由 compute_zz1000_factors.py 或 high_frequency_factors.py 生成。",
        update_scripts=["compute_zz1000_factors.py", "high_frequency_factors.py"],
        description="中证1000或全市场单日因子面板，以及汇总文件 zz1000_all_factors.parquet。",
    ),
    DatasetSpec(
        name="by_factor_wide_tables",
        category="factor_wide",
        path="factor/by_factor",
        file_glob="zz1000_*.parquet",
        file_regex=None,
        frequency="日频宽表",
        source="由 factor/daily 聚合得到",
        update_method="使用 batch_aggregate_factors.py 将日面板转为按因子宽表，含整段和按年文件。",
        update_scripts=["batch_aggregate_factors.py"],
        description="高频因子宽表，按因子拆文件，行是日期，列是股票代码。",
        notes="包含 bid_ask_spread、vwap_deviation、trade_imbalance 等 10 个高频因子。",
    ),
    DatasetSpec(
        name="forward_returns",
        category="factor_aux",
        path="factor/by_factor",
        file_glob="return_*d.parquet",
        file_regex=r"return_(?P<days>\d+)d\.parquet$",
        frequency="日频标签",
        source="由 factor/daily 中的 lastPrice 计算得到",
        update_method="使用 calculate_returns.py 批量重建 1/5/10 日远期收益率。",
        update_scripts=["calculate_returns.py"],
        description="因子分析标签文件，当前包括 1d、5d、10d 未来收益率。",
    ),
    DatasetSpec(
        name="kzz_call_auction_factor",
        category="factor_aux",
        path="factor/by_factor",
        file_glob="*call_auction*.parquet",
        file_regex=None,
        frequency="日频宽表",
        source="由 tick_raw 中可转债集合竞价成交额计算得到",
        update_method="使用 build_kzz_call_auction_factor.py 回填或 update_kzz_call_auction_factor.py 增量更新。",
        update_scripts=["build_kzz_call_auction_factor.py", "update_kzz_call_auction_factor.py", "hf_factor_auto_update.py"],
        description="可转债集合竞价成交额因子宽表及年度文件。",
    ),
    DatasetSpec(
        name="call_auction_snapshot_daily",
        category="factor_daily",
        path="factor/high_frequency/call_auction_snapshot",
        file_glob="call_auction_snapshot_*.parquet",
        file_regex=r"call_auction_snapshot_(?P<date>\d{4}_\d{2}_\d{2})\.parquet$",
        frequency="日频",
        source="由 tick_raw 中集合竞价阶段盘口快照计算得到",
        update_method="使用 build_call_auction_snapshot_factors.py 按日期范围回填或增量计算。",
        update_scripts=["build_call_auction_snapshot_factors.py"],
        description="集合竞价盘口快照日频明细文件，每行对应单只股票在 09:15:00 <= t < 09:25:00 窗口内的 4 个快照指标。",
        notes="字段包括 auction_last1_ask1_ret、auction_last2_ask1_ret、auction_last1_askVol1、auction_last2_askVol1。",
    ),
    DatasetSpec(
        name="call_auction_snapshot_wide",
        category="factor_wide",
        path="factor/by_factor",
        file_glob="auction_last*.parquet",
        file_regex=None,
        frequency="日频宽表",
        source="由 call_auction_snapshot 日频明细透视得到",
        update_method="使用 build_call_auction_snapshot_factors.py 在生成日频明细后同步更新年度宽表。",
        update_scripts=["build_call_auction_snapshot_factors.py"],
        description="集合竞价盘口快照因子年度宽表，行是日期，列是股票代码，每个指标单独保存一个 parquet 文件。",
        notes="当前包含 4 个宽表：auction_last1_ask1_ret、auction_last2_ask1_ret、auction_last1_askVol1、auction_last2_askVol1。",
    ),
    DatasetSpec(
        name="fundamental_factor_tables",
        category="factor_fundamental",
        path="factor/fundamental",
        file_glob="*.parquet",
        file_regex=None,
        frequency="日频宽表",
        source="由中文财务日频数据和 daily_basic 推导",
        update_method="使用 build_fundamental_factors.py 重建，支持全市场和 zz1000 两套输出。",
        update_scripts=["build_fundamental_factors.py"],
        description="基本面因子宽表，当前包含 ROE、ROA、Book-to-Market、FCF Yield 等。",
    ),
    DatasetSpec(
        name="preprocessed_factors",
        category="factor_preprocessed",
        path="factor/preprocessed",
        file_glob="*.parquet",
        file_regex=None,
        frequency="日频宽表",
        source="由原始因子宽表经过预处理得到",
        update_method="由因子预处理流程生成，当前仓库以结果文件为主。",
        update_scripts=["mylib/factor_preprocessor.py", "convert_factors.py"],
        description="标准化或裁剪后的因子文件，例如 zscore、winsorize 结果。",
    ),
    DatasetSpec(
        name="factor_analysis_outputs",
        category="analysis_output",
        path="factor/analysis",
        file_glob="**/*",
        file_regex=None,
        frequency="按分析任务生成",
        source="因子分析结果输出",
        update_method="由 factor_analysis.py、financial_factor_analysis.py 等分析脚本生成。",
        update_scripts=["factor_analysis.py", "financial_factor_analysis.py", "zz1000_factor_analysis.py"],
        description="分析报告、IC 统计、分层收益等结果目录。",
    ),
]


def _iter_matching_files(base: Path, file_glob: str, file_regex: Optional[str]) -> List[Path]:
    if not base.exists():
        return []

    files = [p for p in base.glob(file_glob) if p.is_file()]
    if file_regex is None:
        return sorted(files)

    regex = re.compile(file_regex)
    return sorted([p for p in files if regex.search(p.as_posix()) or regex.search(p.name)])


def _extract_dates(files: Iterable[Path], pattern: Optional[str]) -> List[str]:
    if pattern is None:
        return []

    regex = re.compile(pattern)
    dates: List[str] = []
    for fp in files:
        match = regex.search(fp.as_posix()) or regex.search(fp.name)
        if not match:
            continue
        value = match.groupdict().get("date")
        if value:
            dates.append(value.replace("/", "-"))
    return sorted(dates)


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _format_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num} B"


def _build_module_inventory() -> List[Dict]:
    modules: List[Dict] = []
    for spec in MODULE_SPECS:
        path = ROOT / spec.path
        modules.append(
            {
                "path": spec.path,
                "exists": path.exists(),
                "purpose": spec.purpose,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                if path.exists()
                else None,
            }
        )
    return modules


def _build_dataset_inventory() -> List[Dict]:
    datasets: List[Dict] = []
    for spec in DATASET_SPECS:
        base = ROOT / spec.path
        files = _iter_matching_files(base, spec.file_glob, spec.file_regex)
        latest = max(files, key=lambda p: p.stat().st_mtime) if files else None
        dates = _extract_dates(files, spec.file_regex)
        datasets.append(
            {
                "name": spec.name,
                "category": spec.category,
                "path": spec.path,
                "absolute_path": str(base.resolve()),
                "exists": base.exists(),
                "file_pattern": spec.file_glob,
                "file_name_example": files[0].name if files else None,
                "sample_file": str(files[0].relative_to(ROOT)) if files else None,
                "file_count": len(files),
                "date_min": dates[0] if dates else None,
                "date_max": dates[-1] if dates else None,
                "frequency": spec.frequency,
                "source": spec.source,
                "update_method": spec.update_method,
                "update_scripts": spec.update_scripts,
                "description": spec.description,
                "notes": spec.notes,
                "latest_file": str(latest.relative_to(ROOT)) if latest else None,
                "latest_updated_at": datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                if latest
                else None,
                "dir_size_bytes": _dir_size_bytes(base),
                "dir_size_human": _format_bytes(_dir_size_bytes(base)),
            }
        )
    return datasets


def _build_directory_summary(paths: List[str]) -> List[Dict]:
    summary = []
    for rel in paths:
        path = ROOT / rel
        parquet_count = sum(1 for _ in path.rglob("*.parquet")) if path.exists() else 0
        summary.append(
            {
                "path": rel,
                "absolute_path": str(path.resolve()),
                "exists": path.exists(),
                "parquet_count": parquet_count,
                "size_bytes": _dir_size_bytes(path),
                "size_human": _format_bytes(_dir_size_bytes(path)),
            }
        )
    return summary


def _render_markdown(metadata: Dict) -> str:
    lines: List[str] = []
    lines.append("# 项目数据资产说明")
    lines.append("")
    lines.append(f"- 生成时间: {metadata['generated_at']}")
    lines.append(f"- 项目根目录: `{metadata['project_root']}`")
    lines.append("")
    lines.append("## 1. 总览")
    lines.append("")
    lines.append("| 目录 | 是否存在 | Parquet文件数 | 目录体积 |")
    lines.append("|------|----------|--------------|----------|")
    for item in metadata["directory_summary"]:
        exists = "是" if item["exists"] else "否"
        lines.append(f"| `{item['path']}` | {exists} | {item['parquet_count']} | {item['size_human']} |")
    lines.append("")
    lines.append("## 2. mylib 库文件")
    lines.append("")
    lines.append("| 文件 | 作用 | 最近更新时间 |")
    lines.append("|------|------|--------------|")
    for mod in metadata["library_modules"]:
        lines.append(f"| `{mod['path']}` | {mod['purpose']} | {mod['updated_at'] or '-'} |")
    lines.append("")
    lines.append("## 3. 数据集清单")
    lines.append("")
    for ds in metadata["datasets"]:
        lines.append(f"### {ds['name']}")
        lines.append("")
        lines.append(f"- 数据分类: `{ds['category']}`")
        lines.append(f"- 数据保存地址: `{ds['path']}`")
        lines.append(f"- 绝对路径: `{ds['absolute_path']}`")
        lines.append(f"- 文件命名模式: `{ds['file_pattern']}`")
        lines.append(f"- 文件名示例: `{ds['file_name_example'] or '-'} `")
        lines.append(f"- 数据频度: {ds['frequency']}")
        lines.append(f"- 数据来源: {ds['source']}")
        lines.append(f"- 更新方式: {ds['update_method']}")
        lines.append(f"- 相关脚本: {', '.join(f'`{s}`' for s in ds['update_scripts']) if ds['update_scripts'] else '-'}")
        lines.append(f"- 文件数: {ds['file_count']}")
        lines.append(f"- 日期范围: {ds['date_min'] or '-'} -> {ds['date_max'] or '-'}")
        lines.append(f"- 最新文件: `{ds['latest_file'] or '-'} `")
        lines.append(f"- 最近更新时间: {ds['latest_updated_at'] or '-'}")
        lines.append(f"- 目录体积: {ds['dir_size_human']}")
        lines.append(f"- 说明: {ds['description']}")
        if ds["notes"]:
            lines.append(f"- 备注: {ds['notes']}")
        lines.append("")
    lines.append("## 4. 说明")
    lines.append("")
    lines.append("- `tick_2026` 当前是软链接，项目内高频因子与 tick 读取逻辑默认消费该目录。")
    lines.append("- `daily_data/*_daily` 目录中既有逐日文件，也混有 `YYYY_full.parquet`、`*_cn.parquet` 等汇总或变体文件；上面的条目已按主要命名模式拆分说明。")
    lines.append("- `factor/by_factor` 同时承载高频因子宽表、收益率标签和可转债集合竞价因子宽表。")
    lines.append("- 如果后续数据继续增长，可重复运行 `python generate_data_inventory.py` 刷新文档与 JSON 元数据。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    metadata = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(ROOT),
        "directory_summary": _build_directory_summary(["mylib", "tick_2026", "daily_data", "factor"]),
        "library_modules": _build_module_inventory(),
        "datasets": _build_dataset_inventory(),
    }

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    MD_OUT.write_text(_render_markdown(metadata), encoding="utf-8")
    print(f"wrote {MD_OUT}")
    print(f"wrote {JSON_OUT}")


if __name__ == "__main__":
    main()
