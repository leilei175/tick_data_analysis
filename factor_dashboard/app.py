"""
高频因子分析平台
专业金融风格Web仪表盘
支持按因子存储的宽格式数据
"""

import os
import sys
import html
import hashlib
import re
import json
import uuid
import time
import shlex
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

from flask import Flask, render_template, jsonify, request, redirect, url_for, session

# 添加父目录到路径（确保正确导入 update_data）
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from factor_analysis import FactorAnalysis

# 导入 update_data 模块
import update_data as _update_data_module
get_all_latest_dates = _update_data_module.get_all_latest_dates
init_tushare = _update_data_module.init_tushare
update_daily_data = _update_data_module.download_daily_data
update_financial_data = _update_data_module.update_financial_data
is_after_market_close = _update_data_module.is_after_market_close
get_today_str = _update_data_module.get_today_str
parse_date = _update_data_module.parse_date
date_to_str = _update_data_module.date_to_str

try:
    import markdown as _markdown_lib
except Exception:
    _markdown_lib = None

# ==================== 配置 ====================
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'hf-factor-secret-key-2026')
    PORT = 9999
    DEBUG = False

# ==================== Flask应用 ====================
app = Flask(__name__)
app.config.from_object(Config)

BASE_DIR = Path(__file__).resolve().parent.parent
DOC_DIR = BASE_DIR / "doc"
REPORT_DOC_DIRS = [
    BASE_DIR / "factor_analysis_results" / "reports",
    BASE_DIR / "factor_analysis_results" / "financial_reports",
]

factor_analyzer = None
analysis_engine = None

BACKTEST_DIR = BASE_DIR / "backtest"
BACKTEST_RUNS_DIR = BACKTEST_DIR / "web_runs"
BACKTEST_RUNS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_BACKTEST_DIR = BACKTEST_DIR / "custom_scripts"
CUSTOM_BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_COMMON_PARAMS = [
    {"name": "start", "type": "str", "default": "20220101", "required": True, "label": "开始日期(YYYYMMDD)"},
    {"name": "end", "type": "str", "default": "20221231", "required": True, "label": "结束日期(YYYYMMDD)"},
    {"name": "daily_dir", "type": "str", "default": "daily_data/daily", "required": True, "label": "行情目录"},
    {"name": "symbols", "type": "str", "default": "", "required": False, "label": "股票列表(逗号分隔)"},
    {"name": "symbol_limit", "type": "int", "default": 50, "required": True, "label": "自动选股数量"},
    {"name": "cash", "type": "float", "default": 1000000.0, "required": True, "label": "初始资金"},
    {"name": "commission", "type": "float", "default": 0.001, "required": True, "label": "手续费率"},
    {"name": "extra_args", "type": "str", "default": "", "required": False, "label": "额外命令行参数(可选)"},
]

BACKTEST_SCRIPT_CONFIG = {
    "run_backtest.py": {
        "label": "均线策略回测（EqualWeightSmaCross）",
        "description": "多标的均线筛选，等权持仓。",
        "script_path": BACKTEST_DIR / "run_backtest.py",
        "default_output_subdir": "output",
        "params": [
            {"name": "start", "type": "str", "default": "20220101", "required": True, "label": "开始日期(YYYYMMDD)"},
            {"name": "end", "type": "str", "default": "20221231", "required": True, "label": "结束日期(YYYYMMDD)"},
            {"name": "daily_dir", "type": "str", "default": "daily_data/daily", "required": True, "label": "行情目录"},
            {"name": "symbols", "type": "str", "default": "", "required": False, "label": "股票列表(逗号分隔)"},
            {"name": "symbol_limit", "type": "int", "default": 20, "required": True, "label": "自动选股数量"},
            {"name": "cash", "type": "float", "default": 1000000.0, "required": True, "label": "初始资金"},
            {"name": "commission", "type": "float", "default": 0.001, "required": True, "label": "手续费率"},
            {"name": "short_window", "type": "int", "default": 10, "required": True, "label": "短均线窗口"},
            {"name": "long_window", "type": "int", "default": 30, "required": True, "label": "长均线窗口"},
            {"name": "rebalance_days", "type": "int", "default": 5, "required": True, "label": "调仓频率(天)"},
        ],
    },
    "run_factor_topn_demo.py": {
        "label": "因子打分 TopN 调仓 Demo",
        "description": "按过去N日收益率打分，截面选TopN等权调仓。",
        "script_path": BACKTEST_DIR / "run_factor_topn_demo.py",
        "default_output_subdir": "output_factor_topn",
        "params": [
            {"name": "start", "type": "str", "default": "20220101", "required": True, "label": "开始日期(YYYYMMDD)"},
            {"name": "end", "type": "str", "default": "20221231", "required": True, "label": "结束日期(YYYYMMDD)"},
            {"name": "daily_dir", "type": "str", "default": "daily_data/daily", "required": True, "label": "行情目录"},
            {"name": "symbols", "type": "str", "default": "", "required": False, "label": "股票列表(逗号分隔)"},
            {"name": "symbol_limit", "type": "int", "default": 50, "required": True, "label": "自动选股数量"},
            {"name": "cash", "type": "float", "default": 1000000.0, "required": True, "label": "初始资金"},
            {"name": "commission", "type": "float", "default": 0.001, "required": True, "label": "手续费率"},
            {"name": "lookback", "type": "int", "default": 20, "required": True, "label": "因子回看窗口"},
            {"name": "topn", "type": "int", "default": 10, "required": True, "label": "TopN持仓数"},
            {"name": "rebalance_days", "type": "int", "default": 5, "required": True, "label": "调仓频率(天)"},
        ],
    },
}

_backtest_tasks: Dict[str, Dict] = {}
_backtest_task_lock = threading.Lock()

# ==================== 登录装饰器 ====================
def login_required(f):
    """验证用户是否登录"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# 缓存因子数据
_factor_data_cache = {}

def get_by_factor_dir() -> Path:
    """获取按因子存储的目录"""
    return Path(__file__).resolve().parent.parent / "factor" / "by_factor"

def load_factor_data(factor_name: str, year: int = None) -> Optional[pd.DataFrame]:
    """懒加载因子数据"""
    key = f"{factor_name}_{year}" if year else factor_name

    if key not in _factor_data_cache:
        # 优先加载不带年份的完整文件
        if year is None:
            filepath = get_by_factor_dir() / f"zz1000_{factor_name}.parquet"
            if not filepath.exists():
                # 回退到年份文件
                year_files = sorted(get_by_factor_dir().glob(f"zz1000_{factor_name}_*.parquet"))
                if year_files:
                    dfs = [pd.read_parquet(f) for f in year_files]
                    _factor_data_cache[key] = pd.concat(dfs).sort_index()
                    return _factor_data_cache[key]
                return None
        else:
            filepath = get_by_factor_dir() / f"zz1000_{factor_name}_{year}.parquet"

        if filepath.exists():
            _factor_data_cache[key] = pd.read_parquet(filepath)
        else:
            return None

    return _factor_data_cache[key]

def list_available_factors() -> pd.DataFrame:
    """列出所有可用的因子文件"""
    data_path = get_by_factor_dir()
    if not data_path.exists():
        return pd.DataFrame()

    files = list(data_path.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    info = []
    for f in files:
        # 只处理zz1000_开头的因子文件
        if not f.stem.startswith('zz1000_'):
            continue
        # 解析文件名: zz1000_factor_name_year.parquet
        parts = f.stem.split('_')
        if len(parts) < 4:
            continue
        prefix = parts[0]
        factor = '_'.join(parts[1:-1])  # 处理因子名中可能的下划线
        year_str = parts[-1]
        # 跳过非年份文件（如 return_1d）
        if not year_str.isdigit():
            continue
        info.append({
            'filename': f.name,
            'prefix': prefix,
            'factor': factor,
            'year': int(year_str),
            'file_size_mb': float(round(f.stat().st_size / (1024 * 1024), 2)),
            'file_path': str(f)
        })

    return pd.DataFrame(info) if info else pd.DataFrame()


def list_all_factor_names() -> list:
    """列出所有可用的因子名称（包括不带年份的合并文件）"""
    data_path = get_by_factor_dir()
    if not data_path.exists():
        return []

    files = list(data_path.glob("zz1000_*.parquet"))
    factors = set()
    for f in files:
        stem = f.stem
        if stem.startswith('zz1000_'):
            # 去掉zz1000_前缀
            rest = stem[7:]  # len('zz1000_') == 7
            # 如果剩余部分以数字年份结尾，去掉它
            parts = rest.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                factor = parts[0]
            else:
                factor = rest
            factors.add(factor)

    return sorted(list(factors))

# 缓存收益率数据
_returns_cache = {}

def load_returns(period: int = 1) -> Optional[pd.DataFrame]:
    """加载收益率数据"""
    if period not in _returns_cache:
        filepath = get_by_factor_dir() / f"return_{period}d.parquet"
        if filepath.exists():
            _returns_cache[period] = pd.read_parquet(filepath)
        else:
            return None
    return _returns_cache[period]


def _safe_doc_title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ")


def _infer_doc_tags(path: Path, title: str, content: str) -> List[str]:
    tags = set()
    rel = path.relative_to(BASE_DIR).as_posix()
    lower_title = title.lower()
    lower_content = content.lower()

    if rel.startswith("doc/"):
        tags.add("指南")
    if rel.startswith("factor_analysis_results/reports"):
        tags.add("因子报告")
    if rel.startswith("factor_analysis_results/financial_reports"):
        tags.add("财务报告")
    if "update" in lower_title or "update" in rel:
        tags.add("数据更新")
    if "download" in lower_title or "downloader" in rel:
        tags.add("数据下载")
    if "factor" in lower_title or "因子" in title:
        tags.add("因子")
    if "converter" in rel or "转换" in title:
        tags.add("数据转换")
    if "analysis" in lower_title or "分析" in title:
        tags.add("分析")
    if "tushare" in lower_content:
        tags.add("Tushare")

    if not tags:
        tags.add("其他")
    return sorted(tags)


def _highlight_text(text: str, query: str) -> str:
    escaped = html.escape(text)
    if not query:
        return escaped
    pattern = re.compile(re.escape(html.escape(query)), flags=re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)


def _collect_doc_files() -> List[Path]:
    doc_files: List[Path] = []
    if DOC_DIR.exists():
        doc_files.extend(sorted(DOC_DIR.glob("*.md")))
    for report_dir in REPORT_DOC_DIRS:
        if report_dir.exists():
            doc_files.extend(sorted(report_dir.glob("*.md")))
    return doc_files


def _build_doc_index() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for path in _collect_doc_files():
        rel_path = path.relative_to(BASE_DIR).as_posix()
        doc_id = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:16]
        stat = path.stat()
        created_ts = float(getattr(stat, "st_ctime", stat.st_mtime))
        updated_ts = float(stat.st_mtime)
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raw = ""
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        title = _safe_doc_title(path)
        for line in lines[:8]:
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                break
        snippet = " ".join(lines[:6])[:220]
        docs.append({
            "id": doc_id,
            "title": title,
            "path": rel_path,
            "snippet": snippet,
            "content": raw,
            "tags": _infer_doc_tags(path, title, raw),
            "created_at_ts": created_ts,
            "updated_at_ts": updated_ts,
            "created_at": datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.fromtimestamp(updated_ts).strftime("%Y-%m-%d %H:%M:%S"),
        })
    docs.sort(key=lambda x: x["path"])
    return docs


def _render_markdown(raw_text: str) -> str:
    if _markdown_lib is not None:
        return _markdown_lib.markdown(
            raw_text,
            extensions=["fenced_code", "tables", "toc", "nl2br"],
        )
    escaped = html.escape(raw_text)
    return f"<pre>{escaped}</pre>"


def _load_factor_returns_merged(factor: str, period: int, year: int = None) -> tuple:
    """加载并合并因子与收益率数据，返回 (merged_df, error_message)。"""
    factor_df = load_factor_data(factor, year)
    if factor_df is None or factor_df.empty:
        return None, f'因子文件不存在或为空: {factor}'

    returns_df = load_returns(period)
    if returns_df is None or returns_df.empty:
        return None, '收益率数据为空'

    factor_long = factor_df.stack().reset_index()
    factor_long.columns = ['date', 'stock_code', 'factor_value']
    factor_long['date'] = pd.to_datetime(factor_long['date']).dt.strftime('%Y-%m-%d')

    returns_df = returns_df.copy()
    returns_df['date'] = pd.to_datetime(returns_df['date']).dt.strftime('%Y-%m-%d')
    returns_col = f'return_{period}d'
    if 'return_1d' in returns_df.columns:
        returns_df = returns_df.rename(columns={'return_1d': returns_col})
    if returns_col not in returns_df.columns:
        return None, f'收益率字段缺失: {returns_col}'

    returns_long = returns_df[['date', 'stock_code', returns_col]].copy()
    merged = factor_long.merge(returns_long, on=['date', 'stock_code']).dropna()
    if merged.empty:
        return None, '没有可用的重叠数据'
    return merged, None


def _calc_quantile_result(merged: pd.DataFrame, period: int, quantiles: int) -> list:
    returns_col = f'return_{period}d'
    merged = merged.copy()
    merged['quantile'] = merged.groupby('date')['factor_value'].transform(
        lambda x: pd.qcut(x, quantiles, labels=False, duplicates='drop') + 1
    )
    result = []
    for q in range(1, quantiles + 1):
        q_data = merged[merged['quantile'] == q]
        result.append({
            'quantile': q,
            'factor_mean': float(q_data['factor_value'].mean()),
            'return_mean': float(q_data[returns_col].mean()),
            'return_std': float(q_data[returns_col].std()),
            'count': int(len(q_data)),
            'ic': float(q_data['factor_value'].corr(q_data[returns_col])),
        })
    return result


def _cast_param_value(raw_val, param_type: str):
    if raw_val is None:
        return None
    if param_type == "int":
        return int(raw_val)
    if param_type == "float":
        return float(raw_val)
    return str(raw_val)


def _task_meta_path(output_dir: Path) -> Path:
    return output_dir / "task_meta.json"


def _run_log_path(output_dir: Path) -> Path:
    return output_dir / "run.log"


def _save_task_meta(task: Dict):
    try:
        output_dir = Path(task["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "task_id": task.get("task_id"),
            "script": task.get("script"),
            "status": task.get("status"),
            "message": task.get("message"),
            "progress": task.get("progress"),
            "params": task.get("params", {}),
            "output_dir": task.get("output_dir"),
            "created_at": task.get("created_at"),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
        }
        _task_meta_path(output_dir).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_task_meta(task_id: str) -> Optional[Dict]:
    meta_path = _task_meta_path(BACKTEST_RUNS_DIR / task_id)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail_lines(file_path: Path, max_lines: int = 100) -> List[str]:
    if not file_path.exists():
        return []
    try:
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-max_lines:]
    except Exception:
        return []


def _list_custom_backtest_scripts() -> List[str]:
    files = []
    for p in sorted(CUSTOM_BACKTEST_DIR.glob("*.py")):
        if p.name.startswith("_"):
            continue
        files.append(p.name)
    return files


def _resolve_custom_script_path(script_name: str) -> Optional[Path]:
    name = str(script_name).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.py", name):
        return None
    path = (CUSTOM_BACKTEST_DIR / name).resolve()
    if path.parent != CUSTOM_BACKTEST_DIR.resolve():
        return None
    return path


def _build_backtest_script_list() -> List[Dict]:
    scripts: List[Dict] = []
    for key, cfg in BACKTEST_SCRIPT_CONFIG.items():
        scripts.append({
            "script": key,
            "label": cfg["label"],
            "description": cfg["description"],
            "params": cfg["params"],
            "is_custom": False,
        })
    for name in _list_custom_backtest_scripts():
        scripts.append({
            "script": f"custom:{name}",
            "label": f"自定义脚本: {name}",
            "description": "用户自定义回测脚本（需支持通用命令行参数）",
            "params": BACKTEST_COMMON_PARAMS,
            "is_custom": True,
        })
    return scripts


def _load_history_items(limit: int = 200) -> List[Dict]:
    items = []
    for run_dir in BACKTEST_RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        task_id = run_dir.name
        meta = _load_task_meta(task_id) or {}
        metrics = {}
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                metrics = {}
        created_at = meta.get("created_at")
        if not created_at:
            created_at = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        items.append({
            "task_id": task_id,
            "script": meta.get("script", "unknown"),
            "status": meta.get("status", "unknown"),
            "created_at": created_at,
            "finished_at": meta.get("finished_at"),
            "final_value": metrics.get("final_value"),
            "total_return": metrics.get("total_return"),
            "result_url": f"/backtest/result/{task_id}",
        })
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return items[:limit]


def _append_backtest_log(task_id: str, line: str, max_lines: int = 500):
    log_file = None
    with _backtest_task_lock:
        task = _backtest_tasks.get(task_id)
        if not task:
            return
        log_file = task.get("log_file")
        task["logs"].append(line.rstrip("\n"))
        if len(task["logs"]) > max_lines:
            task["logs"] = task["logs"][-max_lines:]
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line if line.endswith("\n") else line + "\n")
        except Exception:
            pass


def _estimate_running_progress(start_ts: float) -> int:
    elapsed = max(0.0, time.time() - start_ts)
    # 以120秒为“典型耗时”估算进度，运行中封顶95%
    p = int(min(95, 5 + elapsed / 120.0 * 90))
    return max(5, p)


def _load_backtest_result(output_dir: Path) -> Dict:
    result = {
        "metrics": {},
        "equity_curve": [],
        "rebalance_log": [],
    }
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        try:
            import json as _json
            result["metrics"] = _json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            result["metrics"] = {}

    equity_file = output_dir / "equity_curve.csv"
    if equity_file.exists():
        try:
            eq_df = pd.read_csv(equity_file)
            if "datetime" in eq_df.columns:
                eq_df = eq_df.rename(columns={"datetime": "date"})
            if "date" not in eq_df.columns and len(eq_df.columns) >= 1:
                eq_df = eq_df.rename(columns={eq_df.columns[0]: "date"})
            if "nav" not in eq_df.columns and len(eq_df.columns) >= 2:
                eq_df = eq_df.rename(columns={eq_df.columns[1]: "nav"})
            result["equity_curve"] = eq_df[["date", "nav"]].to_dict(orient="records")
        except Exception:
            result["equity_curve"] = []

    rebalance_file = output_dir / "rebalance_log.csv"
    if rebalance_file.exists():
        try:
            rebalance_df = pd.read_csv(rebalance_file).head(200)
            result["rebalance_log"] = rebalance_df.to_dict(orient="records")
        except Exception:
            result["rebalance_log"] = []
    return result


def _run_backtest_task(task_id: str):
    with _backtest_task_lock:
        task = _backtest_tasks.get(task_id)
        if not task:
            return
        task["status"] = "running"
        task["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task["start_ts"] = time.time()
        task["progress"] = 5
        _save_task_meta(task)

    try:
        proc = subprocess.Popen(
            task["cmd"],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with _backtest_task_lock:
            task = _backtest_tasks.get(task_id)
            if task is not None:
                task["pid"] = proc.pid

        if proc.stdout is not None:
            for line in proc.stdout:
                _append_backtest_log(task_id, line)
                with _backtest_task_lock:
                    task = _backtest_tasks.get(task_id)
                    if task is None:
                        continue
                    task["progress"] = _estimate_running_progress(task.get("start_ts", time.time()))

        return_code = proc.wait()
        with _backtest_task_lock:
            task = _backtest_tasks.get(task_id)
            if task is None:
                return
            if return_code == 0:
                task["status"] = "success"
                task["progress"] = 100
                task["message"] = "回测完成"
                task["result"] = _load_backtest_result(Path(task["output_dir"]))
            else:
                task["status"] = "error"
                task["progress"] = 100
                task["message"] = f"回测失败，退出码: {return_code}"
            task["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _save_task_meta(task)
    except Exception as e:
        with _backtest_task_lock:
            task = _backtest_tasks.get(task_id)
            if task is None:
                return
            task["status"] = "error"
            task["progress"] = 100
            task["message"] = f"回测异常: {e}"
            task["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _save_task_meta(task)

# ==================== 路由 ====================

@app.route('/')
def index():
    """主页"""
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    """仪表盘页面"""
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/correlation')
def correlation():
    """相关性分析页面"""
    return render_template('correlation.html', active_page='correlation')

@app.route('/ic-analysis')
def ic_analysis():
    """IC分析页面"""
    return render_template('ic_analysis.html', active_page='ic')

@app.route('/quantile')
def quantile():
    """分层分析页面"""
    return render_template('quantile.html', active_page='quantile')

@app.route('/long-short')
def long_short():
    """多空组合页面"""
    return render_template('long_short.html', active_page='longshort')

@app.route('/data-manager')
def data_manager():
    """数据管理页面"""
    return render_template('data_manager.html', active_page='data-manager')


@app.route('/docs')
def docs_center():
    """文档中心页面"""
    return render_template('docs_center.html', active_page='docs')


@app.route('/backtest')
def backtest_page():
    """回测页面"""
    return render_template('backtest.html', active_page='backtest')


@app.route('/backtest/result/<task_id>')
def backtest_result_page(task_id: str):
    """回测结果页面"""
    return render_template('backtest_result.html', active_page='backtest', task_id=task_id)


@app.route('/api/backtest/scripts')
def api_backtest_scripts():
    """API: 获取可用回测脚本与参数定义"""
    try:
        scripts = _build_backtest_script_list()
        return jsonify({"status": "success", "data": {"scripts": scripts}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/backtest/run', methods=['POST'])
def api_backtest_run():
    """API: 启动回测任务"""
    try:
        payload = request.get_json() or {}
        script_name = str(payload.get("script", "")).strip()
        params = payload.get("params", {}) or {}

        is_custom = script_name.startswith("custom:")
        if is_custom:
            custom_name = script_name.split("custom:", 1)[1].strip()
            script_path = _resolve_custom_script_path(custom_name)
            if script_path is None or not script_path.exists():
                return jsonify({"status": "error", "message": f"自定义脚本不存在: {custom_name}"})
            cfg = {
                "label": f"自定义脚本: {custom_name}",
                "params": BACKTEST_COMMON_PARAMS,
                "allow_extra_args": True,
            }
        else:
            if script_name not in BACKTEST_SCRIPT_CONFIG:
                return jsonify({"status": "error", "message": f"不支持的脚本: {script_name}"})
            cfg = BACKTEST_SCRIPT_CONFIG[script_name]
            script_path = cfg["script_path"]
            if not script_path.exists():
                return jsonify({"status": "error", "message": f"脚本不存在: {script_path}"})

        casted = {}
        for p in cfg["params"]:
            name = p["name"]
            ptype = p["type"]
            required = p.get("required", False)
            default = p.get("default")
            raw_val = params.get(name, default)
            if required and (raw_val is None or str(raw_val).strip() == ""):
                return jsonify({"status": "error", "message": f"缺少参数: {name}"})
            if raw_val is None or (ptype == "str" and str(raw_val).strip() == ""):
                casted[name] = "" if ptype == "str" else default
                continue
            try:
                casted[name] = _cast_param_value(raw_val, ptype)
            except Exception:
                return jsonify({"status": "error", "message": f"参数类型错误: {name}"})

        task_id = uuid.uuid4().hex[:12]
        output_dir = BACKTEST_RUNS_DIR / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, str(script_path)]
        for p in cfg["params"]:
            name = p["name"]
            val = casted.get(name)
            if name == "extra_args":
                continue
            if p["type"] == "str" and str(val).strip() == "":
                continue
            cmd.extend([f"--{name.replace('_', '-')}", str(val)])
        if cfg.get("allow_extra_args"):
            extra_args = str(casted.get("extra_args", "")).strip()
            if extra_args:
                cmd.extend(shlex.split(extra_args))
        cmd.extend(["--output-dir", str(output_dir)])

        task = {
            "task_id": task_id,
            "script": script_name,
            "status": "queued",
            "message": "任务排队中",
            "progress": 0,
            "params": casted,
            "cmd": cmd,
            "logs": [],
            "result": None,
            "pid": None,
            "output_dir": str(output_dir),
            "log_file": str(_run_log_path(output_dir)),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": None,
            "finished_at": None,
            "start_ts": None,
        }
        _run_log_path(output_dir).write_text("", encoding="utf-8")
        with _backtest_task_lock:
            _backtest_tasks[task_id] = task
            _save_task_meta(task)

        thread = threading.Thread(target=_run_backtest_task, args=(task_id,), daemon=True)
        thread.start()

        return jsonify({
            "status": "success",
            "message": "回测任务已启动",
            "data": {
                "task_id": task_id,
                "status_url": url_for("api_backtest_task_status", task_id=task_id),
                "result_url": url_for("backtest_result_page", task_id=task_id),
            },
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/backtest/task/<task_id>')
def api_backtest_task_status(task_id: str):
    """API: 获取回测任务状态与日志"""
    with _backtest_task_lock:
        task = _backtest_tasks.get(task_id)
        if task:
            if task["status"] == "running" and task.get("start_ts"):
                task["progress"] = _estimate_running_progress(task["start_ts"])
            data = {
                "task_id": task_id,
                "script": task["script"],
                "status": task["status"],
                "message": task["message"],
                "progress": task["progress"],
                "created_at": task["created_at"],
                "started_at": task["started_at"],
                "finished_at": task["finished_at"],
                "logs": task["logs"][-100:],
                "params": task["params"],
                "output_dir": task["output_dir"],
            }
            return jsonify({"status": "success", "data": data})

    meta = _load_task_meta(task_id)
    run_dir = BACKTEST_RUNS_DIR / task_id
    if not meta and not run_dir.exists():
        return jsonify({"status": "error", "message": "任务不存在"})
    if not meta:
        inferred_status = "success" if (run_dir / "metrics.json").exists() else "unknown"
        meta = {
            "task_id": task_id,
            "script": "unknown",
            "status": inferred_status,
            "message": "从本地目录恢复",
            "progress": 100 if inferred_status == "success" else 0,
            "params": {},
            "output_dir": str(run_dir),
            "created_at": datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": None,
            "finished_at": None,
        }
    logs = _tail_lines(_run_log_path(Path(meta.get("output_dir", run_dir))), 100)
    data = {
        "task_id": task_id,
        "script": meta.get("script"),
        "status": meta.get("status", "unknown"),
        "message": meta.get("message", ""),
        "progress": int(meta.get("progress", 100 if meta.get("status") in ("success", "error") else 0)),
        "created_at": meta.get("created_at"),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "logs": logs,
        "params": meta.get("params", {}),
        "output_dir": meta.get("output_dir"),
    }
    return jsonify({"status": "success", "data": data})


@app.route('/api/backtest/result/<task_id>')
def api_backtest_result(task_id: str):
    """API: 获取回测结果详情"""
    with _backtest_task_lock:
        task = _backtest_tasks.get(task_id)
        if task:
            if task["status"] != "success":
                return jsonify({"status": "error", "message": f"任务尚未完成，当前状态: {task['status']}"})
            data = {
                "task_id": task_id,
                "script": task["script"],
                "params": task["params"],
                "created_at": task["created_at"],
                "started_at": task["started_at"],
                "finished_at": task["finished_at"],
                "result": task.get("result") or {},
                "logs": task["logs"][-200:],
                "output_dir": task["output_dir"],
            }
            return jsonify({"status": "success", "data": data})

    meta = _load_task_meta(task_id)
    run_dir = BACKTEST_RUNS_DIR / task_id
    if not meta and not run_dir.exists():
        return jsonify({"status": "error", "message": "任务不存在"})
    if not meta:
        if not (run_dir / "metrics.json").exists():
            return jsonify({"status": "error", "message": "任务尚未完成，且未找到结果文件"})
        meta = {
            "task_id": task_id,
            "script": "unknown",
            "status": "success",
            "params": {},
            "created_at": datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": None,
            "finished_at": None,
            "output_dir": str(run_dir),
        }
    if meta.get("status") != "success":
        return jsonify({"status": "error", "message": f"任务尚未完成，当前状态: {meta.get('status', 'unknown')}"})
    output_dir = Path(meta.get("output_dir", run_dir))
    data = {
        "task_id": task_id,
        "script": meta.get("script"),
        "params": meta.get("params", {}),
        "created_at": meta.get("created_at"),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "result": _load_backtest_result(output_dir),
        "logs": _tail_lines(_run_log_path(output_dir), 200),
        "output_dir": str(output_dir),
    }
    return jsonify({"status": "success", "data": data})


@app.route('/api/backtest/history')
def api_backtest_history():
    """API: 获取历史回测结果列表"""
    try:
        limit = request.args.get("limit", default=200, type=int)
        items = _load_history_items(limit=max(1, min(1000, limit)))
        return jsonify({"status": "success", "data": {"items": items}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/backtest/custom/list')
def api_backtest_custom_list():
    """API: 列出自定义回测脚本"""
    try:
        items = []
        for name in _list_custom_backtest_scripts():
            p = CUSTOM_BACKTEST_DIR / name
            items.append({
                "name": name,
                "updated_at": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_bytes": int(p.stat().st_size),
            })
        return jsonify({"status": "success", "data": {"scripts": items}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/backtest/custom/content')
def api_backtest_custom_content():
    """API: 读取自定义回测脚本内容"""
    try:
        name = request.args.get("name", "").strip()
        script_path = _resolve_custom_script_path(name)
        if script_path is None or not script_path.exists():
            return jsonify({"status": "error", "message": "脚本不存在"})
        content = script_path.read_text(encoding="utf-8", errors="ignore")
        return jsonify({"status": "success", "data": {"name": name, "content": content}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/backtest/custom/save', methods=['POST'])
def api_backtest_custom_save():
    """API: 保存自定义回测脚本"""
    try:
        payload = request.get_json() or {}
        name = str(payload.get("name", "")).strip()
        content = str(payload.get("content", ""))
        script_path = _resolve_custom_script_path(name)
        if script_path is None:
            return jsonify({"status": "error", "message": "脚本名称非法，仅支持 *.py"})
        if not content.strip():
            return jsonify({"status": "error", "message": "脚本内容不能为空"})
        script_path.write_text(content, encoding="utf-8")
        return jsonify({"status": "success", "message": "脚本已保存", "data": {"name": name}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ==================== 新版 API（按因子宽格式数据） ====================

@app.route('/api/factors/list')
def api_factors_list():
    """API: 列出所有可用的因子文件"""
    try:
        # 获取所有因子名称
        factors = list_all_factor_names()
        if not factors:
            return jsonify({'status': 'error', 'message': '没有找到因子文件'})

        # 获取年份信息
        df = list_available_factors()
        years = sorted([int(y) for y in df['year'].unique()]) if not df.empty else []

        # 构建文件信息
        files = []
        if not df.empty:
            for _, row in df.iterrows():
                files.append({
                    'filename': str(row['filename']),
                    'prefix': str(row['prefix']),
                    'factor': str(row['factor']),
                    'year': int(row['year']),
                    'file_size_mb': float(row['file_size_mb']),
                    'file_path': str(row['file_path'])
                })

        return jsonify({
            'status': 'success',
            'data': {
                'factors': factors,
                'years': years,
                'files': files
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/factor/info')
def api_factor_info():
    """API: 获取因子数据汇总信息"""
    try:
        df = list_available_factors()
        if df.empty:
            return jsonify({'status': 'error', 'message': '没有找到因子文件'})

        # 按因子统计
        factor_summary = []
        for factor in df['factor'].unique():
            factor_files = df[df['factor'] == factor]
            years = [int(y) for y in factor_files['year'].tolist()]
            total_size = float(factor_files['file_size_mb'].sum())

            factor_summary.append({
                'factor': str(factor),
                'years': years,
                'file_count': len(factor_files),
                'total_size_mb': total_size
            })

        # 按年份统计
        year_summary = []
        for year in sorted(df['year'].unique()):
            year_files = df[df['year'] == year]
            year_summary.append({
                'year': int(year),
                'file_count': len(year_files),
                'factors': [str(f) for f in year_files['factor'].tolist()]
            })

        return jsonify({
            'status': 'success',
            'data': {
                'data_source': str(get_by_factor_dir()),
                'factor_summary': factor_summary,
                'year_summary': year_summary
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/factor/data')
def api_factor_data():
    """API: 获取因子数据（宽格式）"""
    try:
        factor = request.args.get('factor')
        year = request.args.get('year', type=int)

        if not factor:
            return jsonify({'status': 'error', 'message': '缺少因子名称参数'})

        # 加载因子数据
        if year:
            df = load_factor_data(factor, year)
            desc = f"{factor} - {year}年"
        else:
            # 加载所有年份
            df_list = []
            for f in get_by_factor_dir().glob(f"zz1000_{factor}_*.parquet"):
                df_list.append(pd.read_parquet(f))
            if df_list:
                df = pd.concat(df_list).sort_index()
                desc = f"{factor} - 所有年份"
            else:
                return jsonify({'status': 'error', 'message': '未找到因子数据'})

        if df is None or df.empty:
            return jsonify({'status': 'error', 'message': '因子数据为空'})

        # 返回数据信息
        return jsonify({
            'status': 'success',
            'data': {
                'factor': factor,
                'year': year or 'all',
                'description': desc,
                'shape': {'rows': len(df), 'cols': len(df.columns)},
                'date_range': {
                    'start': str(df.index.min()),
                    'end': str(df.index.max())
                },
                'stock_count': len(df.columns),
                'date_count': len(df)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/factor/sample')
def api_factor_sample():
    """API: 获取因子数据样例"""
    try:
        factor = request.args.get('factor')
        year = request.args.get('year', type=int)

        # 加载因子数据
        if year:
            df = load_factor_data(factor, year)
        else:
            files = sorted(get_by_factor_dir().glob(f"zz1000_{factor}_*.parquet"))
            if not files:
                return jsonify({'status': 'error', 'message': '未找到因子数据'})
            df = pd.read_parquet(files[0])

        if df is None or df.empty:
            return jsonify({'status': 'error', 'message': '因子数据为空'})

        # 转换为长格式样例
        sample = df.head(10)

        # 构建返回数据
        records = []
        for date_idx in sample.index:
            for stock in sample.columns[:20]:  # 只返回前20只股票
                value = sample.loc[date_idx, stock]
                records.append({
                    'date': str(date_idx),
                    'stock': stock,
                    'factor': round(float(value), 6) if pd.notna(value) else None
                })

        return jsonify({
            'status': 'success',
            'data': {
                'factor': factor,
                'filename': f"zz1000_{factor}_{year or df.index.year.min()}.parquet",
                'sample_data': records,
                'total_rows': len(df),
                'total_stocks': len(df.columns)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/factor/stats')
def api_factor_stats():
    """API: 计算因子统计信息"""
    try:
        factor = request.args.get('factor')
        year = request.args.get('year', type=int)

        # 加载因子数据
        if year:
            df = load_factor_data(factor, year)
        else:
            files = sorted(get_by_factor_dir().glob(f"zz1000_{factor}_*.parquet"))
            if not files:
                return jsonify({'status': 'error', 'message': '未找到因子数据'})
            df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()

        if df is None or df.empty:
            return jsonify({'status': 'error', 'message': '因子数据为空'})

        # 计算统计信息
        flat_values = df.values.flatten()
        flat_values = flat_values[~np.isnan(flat_values)]

        stats = {
            'factor': factor,
            'count': len(flat_values),
            'mean': float(np.mean(flat_values)),
            'std': float(np.std(flat_values)),
            'min': float(np.min(flat_values)),
            'max': float(np.max(flat_values)),
            'median': float(np.median(flat_values)),
            'skew': float(pd.Series(flat_values).skew()),
            'kurtosis': float(pd.Series(flat_values).kurtosis())
        }

        # 计算每日的均值和标准差
        daily_means = df.mean(axis=1)
        daily_stds = df.std(axis=1)
        stats['daily_mean_std'] = {
            'mean': float(daily_means.mean()),
            'std': float(daily_means.std())
        }

        return jsonify({'status': 'success', 'data': stats})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/factor-quick')
def api_factor_quick():
    """兼容旧版页面: 快速因子文件信息"""
    try:
        factor_dir = get_by_factor_dir()
        if not factor_dir.exists():
            return jsonify({'status': 'error', 'message': f'目录不存在: {factor_dir}'})

        files = sorted(factor_dir.glob("*.parquet"))
        file_items = []
        large_files = []

        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            is_large = size_mb > 80
            file_info = {
                "filename": f.name,
                "file_size_mb": round(size_mb, 2),
                "is_large": is_large,
            }
            file_items.append(file_info)
            if is_large:
                large_files.append({
                    "filename": f.name,
                    "size_mb": round(size_mb, 2),
                    "reason": "文件较大，详情加载会做延迟处理",
                })

        return jsonify({
            "status": "success",
            "data": {
                "data_source": str(factor_dir),
                "total_files": len(file_items),
                "zz1000_files": len([x for x in file_items if x["filename"].startswith("zz1000_")]),
                "daily_files": len([x for x in file_items if x["filename"].startswith("daily_")]),
                "files": file_items,
                "large_files": large_files,
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/factor-file-info/<path:filename>')
def api_factor_file_info(filename):
    """兼容旧版页面: 文件详情"""
    try:
        factor_dir = get_by_factor_dir().resolve()
        file_path = (factor_dir / filename).resolve()
        if factor_dir not in file_path.parents or not file_path.exists():
            return jsonify({'status': 'error', 'message': '文件不存在或路径非法'})
        if file_path.suffix != ".parquet":
            return jsonify({'status': 'error', 'message': '仅支持 parquet 文件'})

        df = pd.read_parquet(file_path)
        date_values = pd.to_datetime(df.index, errors="coerce")
        date_values = date_values[~pd.isna(date_values)]
        date_start = str(date_values.min().date()) if len(date_values) > 0 else None
        date_end = str(date_values.max().date()) if len(date_values) > 0 else None

        return jsonify({
            "status": "success",
            "data": {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "record_count": int(len(df)),
                "stock_count": int(len(df.columns)),
                "date_range": {"start": date_start, "end": date_end},
                "columns": [str(c) for c in df.columns.tolist()],
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/factor-sample')
def api_factor_sample_legacy():
    """兼容旧版页面: 因子样例"""
    try:
        files = sorted(get_by_factor_dir().glob("*.parquet"))
        if not files:
            return jsonify({'status': 'error', 'message': '未找到任何因子文件'})
        sample_file = files[0]
        df = pd.read_parquet(sample_file).head(10)

        table_df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        columns = [str(c) for c in table_df.columns.tolist()]
        sample_data = table_df.where(pd.notnull(table_df), None).to_dict(orient="records")

        return jsonify({
            "status": "success",
            "data": {
                "filename": sample_file.name,
                "total_rows": int(len(df)),
                "columns": columns,
                "sample_data": sample_data,
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/docs/index')
def api_docs_index():
    """文档索引与搜索"""
    try:
        query_raw = (request.args.get("q") or "").strip()
        query = query_raw.lower()
        selected_tag = (request.args.get("tag") or "").strip()
        sort_by = (request.args.get("sort_by") or "created_at").strip()
        order = (request.args.get("order") or "desc").strip().lower()
        docs = _build_doc_index()
        tag_counts = {}
        for d in docs:
            for tag in d["tags"]:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if selected_tag:
            docs = [d for d in docs if selected_tag in d["tags"]]
        if query:
            docs = [
                d for d in docs
                if query in d["title"].lower() or query in d["path"].lower() or query in d["content"].lower()
            ]

        reverse = order != "asc"
        if sort_by == "updated_at":
            docs.sort(key=lambda x: x.get("updated_at_ts", 0), reverse=reverse)
        elif sort_by == "title":
            docs.sort(key=lambda x: x.get("title", "").lower(), reverse=reverse)
        else:
            docs.sort(key=lambda x: x.get("created_at_ts", 0), reverse=reverse)

        data = []
        for d in docs:
            snippet = d["snippet"]
            title = d["title"]
            if query:
                snippet = _highlight_text(snippet, query_raw)
                title = _highlight_text(title, query_raw)
            data.append({
                "id": d["id"],
                "title": title,
                "path": d["path"],
                "snippet": snippet,
                "tags": d["tags"],
                "created_at": d["created_at"],
                "updated_at": d["updated_at"],
            })
        return jsonify({
            "status": "success",
            "data": {
                "total": len(data),
                "docs": data,
                "tags": sorted([{"name": k, "count": v} for k, v in tag_counts.items()], key=lambda x: (-x["count"], x["name"])),
                "sort_by": sort_by,
                "order": order,
            },
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/docs/content/<doc_id>')
def api_docs_content(doc_id):
    """文档详情"""
    try:
        docs = _build_doc_index()
        matched = next((d for d in docs if d["id"] == doc_id), None)
        if matched is None:
            return jsonify({"status": "error", "message": "文档不存在"})
        return jsonify({
            "status": "success",
            "data": {
                "id": matched["id"],
                "title": matched["title"],
                "path": matched["path"],
                "html": _render_markdown(matched["content"]),
                "raw": matched["content"],
                "created_at": matched["created_at"],
                "updated_at": matched["updated_at"],
            },
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/factor/ic')
def api_factor_ic():
    """API: 计算因子IC"""
    try:
        factor = request.args.get('factor')
        year = request.args.get('year', type=int)
        period = request.args.get('period', default=1, type=int)

        merged, err = _load_factor_returns_merged(factor=factor, period=period, year=year)
        if err:
            return jsonify({'status': 'error', 'message': err})

        # 计算IC
        returns_col = f'return_{period}d'
        ic_series = merged.groupby('date').apply(
            lambda x: x['factor_value'].corr(x[returns_col])
        )

        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        ic_ir = ic_mean / ic_std if ic_std != 0 else 0
        ic_positive_ratio = float((ic_series > 0).mean())

        return jsonify({
            'status': 'success',
            'data': {
                'factor': factor,
                'period': period,
                'ic_series': [{'date': str(d), 'ic': float(v)} for d, v in ic_series.items()],
                'ic_stats': {
                    'ic_mean': ic_mean,
                    'ic_std': ic_std,
                    'ic_ir': ic_ir,
                    'ic_positive_ratio': ic_positive_ratio
                }
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/factor/cross-correlation')
def api_factor_cross_correlation():
    """API: 计算因子之间的相关性"""
    try:
        factors = request.args.getlist('factors')
        year = request.args.get('year', type=int)

        if len(factors) < 2:
            return jsonify({'status': 'error', 'message': '至少需要两个因子'})

        # 加载因子数据
        dfs = {}
        for factor in factors:
            if year:
                df = load_factor_data(factor, year)
            else:
                files = sorted(get_by_factor_dir().glob(f"zz1000_{factor}_*.parquet"))
                if files:
                    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
                else:
                    continue
            if df is not None:
                dfs[factor] = df

        if len(dfs) < 2:
            return jsonify({'status': 'error', 'message': '无法加载足够的因子数据'})

        # 合并数据并计算相关性
        common_dates = sorted(set.intersection(*[set(df.index) for df in dfs.values()]))

        corr_data = {}
        for f1 in factors:
            if f1 not in dfs:
                continue
            corr_data[f1] = {}
            for f2 in factors:
                if f2 not in dfs:
                    continue
                # 计算截面相关性
                common_df = pd.DataFrame({
                    f1: dfs[f1].loc[common_dates].mean(axis=1),
                    f2: dfs[f2].loc[common_dates].mean(axis=1),
                }).dropna()
                if len(common_df) > 10:
                    corr_data[f1][f2] = round(common_df[f1].corr(common_df[f2]), 4)
                else:
                    corr_data[f1][f2] = None

        return jsonify({
            'status': 'success',
            'data': {
                'factors': factors,
                'year': year,
                'correlation_matrix': corr_data
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/quantile')
def api_quantile():
    """API: 分层分析"""
    try:
        factor = request.args.get('factor')
        period = request.args.get('period', default=1, type=int)
        quantiles = request.args.get('quantiles', default=5, type=int)

        if not factor:
            return jsonify({'status': 'error', 'message': '缺少因子参数'})

        merged, err = _load_factor_returns_merged(factor=factor, period=period)
        if err:
            return jsonify({'status': 'error', 'message': err})
        return jsonify({'status': 'success', 'data': _calc_quantile_result(merged, period, quantiles)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/long-short')
def api_long_short():
    """API: 多空组合分析"""
    try:
        factor = request.args.get('factor')
        period = request.args.get('period', default=1, type=int)

        if not factor:
            return jsonify({'status': 'error', 'message': '缺少因子参数'})

        merged, err = _load_factor_returns_merged(factor=factor, period=period)
        if err:
            return jsonify({'status': 'error', 'message': err})

        # 计算每日因子分位数
        returns_col = f'return_{period}d'
        merged['quantile'] = merged.groupby('date')['factor_value'].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1
        )

        # 计算多空收益
        top_returns = merged[merged['quantile'] == 5].groupby('date')[returns_col].mean()
        bottom_returns = merged[merged['quantile'] == 1].groupby('date')[returns_col].mean()
        ls_returns = top_returns - bottom_returns

        # 统计指标
        total_return = (1 + ls_returns).prod() - 1
        mean_daily = ls_returns.mean()
        std_daily = ls_returns.std()
        sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily != 0 else 0
        win_rate = (ls_returns > 0).mean()

        # 计算最大回撤
        cumulative = (1 + ls_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return jsonify({
            'status': 'success',
            'data': {
                'factor': factor,
                'period': period,
                'total_return': float(total_return),
                'mean_daily': float(mean_daily),
                'std_daily': float(std_daily),
                'sharpe': float(sharpe),
                'win_rate': float(win_rate),
                'max_drawdown': float(max_drawdown)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==================== 原有 API（保持兼容性） ====================

@app.route('/api/summary')
def api_summary():
    """API: 获取汇总数据"""
    try:
        # 尝试使用新版因子列表
        factors_df = list_available_factors()
        if not factors_df.empty:
            years = sorted([int(y) for y in factors_df['year'].unique()])

            summary = {
                'total_files': int(len(factors_df)),
                'factor_count': int(len(factors_df['factor'].unique())),
                'factors': [str(f) for f in factors_df['factor'].unique()],
                'years': years,
                'data_source': str(get_by_factor_dir()),
                'date_range': {
                    'start': f"{years[0]}-01-01" if years else None,
                    'end': f"{years[-1]}-12-31" if years else None
                }
            }
            return jsonify({'status': 'success', 'data': summary})

        # 回退到原有方式
        global factor_analyzer
        if factor_analyzer is None:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            factor_analyzer = FactorAnalysis(os.path.join(base_path, "factor/daily"))
            try:
                factor_analyzer.load_all_factors()
                factor_analyzer.compute_returns(periods=[1, 5, 10])
            except:
                pass
        analyzer = factor_analyzer

        if analyzer and analyzer.factors_df is not None:
            summary = {
                'total_records': int(len(analyzer.factors_df)),
                'total_stocks': int(analyzer.factors_df['stock_code'].nunique()),
                'total_dates': int(analyzer.factors_df['date'].nunique()),
                'factor_count': len(analyzer.factor_cols),
                'date_range': {
                    'start': str(analyzer.factors_df['date'].min()),
                    'end': str(analyzer.factors_df['date'].max())
                },
                'factors': analyzer.factor_cols
            }
            return jsonify({'status': 'success', 'data': summary})

        return jsonify({'status': 'error', 'message': '无法加载数据'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/descriptive-stats')
def api_descriptive_stats():
    """API: 描述性统计"""
    try:
        analyzer = get_analyzer()
        stats = analyzer.descriptive_stats()
        return jsonify({'status': 'success', 'data': stats.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/correlation')
def api_correlation():
    """API: 相关性矩阵"""
    try:
        analyzer = get_analyzer()
        corr = analyzer.factor_correlation(method='spearman')
        return jsonify({'status': 'success', 'data': corr.to_dict()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/ic-stats')
def api_ic_stats():
    """API: IC统计"""
    try:
        analyzer = get_analyzer()
        ic_result = analyzer.ic_analysis(return_col='return_1', by='date')
        stats = ic_result['ic_stats'].reset_index()
        stats.columns = ['factor', 'ic_mean', 'ic_std', 'ic_ir', 'ic_positive_ratio', 'ic_t_stat']
        return jsonify({'status': 'success', 'data': stats.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/ic-series')
def api_ic_series():
    """API: IC时间序列"""
    try:
        analyzer = get_analyzer()
        ic_result = analyzer.ic_analysis(return_col='return_1', by='date')

        series_data = {}
        for factor, ic_vals in ic_result['ic_series'].items():
            series_data[factor] = [
                {'date': str(d), 'ic': float(v)} for d, v in zip(ic_result['ic_series'][factor].index, ic_vals)
            ]

        return jsonify({'status': 'success', 'data': series_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/ic-decay')
def api_ic_decay():
    """API: IC衰减"""
    try:
        analyzer = get_analyzer()
        decay = analyzer.ic_decay_analysis(max_lag=20)
        return jsonify({'status': 'success', 'data': decay.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/quantile/<factor>')
def api_quantile_by_factor(factor):
    """API: 分层分析（带因子参数）"""
    try:
        period = request.args.get('period', default=1, type=int)
        quantiles = request.args.get('quantiles', default=5, type=int)

        merged, err = _load_factor_returns_merged(factor=factor, period=period)
        if err:
            return jsonify({'status': 'error', 'message': err})
        return jsonify({'status': 'success', 'data': _calc_quantile_result(merged, period, quantiles)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==================== 页面路由 ====================

@app.route('/factor-info')
def factor_info():
    """因子信息查看页面"""
    return render_template('factor_info.html', active_page='factor-info')

@app.route('/factor-viewer')
def factor_viewer():
    """因子数据查看器页面"""
    return render_template('factor_viewer.html', active_page='factor-viewer')

# ==================== 数据管理 API ====================

@app.route('/api/data/status')
def api_data_status():
    """API: 获取daily_data目录状态"""
    try:
        # 获取各类数据的最新日期
        latest_dates = get_all_latest_dates()

        # 获取交易日信息
        try:
            pro = init_tushare()
            today = get_today_str()
            trade_dates = pro.trade_cal(
                exchange='SSE',
                start_date=today,
                end_date=today,
                is_open='1'
            )
            is_trading_day = len(trade_dates) > 0
        except:
            is_trading_day = False

        # 数据表配置
        data_tables = {
            'daily': {
                'name': '日线行情',
                'description': 'OHLCV行情数据',
                'fields': ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'],
                'directory': 'daily_data/daily/'
            },
            'daily_basic': {
                'name': '每日基本面',
                'description': '每日基本面指标',
                'fields': ['ts_code', 'trade_date', 'close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv'],
                'directory': 'daily_data/daily_basic/'
            },
            'cashflow_daily': {
                'name': '现金流量表',
                'description': '每日推算的现金流量数据',
                'fields': ['ts_code', 'trade_date', 'n_cashflow_act', 'n_cashflow_inv_act', 'n_cash_flows_fnc_act', 'c_fr_sale_sg', 'c_paid_goods_s', 'net_profit'],
                'directory': 'daily_data/cashflow_daily/'
            },
            'income_daily': {
                'name': '利润表',
                'description': '每日推算的利润数据',
                'fields': ['ts_code', 'trade_date', 'total_revenue', 'revenue', 'oper_cost', 'operate_profit', 'total_profit', 'n_income', 'basic_eps'],
                'directory': 'daily_data/income_daily/'
            },
            'balance_daily': {
                'name': '资产负债表',
                'description': '每日推算的资产负债数据',
                'fields': ['ts_code', 'trade_date', 'total_assets', 'total_liab', 'total_hldr_eqy_exc_min_int', 'total_cur_assets', 'total_cur_liab', 'cash_reser_cb'],
                'directory': 'daily_data/balance_daily/'
            }
        }

        # 构建返回数据
        tables_info = []
        for table_id, config in data_tables.items():
            latest = latest_dates.get(table_id)
            tables_info.append({
                'id': table_id,
                'name': config['name'],
                'description': config['description'],
                'fields': config['fields'],
                'latest_date': latest,
                'directory': config['directory'],
                'status': 'updated' if latest else 'empty'
            })

        return jsonify({
            'status': 'success',
            'data': {
                'tables': tables_info,
                'is_trading_day': is_trading_day,
                'is_after_market_close': is_after_market_close(),
                'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'today': get_today_str()
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/data/update', methods=['POST'])
def api_data_update():
    """API: 执行数据更新"""
    try:
        import threading
        import subprocess

        data = request.get_json() or {}
        update_type = data.get('type', 'all')
        include_today = data.get('include_today', is_after_market_close())

        # 启动后台任务执行更新
        def run_update():
            try:
                cmd = [sys.executable, 'update_data.py']
                if update_type == 'daily':
                    cmd.extend(['--daily'])
                elif update_type == 'daily_basic':
                    cmd.extend(['--daily-basic'])
                elif update_type == 'financial':
                    cmd.extend(['--financial'])

                if include_today:
                    cmd.append('--include-today')

                subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            except Exception as e:
                print(f"数据更新错误: {e}")

        # 后台执行
        thread = threading.Thread(target=run_update)
        thread.start()

        return jsonify({
            'status': 'success',
            'message': '数据更新任务已启动',
            'data': {
                'update_type': update_type,
                'include_today': include_today
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/data/update/sync', methods=['POST'])
def api_data_update_sync():
    """API: 同步执行数据更新（返回进度）"""
    try:
        data = request.get_json() or {}
        update_type = data.get('type', 'all')
        include_today = data.get('include_today', is_after_market_close())
        valid_types = {'all', 'daily', 'daily_basic', 'financial'}

        if update_type not in valid_types:
            return jsonify({
                'status': 'error',
                'message': f'不支持的更新类型: {update_type}'
            })

        table_to_path = {
            'daily': BASE_DIR / 'daily_data' / 'daily',
            'daily_basic': BASE_DIR / 'daily_data' / 'daily_basic',
            'cashflow_daily': BASE_DIR / 'daily_data' / 'cashflow_daily',
            'income_daily': BASE_DIR / 'daily_data' / 'income_daily',
            'balance_daily': BASE_DIR / 'daily_data' / 'balance_daily',
        }
        table_to_prefix = {
            'daily': 'daily',
            'daily_basic': 'daily_basic',
            'cashflow_daily': 'cashflow_daily',
            'income_daily': 'income_daily',
            'balance_daily': 'balance_daily',
        }

        def _collect_existing_dates(table_name: str) -> set:
            base_dir = table_to_path[table_name]
            prefix = table_to_prefix[table_name]
            dates = set()
            if not base_dir.exists():
                return dates
            for file_path in base_dir.rglob(f'{prefix}_*.parquet'):
                m = re.match(rf'^{prefix}_(\d{{8}})(?:_.*)?\.parquet$', file_path.name)
                if m:
                    dates.add(m.group(1))
            return dates

        # 执行更新
        try:
            pro = init_tushare()
            before_dates = {
                name: _collect_existing_dates(name) for name in table_to_path.keys()
            }

            # 确定更新范围
            latest_dates = get_all_latest_dates()
            if update_type == 'daily':
                latest = latest_dates.get('daily') or '20250101'
            elif update_type == 'daily_basic':
                latest = latest_dates.get('daily_basic') or '20250101'
            else:
                latest = max(
                    latest_dates.get('daily') or '20250101',
                    latest_dates.get('daily_basic') or '20250101'
                )

            trade_dates = []
            if update_type in ('all', 'daily', 'daily_basic'):
                # 获取需要更新的交易日
                today = get_today_str()
                trade_dates = pro.trade_cal(
                    exchange='SSE',
                    start_date=latest,
                    end_date=today,
                    is_open='1'
                )['cal_date'].tolist()
                # Tushare 返回顺序不稳定，这里统一升序，确保后续截取的是最近交易日
                trade_dates = sorted(set(trade_dates))

                if not trade_dates:
                    return jsonify({
                        'status': 'success',
                        'message': '没有需要更新的交易日',
                        'data': {'updated_count': 0}
                    })

                # 如果不包含今天，过滤掉
                if not include_today and not is_after_market_close():
                    trade_dates = [d for d in trade_dates if d != today]

                if not trade_dates:
                    return jsonify({
                        'status': 'success',
                        'message': '没有需要更新的交易日',
                        'data': {'updated_count': 0}
                    })

                # 限制为最新5个交易日
                trade_dates = trade_dates[-5:]

            updated_count = 0

            if update_type in ('all', 'daily'):
                from update_data import download_daily_data
                download_daily_data(pro, trade_dates[0], trade_dates[-1])

            if update_type in ('all', 'daily_basic'):
                from update_data import download_daily_basic_data
                download_daily_basic_data(pro, trade_dates[0], trade_dates[-1])

            if update_type == 'financial':
                update_financial_data()

            after_dates = {
                name: _collect_existing_dates(name) for name in table_to_path.keys()
            }

            updated_details = {}
            for name in table_to_path.keys():
                updated_details[name] = sorted(after_dates[name] - before_dates[name])

            if update_type == 'daily':
                updated_count = len(updated_details['daily'])
            elif update_type == 'daily_basic':
                updated_count = len(updated_details['daily_basic'])
            elif update_type == 'financial':
                updated_count = (
                    len(updated_details['cashflow_daily']) +
                    len(updated_details['income_daily']) +
                    len(updated_details['balance_daily'])
                )
            else:
                updated_count = (
                    len(updated_details['daily']) +
                    len(updated_details['daily_basic'])
                )

            if updated_count > 0:
                message = f'数据更新完成，新增 {updated_count} 个数据文件'
            else:
                message = f'更新完成，无新增数据（检查了 {len(trade_dates)} 个交易日）'

            return jsonify({
                'status': 'success',
                'message': message,
                'data': {
                    'updated_count': updated_count,
                    'trade_dates': trade_dates,
                    'update_type': update_type,
                    'updated_details': updated_details
                }
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'更新失败: {str(e)}'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/data/refresh', methods=['POST'])
def api_data_refresh():
    """API: 刷新数据状态"""
    try:
        # 重新获取最新数据状态
        latest_dates = get_all_latest_dates()

        return jsonify({
            'status': 'success',
            'message': '数据状态已刷新',
            'data': {
                'latest_dates': latest_dates,
                'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/data/high-frequency/compute', methods=['POST'])
def api_high_frequency_compute():
    """API: 根据tick原始数据计算并保存单日高频因子"""
    try:
        data = request.get_json() or {}
        date_raw = str(data.get('date', '')).strip()
        stock_code = data.get('stock_code', 'all')

        if not date_raw:
            return jsonify({'status': 'error', 'message': '请提供日期'})

        date_digits = date_raw.replace('-', '').replace('/', '')
        if not re.fullmatch(r'\d{8}', date_digits):
            return jsonify({'status': 'error', 'message': '日期格式错误，支持 YYYYMMDD 或 YYYY-MM-DD'})

        date_norm = f"{date_digits[:4]}-{date_digits[4:6]}-{date_digits[6:8]}"
        tick_base_dir = BASE_DIR / 'tick_2026'
        tick_day_dir = tick_base_dir / date_digits[:4] / date_digits[4:6] / date_digits[6:8]

        if not tick_day_dir.exists():
            return jsonify({
                'status': 'error',
                'message': f'tick数据目录不存在: {tick_day_dir}'
            })

        tick_files = list(tick_day_dir.glob('*.parquet'))
        if not tick_files:
            return jsonify({
                'status': 'error',
                'message': f'当天无tick数据文件: {tick_day_dir}'
            })

        output_dir = BASE_DIR / 'factor' / 'high_frequency'

        from time import perf_counter
        start_ts = perf_counter()

        from high_frequency_factors import calc_high_frequency, FACTORS
        result_df = calc_high_frequency(
            date=date_norm,
            stock_code=stock_code,
            base_dir=str(tick_base_dir),
            output_dir=str(output_dir)
        )

        elapsed_sec = round(perf_counter() - start_ts, 2)
        output_file = output_dir / f"{date_digits[:4]}_{date_digits[4:6]}_{date_digits[6:8]}.parquet"
        factor_cols = [f for f in FACTORS if f in result_df.columns]

        return jsonify({
            'status': 'success',
            'message': f'高频因子计算完成: {date_norm}',
            'data': {
                'date': date_norm,
                'tick_dir': str(tick_day_dir),
                'tick_file_count': len(tick_files),
                'stock_count': int(len(result_df)),
                'factor_count': len(factor_cols),
                'factors': factor_cols,
                'output_file': str(output_file),
                'elapsed_sec': elapsed_sec
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'高频因子计算失败: {str(e)}'})

# ==================== 登录路由 ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # 简单的认证（实际应用中应使用数据库存储）
        # 默认账号: admin/admin 或 user/user
        valid_users = {
            'admin': 'admin',
            'user': 'user'
        }

        if username in valid_users and valid_users[username] == password:
            session['user_id'] = username
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='用户名或密码错误')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """退出登录"""
    session.clear()
    return redirect(url_for('login'))


# ==================== 因子实验室路由 ====================

@app.route('/factor-lab')
def factor_lab():
    """因子分析实验室页面"""
    return render_template('factor_lab.html', active_page='factor-lab')


# ==================== 因子实验室 API ====================

@app.route('/api/factor/factories', methods=['POST'])
def api_factor_factories():
    """API: 获取可用因子列表"""
    try:
        data = request.get_json() or {}
        source = data.get('source')

        from mylib.factor_factory import get_factory

        factory = get_factory()

        if source:
            factors = factory.list_factors(source)
            return jsonify({
                'status': 'success',
                'data': {
                    'source': source,
                    'factors': factors
                }
            })

        # 返回所有因子来源
        return jsonify({
            'status': 'success',
            'data': {
                'sources': factory.list_sources(),
                'factors': factory.list_factors(),
                'stock_pools': factory.list_stock_pools()
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/factor/preview', methods=['POST'])
def api_factor_preview():
    """API: 预览预处理效果"""
    try:
        data = request.get_json()
        factor_name = data.get('factor')
        source = data.get('source', 'high_frequency')
        method = data.get('method')
        params = data.get('params', {})
        start_date = data.get('start')
        end_date = data.get('end')

        if not factor_name or not method:
            return jsonify({'status': 'error', 'message': '缺少因子名称或预处理方法'})

        from mylib.factor_factory import get_factory
        from mylib.factor_preprocessor import get_preprocessor

        factory = get_factory()
        preprocessor = get_preprocessor()

        # 加载因子数据（样本）
        factor_df = factory.get_factor(
            factor_name=factor_name,
            source=source,
            start_date=start_date,
            end_date=end_date
        )

        # 取样本数据
        sample_df = factor_df.head(20)

        # 预览预处理效果
        preview_result = preprocessor.preview(sample_df, method, **params)

        return jsonify({
            'status': 'success',
            'data': preview_result
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/factor/analyze', methods=['POST'])
def api_factor_analyze():
    """API: 执行因子分析"""
    try:
        data = request.get_json()

        factor = data.get('factor')
        source = data.get('source', 'high_frequency')
        stock_pool = data.get('stock_pool', 'zz1000')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        preprocess = data.get('preprocess') or {}
        preprocess_method = preprocess.get('method')
        preprocess_params = preprocess.get('params', {})

        config_params = data.get('config', {})
        returns_type = config_params.get('returns_type', 'close2close_next')
        returns_n = config_params.get('returns_n', 5)
        quantiles = config_params.get('quantiles', 5)

        if not factor:
            return jsonify({'status': 'error', 'message': '缺少因子参数'})

        from mylib.analysis_engine import AnalysisEngine, AnalysisConfig

        engine = get_analysis_engine()

        # 创建分析配置
        config = AnalysisConfig(
            factor_name=factor,
            source=source,
            stock_pool=stock_pool,
            start_date=start_date,
            end_date=end_date,
            preprocess_method=preprocess_method,
            preprocess_params=preprocess_params,
            returns_method=returns_type,
            returns_n=returns_n,
            quantiles=quantiles
        )

        # 执行分析
        result = engine.run_analysis(config)

        return jsonify({
            'status': 'success',
            'data': {
                'ic_stats': result.ic_stats,
                'quantile_returns': result.quantile_returns,
                'long_short_stats': result.long_short_stats,
                'turnover_rate': result.turnover_rate,
                'charts': result.charts_data
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/factor/save-result', methods=['POST'])
def api_factor_save_result():
    """API: 保存分析结果"""
    try:
        from mylib.analysis_engine import AnalysisEngine, AnalysisConfig

        data = request.get_json()

        factor = data.get('factor')
        source = data.get('source', 'high_frequency')
        stock_pool = data.get('stock_pool', 'zz1000')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        preprocess = data.get('preprocess') or {}
        preprocess_method = preprocess.get('method')
        preprocess_params = preprocess.get('params', {})

        config_params = data.get('config', {})
        returns_type = config_params.get('returns_type', 'close2close_next')
        returns_n = config_params.get('returns_n', 5)
        quantiles = config_params.get('quantiles', 5)

        engine = get_analysis_engine()

        config = AnalysisConfig(
            factor_name=factor,
            source=source,
            stock_pool=stock_pool,
            start_date=start_date,
            end_date=end_date,
            preprocess_method=preprocess_method,
            preprocess_params=preprocess_params,
            returns_method=returns_type,
            returns_n=returns_n,
            quantiles=quantiles
        )

        # 执行分析
        result = engine.run_analysis(config)

        # 保存结果
        save_path = engine.save_result(result)

        return jsonify({
            'status': 'success',
            'data': {
                'save_path': save_path,
                'result': result.to_dict()
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/returns/types', methods=['GET'])
def api_returns_types():
    """API: 获取收益率计算类型"""
    try:
        from mylib.returns_calculator import get_calculator

        calculator = get_calculator()
        methods = calculator.list_methods()

        return jsonify({
            'status': 'success',
            'data': {
                'methods': methods
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/stock-pools', methods=['GET'])
def api_stock_pools():
    """API: 获取股票池列表"""
    try:
        from mylib.factor_factory import get_factory

        factory = get_factory()
        pools = factory.list_stock_pools()

        return jsonify({
            'status': 'success',
            'data': {
                'stock_pools': pools
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# ==================== 启动 ====================

def get_analyzer():
    """懒加载分析器（保持兼容性）"""
    global factor_analyzer
    if factor_analyzer is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        factor_analyzer = FactorAnalysis(os.path.join(base_path, "factor/daily"))
        try:
            factor_analyzer.load_all_factors()
            factor_analyzer.compute_returns(periods=[1, 5, 10])
            print(f"Loaded {len(factor_analyzer.factors_df)} records")
        except Exception as e:
            print(f"Warning: Could not load data: {e}")
    return factor_analyzer


def get_analysis_engine():
    """懒加载因子实验室分析引擎"""
    global analysis_engine
    if analysis_engine is None:
        from mylib.analysis_engine import get_analysis_engine as _get_analysis_engine_impl
        analysis_engine = _get_analysis_engine_impl()
    return analysis_engine

if __name__ == '__main__':
    print("=" * 60)
    print("高频因子分析平台")
    print("=" * 60)
    print(f"\n因子数据目录: {get_by_factor_dir()}")
    print(f"启动服务器: http://localhost:{Config.PORT}")
    print(f"API文档: http://localhost:{Config.PORT}/api/factors/list")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 60)

    app.run(host='0.0.0.0', port=Config.PORT, debug=Config.DEBUG)
