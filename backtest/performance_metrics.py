import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _safe_float(val) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return None
        if pd.isna(val):
            return None
        out = float(val)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def build_equal_weight_benchmark_returns(panel: pd.DataFrame, symbols: list) -> pd.Series:
    """
    使用选股池的等权买入持有收益作为基准收益序列。
    """
    use = panel[panel["ts_code"].isin(set(symbols))][["trade_date", "ts_code", "close"]].copy()
    if use.empty:
        return pd.Series(dtype=float)
    px = use.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    bench = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).mean(axis=1)
    bench.index = pd.to_datetime(bench.index)
    return bench


def compute_performance_metrics(
    strategy_returns: pd.Series,
    initial_cash: float,
    final_value: Optional[float] = None,
    benchmark_returns: Optional[pd.Series] = None,
    periods_per_year: int = 252,
    risk_free_rate_annual: float = 0.0,
) -> Dict[str, Optional[float]]:
    """
    统一计算回测绩效指标。

    strategy_returns: 日收益率序列（如 TimeReturn 的 daily return）
    benchmark_returns: 可选，基准日收益率序列（同频率）
    """
    ret = pd.Series(strategy_returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if ret.empty:
        base = {
            "initial_cash": _safe_float(initial_cash),
            "final_value": _safe_float(final_value if final_value is not None else initial_cash),
            "observations": 0,
        }
        return base

    nav = (1.0 + ret).cumprod()
    obs = int(len(ret))
    total_return = nav.iloc[-1] - 1.0
    cagr = (nav.iloc[-1] ** (periods_per_year / obs) - 1.0) if nav.iloc[-1] > 0 else np.nan

    daily_rf = risk_free_rate_annual / periods_per_year
    excess = ret - daily_rf
    ann_vol = ret.std(ddof=0) * math.sqrt(periods_per_year)
    downside = np.minimum(excess, 0.0)
    downside_vol = np.sqrt((downside ** 2).mean()) * math.sqrt(periods_per_year)
    sharpe = (excess.mean() * periods_per_year) / ann_vol if ann_vol > 0 else np.nan
    sortino = (excess.mean() * periods_per_year) / downside_vol if downside_vol > 0 else np.nan

    cummax = nav.cummax()
    drawdown = nav / cummax - 1.0
    max_drawdown = abs(drawdown.min())
    max_drawdown_pct = max_drawdown * 100.0
    calmar = cagr / max_drawdown if max_drawdown > 0 else np.nan

    in_drawdown = drawdown < 0
    dd_durations = []
    streak = 0
    for flag in in_drawdown.values:
        if flag:
            streak += 1
        elif streak > 0:
            dd_durations.append(streak)
            streak = 0
    if streak > 0:
        dd_durations.append(streak)
    max_drawdown_duration = int(max(dd_durations) if dd_durations else 0)

    win_days = int((ret > 0).sum())
    loss_days = int((ret < 0).sum())
    flat_days = int((ret == 0).sum())
    win_rate = win_days / obs if obs > 0 else np.nan
    loss_rate = loss_days / obs if obs > 0 else np.nan

    pos = ret[ret > 0]
    neg = ret[ret < 0]
    avg_win = pos.mean() if not pos.empty else np.nan
    avg_loss = neg.mean() if not neg.empty else np.nan
    payoff_ratio = (avg_win / abs(avg_loss)) if avg_loss is not None and pd.notna(avg_loss) and avg_loss < 0 else np.nan
    profit_factor = (pos.sum() / abs(neg.sum())) if not neg.empty and abs(neg.sum()) > 0 else np.nan

    var_95 = ret.quantile(0.05)
    cvar_95 = ret[ret <= var_95].mean() if (ret <= var_95).any() else np.nan
    omega_ratio = (pos.sum() / abs(neg.sum())) if not neg.empty and abs(neg.sum()) > 0 else np.nan

    ulcer_index = np.sqrt(((drawdown * 100.0) ** 2).mean())

    metrics: Dict[str, Optional[float]] = {
        "initial_cash": _safe_float(initial_cash),
        "final_value": _safe_float(final_value if final_value is not None else initial_cash * nav.iloc[-1]),
        "observations": obs,
        "total_return": _safe_float(total_return),
        "annual_return": _safe_float(cagr),
        "annual_volatility": _safe_float(ann_vol),
        "downside_volatility": _safe_float(downside_vol),
        "sharpe": _safe_float(sharpe),
        "sortino": _safe_float(sortino),
        "calmar": _safe_float(calmar),
        "max_drawdown_pct": _safe_float(max_drawdown_pct),
        "max_drawdown_duration_days": _safe_float(max_drawdown_duration),
        "win_days": _safe_float(win_days),
        "loss_days": _safe_float(loss_days),
        "flat_days": _safe_float(flat_days),
        "win_rate": _safe_float(win_rate),
        "loss_rate": _safe_float(loss_rate),
        "avg_win_daily_return": _safe_float(avg_win),
        "avg_loss_daily_return": _safe_float(avg_loss),
        "payoff_ratio": _safe_float(payoff_ratio),
        "profit_factor": _safe_float(profit_factor),
        "best_day_return": _safe_float(ret.max()),
        "worst_day_return": _safe_float(ret.min()),
        "var_95_daily": _safe_float(var_95),
        "cvar_95_daily": _safe_float(cvar_95),
        "omega_ratio": _safe_float(omega_ratio),
        "skewness": _safe_float(ret.skew()),
        "kurtosis": _safe_float(ret.kurtosis()),
        "ulcer_index": _safe_float(ulcer_index),
    }

    if benchmark_returns is not None:
        bench = pd.Series(benchmark_returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        if not bench.empty:
            aligned = pd.concat([ret, bench], axis=1, join="inner").dropna()
            aligned.columns = ["strategy", "benchmark"]
            if not aligned.empty:
                ex = aligned["strategy"] - aligned["benchmark"]
                tracking_error = ex.std(ddof=0) * math.sqrt(periods_per_year)
                info_ratio = (ex.mean() * periods_per_year) / tracking_error if tracking_error > 0 else np.nan

                bm_var = aligned["benchmark"].var(ddof=0)
                beta = aligned["strategy"].cov(aligned["benchmark"]) / bm_var if bm_var > 0 else np.nan
                alpha = (
                    (aligned["strategy"].mean() - daily_rf) * periods_per_year
                    - (beta * ((aligned["benchmark"].mean() - daily_rf) * periods_per_year))
                ) if beta is not None and pd.notna(beta) else np.nan
                treynor = (
                    ((aligned["strategy"].mean() - daily_rf) * periods_per_year) / beta
                ) if beta is not None and pd.notna(beta) and beta != 0 else np.nan

                up = aligned[aligned["benchmark"] > 0]
                down = aligned[aligned["benchmark"] < 0]
                up_capture = (
                    up["strategy"].mean() / up["benchmark"].mean()
                ) if not up.empty and up["benchmark"].mean() != 0 else np.nan
                down_capture = (
                    down["strategy"].mean() / down["benchmark"].mean()
                ) if not down.empty and down["benchmark"].mean() != 0 else np.nan

                metrics.update({
                    "benchmark_total_return": _safe_float((1.0 + aligned["benchmark"]).prod() - 1.0),
                    "benchmark_annual_return": _safe_float((1.0 + aligned["benchmark"]).prod() ** (periods_per_year / len(aligned)) - 1.0),
                    "tracking_error": _safe_float(tracking_error),
                    "information_ratio": _safe_float(info_ratio),
                    "beta": _safe_float(beta),
                    "alpha": _safe_float(alpha),
                    "treynor_ratio": _safe_float(treynor),
                    "correlation_with_benchmark": _safe_float(aligned["strategy"].corr(aligned["benchmark"])),
                    "up_capture_ratio": _safe_float(up_capture),
                    "down_capture_ratio": _safe_float(down_capture),
                })

    return metrics
