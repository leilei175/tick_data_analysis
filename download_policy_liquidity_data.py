from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from akshare.bond.bond_china_money import __bond_register_service, bond_china_close_return_map


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "daily_data" / "hf_raw"
STANDARD_ROOT = ROOT / "daily_data" / "hf_standard"
DOC_ROOT = ROOT / "doc"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


@dataclass
class DownloadRecord:
    dataset_name: str
    status: str
    rows: int
    raw_path: Optional[str]
    standard_path: Optional[str]
    source_page: str
    note: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def fetch(url: str, verify: bool = True) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=30, verify=verify)
    resp.encoding = "utf-8"
    resp.raise_for_status()
    return resp


def normalize_text(soup: BeautifulSoup) -> str:
    return "\n".join(x.strip() for x in soup.stripped_strings if x.strip())


def pbc_list_articles(list_url: str, title_keyword: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(fetch(list_url).text, "html.parser")
    results: List[Dict[str, str]] = []
    seen = set()
    for a in soup.find_all("a", href=True):
        title = " ".join(a.get_text(" ", strip=True).split())
        if title_keyword not in title:
            continue
        href = urljoin(list_url, a["href"])
        if href in seen or href == list_url:
            continue
        seen.add(href)
        results.append({"title": title, "url": href})
    return results


def pbc_article_payload(article_url: str) -> Dict[str, str]:
    soup = BeautifulSoup(fetch(article_url).text, "html.parser")
    text = normalize_text(soup)
    title = " ".join(soup.title.get_text(" ", strip=True).split()) if soup.title else ""
    m = re.search(r"文章来源：\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    publish_time = m.group(1) if m else ""
    return {"title": title, "publish_time": publish_time, "text": text}


def parse_omo_recent() -> pd.DataFrame:
    list_url = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475/index.html"
    rows: List[Dict[str, object]] = []
    for item in pbc_list_articles(list_url, "公开市场业务交易公告 ["):
        payload = pbc_article_payload(item["url"])
        text = payload["text"]
        compact = re.sub(r"\s+", "", text)
        date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", compact)
        amount_match = re.search(r"开展了(\d+(?:\.\d+)?)亿元(\d+)天期逆回购操作", compact)
        rate_match = re.search(r"(\d+\.\d+)%", compact)
        if not date_match or not amount_match or not rate_match:
            continue
        y, m, d = map(int, date_match.groups())
        rows.append(
            {
                "announcement_title": item["title"],
                "article_url": item["url"],
                "publish_time": payload["publish_time"],
                "operation_date": datetime(y, m, d).date().isoformat(),
                "operation_type": "逆回购",
                "tenor_days": int(amount_match.group(2)),
                "operation_amount_100m": float(amount_match.group(1)),
                "operation_rate_pct": float(rate_match.group(1)),
                "net_injection_100m": None,
                "source_text": text,
            }
        )
    return pd.DataFrame(rows).sort_values("operation_date").reset_index(drop=True)


def parse_mlf_recent() -> pd.DataFrame:
    list_url = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125437/125446/125873/index.html"
    rows: List[Dict[str, object]] = []
    for item in pbc_list_articles(list_url, "中期借贷便利"):
        if "招标公告" not in item["title"] and "开展情况" not in item["title"]:
            continue
        payload = pbc_article_payload(item["url"])
        text = payload["text"]
        compact = re.sub(r"\s+", "", text)
        date_match = re.search(r"(\d{4})年(\d{1,2})月", item["title"])
        amount_match = re.search(r"(?:开展|操作共)(\d+(?:\.\d+)?)亿元(?:中期借贷便利|MLF)?", compact)
        rate_match = re.search(r"(?:中标利率|利率)(?:为)?(\d+\.\d+)%", compact)
        tenor_match = re.search(r"期限(?:为)?(\d+)年期|期限(?:为)?(\d+)年|期限(?:为)?(\d+)个月", compact)
        if not date_match:
            continue
        year, month = map(int, date_match.groups())
        tenor_months = None
        if tenor_match:
            if tenor_match.group(1):
                tenor_months = int(tenor_match.group(1)) * 12
            elif tenor_match.group(2):
                tenor_months = int(tenor_match.group(2)) * 12
            elif tenor_match.group(3):
                tenor_months = int(tenor_match.group(3))
        rows.append(
            {
                "announcement_title": item["title"],
                "article_url": item["url"],
                "publish_time": payload["publish_time"],
                "operation_month": f"{year:04d}-{month:02d}",
                "operation_amount_100m": float(amount_match.group(1)) if amount_match else None,
                "tenor_months": tenor_months,
                "operation_rate_pct": float(rate_match.group(1)) if rate_match else None,
                "source_text": text,
            }
        )
    return pd.DataFrame(rows).sort_values("operation_month").reset_index(drop=True)


def fetch_ncd_aaa_1y(start_date: str = "20251201", end_date: str = "20260331") -> pd.DataFrame:
    name_code_df = bond_china_close_return_map()
    symbol_code = name_code_df[name_code_df["cnLabel"] == "同业存单(AAA)"]["value"].values[0]
    url = "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-currency/ClsYldCurvHis"
    session = __bond_register_service()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    frames: List[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=27), end)
        page = 1
        while True:
            params = {
                "lang": "CN",
                "reference": "1,2,3",
                "bondType": symbol_code,
                "startDate": cursor.strftime("%Y-%m-%d"),
                "endDate": chunk_end.strftime("%Y-%m-%d"),
                "termId": "1",
                "pageNum": str(page),
                "pageSize": "50",
            }
            try:
                resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
                data_json = resp.json()
            except Exception:
                break
            data = data_json.get("records", [])
            if not data:
                break
            df = pd.DataFrame(data)
            df = df.rename(
                columns={
                    "newDateValueCN": "日期",
                    "yearTermStr": "期限",
                    "maturityYieldStr": "到期收益率",
                    "currentYieldStr": "即期收益率",
                    "futureYieldStr": "远期收益率",
                }
            )
            keep_cols = ["日期", "期限", "到期收益率", "即期收益率", "远期收益率"]
            df = df[keep_cols]
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.date
            for col in ["期限", "到期收益率", "即期收益率", "远期收益率"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            frames.append(df)
            total_pages = int(data_json.get("data", {}).get("pageTotal", page))
            if page >= total_pages:
                break
            page += 1
        cursor = chunk_end + pd.Timedelta(days=1)
    all_df = pd.concat(frames, ignore_index=True).drop_duplicates()
    out = all_df.loc[all_df["期限"].round(3) == 1.000].copy()
    out = out.rename(
        columns={
            "日期": "date",
            "期限": "tenor_years",
            "到期收益率": "ytm_pct",
            "即期收益率": "spot_pct",
            "远期收益率": "forward_pct",
        }
    )
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    return out.sort_values("date").reset_index(drop=True)


def dr007_probe() -> pd.DataFrame:
    """
    当前只做公开源探测，不产出数据。
    """
    raise RuntimeError(
        "未找到稳定的可公开程序化 DR007 历史接口。"
        "已验证中国货币网公开回购定盘页只提供 FR/FDR；"
        "可见 DR007 页面需要额外隐藏接口或商业数据服务。"
    )


def bill_rediscount_probe() -> pd.DataFrame:
    """
    当前只做公开源探测，不产出数据。
    """
    raise RuntimeError(
        "上海票据交易所公开页面当前被站点防护拦截，"
        "未在无需登录的情况下拿到稳定历史接口。"
    )


def standardize_omo(df: pd.DataFrame) -> pd.DataFrame:
    sdf = df.copy()
    sdf["observation_time"] = pd.to_datetime(sdf["operation_date"])
    sdf["dataset_name"] = "pboc_omo"
    sdf["source"] = "pbc_official"
    sdf["freq"] = "day"
    sdf["indicator_code"] = "7d_reverse_repo_rate"
    sdf["value"] = sdf["operation_rate_pct"]
    return sdf[
        [
            "dataset_name",
            "source",
            "indicator_code",
            "observation_time",
            "freq",
            "value",
            "operation_amount_100m",
            "tenor_days",
            "net_injection_100m",
            "publish_time",
            "article_url",
            "announcement_title",
            "source_text",
        ]
    ]


def standardize_mlf(df: pd.DataFrame) -> pd.DataFrame:
    sdf = df.copy()
    sdf["observation_time"] = pd.to_datetime(sdf["operation_month"] + "-01")
    sdf["dataset_name"] = "pbc_mlf"
    sdf["source"] = "pbc_official"
    sdf["freq"] = "month"
    sdf["indicator_code"] = "mlf_rate"
    sdf["value"] = sdf["operation_rate_pct"]
    return sdf[
        [
            "dataset_name",
            "source",
            "indicator_code",
            "observation_time",
            "freq",
            "value",
            "operation_amount_100m",
            "tenor_months",
            "publish_time",
            "article_url",
            "announcement_title",
            "source_text",
        ]
    ]


def standardize_ncd(df: pd.DataFrame) -> pd.DataFrame:
    sdf = df.copy()
    sdf["observation_time"] = pd.to_datetime(sdf["date"])
    sdf["dataset_name"] = "ncd_aaa_1y"
    sdf["source"] = "chinamoney_curve"
    sdf["freq"] = "day"
    sdf["indicator_code"] = "ncd_aaa_1y_ytm"
    sdf["value"] = sdf["ytm_pct"]
    return sdf[
        [
            "dataset_name",
            "source",
            "indicator_code",
            "observation_time",
            "freq",
            "value",
            "tenor_years",
            "spot_pct",
            "forward_pct",
        ]
    ]


def persist_dataset(
    dataset_name: str,
    raw_df: pd.DataFrame,
    standard_df: pd.DataFrame,
    source_page: str,
    note: str,
    freq: str = "day",
) -> DownloadRecord:
    raw_path = RAW_ROOT / dataset_name / f"{dataset_name}.parquet"
    standard_path = STANDARD_ROOT / f"dataset={dataset_name}" / f"freq={freq}" / f"{dataset_name}.parquet"
    save_parquet(raw_df, raw_path)
    save_parquet(standard_df, standard_path)
    return DownloadRecord(
        dataset_name=dataset_name,
        status="downloaded",
        rows=len(standard_df),
        raw_path=str(raw_path.relative_to(ROOT)),
        standard_path=str(standard_path.relative_to(ROOT)),
        source_page=source_page,
        note=note,
    )


def main() -> None:
    records: List[DownloadRecord] = []
    failures: List[Dict[str, str]] = []

    omo_raw = parse_omo_recent()
    omo_std = standardize_omo(omo_raw)
    records.append(
        persist_dataset(
            "pboc_omo",
            omo_raw,
            omo_std,
            "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475/index.html",
            "央行官网公告可稳定抓到逆回购利率与操作量；净投放字段当前官方页未直接提供，保留为空。",
            freq="day",
        )
    )

    mlf_raw = parse_mlf_recent()
    mlf_std = standardize_mlf(mlf_raw)
    records.append(
        persist_dataset(
            "pbc_mlf",
            mlf_raw,
            mlf_std,
            "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125437/125446/125873/index.html",
            "央行官网可稳定抓到月度 MLF 招标公告及中标利率。",
            freq="month",
        )
    )

    ncd_raw = fetch_ncd_aaa_1y()
    ncd_std = standardize_ncd(ncd_raw)
    records.append(
        persist_dataset(
            "ncd_aaa_1y",
            ncd_raw,
            ncd_std,
            "https://www.chinamoney.com.cn/chinese/bkcurvclosedyhis/?bondType=CYCC41B&reference=1",
            "通过中国货币网同业存单(AAA)收益率曲线提取 1Y 节点。",
            freq="day",
        )
    )

    for dataset_name, func, source_page, note in [
        (
            "dr007",
            dr007_probe,
            "https://en.macromicro.me/series/5899/cn-dr007",
            "已完成公开源探测，但当前未拿到稳定可程序化的历史接口。",
        ),
        (
            "bill_rediscount_rate",
            bill_rediscount_probe,
            "https://www.shcpe.com.cn/content/shcpe/market/ycuver.html",
            "上海票据交易所公开页存在站点防护，当前环境下无法稳定抓取。",
        ),
    ]:
        try:
            func()
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "dataset_name": dataset_name,
                    "status": "blocked",
                    "source_page": source_page,
                    "note": note,
                    "error": str(exc),
                }
            )

    report = {
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "downloaded": [r.__dict__ for r in records],
        "blocked": failures,
    }
    report_path = DOC_ROOT / "policy_liquidity_download_status.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
