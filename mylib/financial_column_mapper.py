"""
Tushare 财务字段中文映射工具

参考接口:
- 利润表(income): https://tushare.pro/document/2?doc_id=33
- 资产负债表(balancesheet): https://tushare.pro/document/2?doc_id=36
- 现金流量表(cashflow): https://tushare.pro/document/2?doc_id=44

说明:
- 本模块优先覆盖项目中实际下载保存的字段（见 financial_downloader.py）。
- 对未收录的字段，默认保留原列名不变。
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple
import pandas as pd


COMMON_FINANCIAL_FIELD_MAP: Dict[str, str] = {
    "ts_code": "TS代码",
    "ann_date": "公告日期",
    "f_ann_date": "实际公告日期",
    "end_date": "报告期",
    "report_type": "报告类型",
    "comp_type": "公司类型",
    "end_type": "报告期类型",
    "update_flag": "更新标识",
}


INCOME_FIELD_MAP: Dict[str, str] = {
    **COMMON_FINANCIAL_FIELD_MAP,
    "total_revenue": "营业总收入",
    "revenue": "营业收入",
    "int_income": "利息收入",
    "prem_earned": "已赚保费",
    "comm_income": "手续费及佣金收入",
    "n_commis_income": "手续费及佣金净收入",
    "prem_cbl": "赔付支出净额",
    "prem_recd_agd": "保费收入",
    "reinsur_income": "分保费收入",
    "operate_profit": "营业利润",
    "total_profit": "利润总额",
    "income_tax": "所得税费用",
    "n_income": "净利润",
    "n_income_attr_p": "归属于母公司的净利润",
    "minority_gain": "少数股东损益",
    "basic_eps": "基本每股收益",
    "diluted_eps": "稀释每股收益",
    "ebit": "息税前利润",
    "ebitda": "息税折旧摊销前利润",
    "cost_expense": "营业成本及费用合计",
    "bus_tax": "营业税金及附加",
    "sell_exp": "销售费用",
    "admin_exp": "管理费用",
    "fin_exp": "财务费用",
    "asset_impair_loss": "资产减值损失",
    "credit_impair_loss": "信用减值损失",
    "oth_income": "其他收益",
    "invest_income": "投资收益",
    "fair_value_gain": "公允价值变动收益",
    "asset_disp_income": "资产处置收益",
    "cur_liab": "流动负债",
    "nca_deduct": "非流动资产处置净损益",
}


BALANCE_FIELD_MAP: Dict[str, str] = {
    **COMMON_FINANCIAL_FIELD_MAP,
    "total_assets": "资产总计",
    "total_liab": "负债合计",
    "total_hldr_eqy_exc_min_int": "归属于母公司所有者权益合计",
    "total_cur_assets": "流动资产合计",
    "total_nca": "非流动资产合计",
    "cash_reser_cb": "货币资金",
    "trad_asset": "交易性金融资产",
    "notes_receiv": "应收票据",
    "accounts_receiv": "应收账款",
    "inventories": "存货",
    "fix_assets": "固定资产",
    "intan_assets": "无形资产",
    "total_cur_liab": "流动负债合计",
    "total_ncl": "非流动负债合计",
    "st_borr": "短期借款",
    "lt_borr": "长期借款",
    "notes_payable": "应付票据",
    "bonds_payable": "应付债券",
    "preferred_stock": "优先股",
    "perpetual_bond": "永续债",
    "capital_reser": "资本公积",
    "surplus_reser": "盈余公积",
    "special_reser": "专项储备",
    "retained_earnings": "未分配利润",
    "oth_rvnu": "其他权益工具",
    "oth_comprecome": "其他综合收益",
}


CASHFLOW_FIELD_MAP: Dict[str, str] = {
    **COMMON_FINANCIAL_FIELD_MAP,
    "n_cashflow_act": "经营活动产生的现金流量净额",
    "n_cashflow_inv_act": "投资活动产生的现金流量净额",
    "n_cash_flows_fnc_act": "筹资活动产生的现金流量净额",
    "c_fr_sale_sg": "销售商品、提供劳务收到的现金",
    "c_paid_goods_s": "购买商品、接受劳务支付的现金",
    "c_paid_to_for_empl": "支付给职工以及为职工支付的现金",
    "c_recp_borrow": "取得借款收到的现金",
    "proc_issue_bonds": "发行债券收到的现金",
    "dep_draw_deposits": "吸收存款收到的现金",
    "c_pay_empl_sal": "支付给职工以及为职工支付的现金",
    "pay_rdexp": "研发支出",
    "pay_int_exp": "利息支出",
    "pay_tax": "支付的各项税费",
    "rec_hd_handle": "处置交易性金融资产净增加额",
    "stot_cashflow_act": "经营活动现金流量净额",
    "s_cashflow_inv_act": "投资活动现金流量净额",
    "s_cash_flows_fnc_act": "筹资活动现金流量净额",
    "c_oth_pay": "支付其他与经营活动有关的现金",
    "c_recp_tax_rf": "收到的税费返还",
    "c_pay_tax_rf": "支付的税费返还",
    "decr_inventories": "存货的减少",
    "decr_ar": "应收账款及其他应收款减少",
    "incr_ar": "应收账款及其他应收款增加",
    "decr_pay": "应付账款及其他应付款增加",
    "decr_adv_pay": "预收款项减少",
    "incr_adv_pay": "预收款项增加",
    "c_icb": "收回投资收到的现金",
    "c_lend_deposits": "存放央行和同业款项净增加额",
    "c_dbcass_remt": "汇兑收益",
    "c_clct_pledge": "质押贷款净增加额",
    "c_dc_invest": "长期股权投资净增加额",
    "c_dc_fair_value_gain": "以公允价值计量且其变动计入当期损益的金融资产净增加额",
    "c_dc_assets_imp": "非流动资产净增加额",
    "c_dc_right_use_assets": "使用权资产净增加额",
    "c_dc_liab_inc": "租赁负债净增加额",
}


FINANCIAL_FIELD_MAPS: Dict[str, Dict[str, str]] = {
    "income": INCOME_FIELD_MAP,
    "balance": BALANCE_FIELD_MAP,
    "cashflow": CASHFLOW_FIELD_MAP,
}


TABLE_ALIASES: Dict[str, str] = {
    "income": "income",
    "income_vip": "income",
    "balancesheet": "balance",
    "balancesheet_vip": "balance",
    "balance": "balance",
    "cashflow": "cashflow",
    "cashflow_vip": "cashflow",
}


def _normalize_table_name(table: str) -> str:
    key = (table or "").strip().lower()
    if key not in TABLE_ALIASES:
        supported = ", ".join(sorted(TABLE_ALIASES.keys()))
        raise ValueError(f"不支持的财务表类型: {table}. 支持: {supported}")
    return TABLE_ALIASES[key]


def infer_financial_table(columns: Iterable[str]) -> Optional[str]:
    """
    根据列名猜测财务表类型（income/balance/cashflow）。

    返回:
    - 匹配成功: 'income'/'balance'/'cashflow'
    - 无法判断: None
    """
    cols = set(columns)
    scores = {}
    for table, mapping in FINANCIAL_FIELD_MAPS.items():
        field_set = set(mapping.keys()) - set(COMMON_FINANCIAL_FIELD_MAP.keys())
        scores[table] = len(cols.intersection(field_set))

    best_table, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return None
    return best_table


def rename_financial_columns_to_chinese(
    df: pd.DataFrame,
    table: Optional[str] = None,
    *,
    keep_unmapped: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    将财务DataFrame列名从英文字段转换为中文名。

    Args:
        df: 输入DataFrame
        table: 表类型。可选: income/balance/cashflow 及对应 *_vip/aliases。
               若为None，自动根据列名推断。
        keep_unmapped: True 时未映射字段保留原名；False 时仅保留已映射列。

    Returns:
        (转换后的DataFrame, 实际使用的映射字典)
    """
    if table is None:
        inferred = infer_financial_table(df.columns)
        if inferred is None:
            raise ValueError("无法根据列名自动识别财务表类型，请显式传入 table 参数")
        table_key = inferred
    else:
        table_key = _normalize_table_name(table)

    full_map = FINANCIAL_FIELD_MAPS[table_key]
    used_map = {col: full_map[col] for col in df.columns if col in full_map}

    if keep_unmapped:
        converted = df.rename(columns=used_map)
    else:
        mapped_cols = [c for c in df.columns if c in used_map]
        converted = df[mapped_cols].rename(columns=used_map)

    return converted, used_map

