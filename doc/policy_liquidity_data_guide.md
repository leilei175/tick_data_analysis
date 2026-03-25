# 政策利率与流动性数据下载说明

本文档对应脚本 [`download_policy_liquidity_data.py`](/data1/code_git/tick_data_analysis/download_policy_liquidity_data.py)。

目标数据：

- `DR007`
- `OMO`
- `MLF`
- `1Y AAA NCD`
- `票据转贴现`

当前日期为 `2026-03-18`。本次实现中，`OMO`、`MLF`、`1Y AAA NCD` 已完成自动下载；`DR007`、`票据转贴现` 已完成网络源探测，但当前公开环境下仍被数据源限制。

## 1. 运行方式

```bash
python download_policy_liquidity_data.py
```

运行后会更新：

- 原始层：`daily_data/hf_raw/...`
- 标准层：`daily_data/hf_standard/...`
- 状态文件：[`doc/policy_liquidity_download_status.json`](/data1/code_git/tick_data_analysis/doc/policy_liquidity_download_status.json)

## 2. 数据源与实现情况

### 2.1 OMO

来源：

- 中国人民银行公开市场业务交易公告
- 列表页：<https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475/index.html>

实现方式：

- 抓取公告列表页中的文章链接
- 逐篇下载公告正文
- 从正文里提取：
  - `operation_date`
  - `operation_amount_100m`
  - `operation_rate_pct`
  - `tenor_days`

当前限制：

- 央行日公告页面能稳定给出逆回购操作量和利率
- 但日公告正文没有直接给出 `net_injection`
- 因此脚本中 `net_injection_100m` 目前保留为空

保存地址：

- 原始层：[`daily_data/hf_raw/pboc_omo/pboc_omo.parquet`](/data1/code_git/tick_data_analysis/daily_data/hf_raw/pboc_omo/pboc_omo.parquet)
- 标准层：[`daily_data/hf_standard/dataset=pboc_omo/freq=day/pboc_omo.parquet`](/data1/code_git/tick_data_analysis/daily_data/hf_standard/dataset=pboc_omo/freq=day/pboc_omo.parquet)

读取方式：

```python
import pandas as pd

df = pd.read_parquet("daily_data/hf_standard/dataset=pboc_omo/freq=day/pboc_omo.parquet")
print(df[["observation_time", "value", "operation_amount_100m", "tenor_days"]].tail())
```

### 2.2 MLF

来源：

- 中国人民银行中期借贷便利工作信息
- 列表页：<https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125437/125446/125873/index.html>

实现方式：

- 抓取 `中期借贷便利招标公告` 和 `中期借贷便利开展情况`
- 逐篇下载正文
- 解析：
  - `operation_month`
  - `operation_amount_100m`
  - `operation_rate_pct`
  - `tenor_months`

当前限制：

- `2025` 年后部分公告改为“固定数量、利率招标、多重价位中标”，正文不总是给单一 `中标利率`
- 因此部分月份 `operation_rate_pct` 会为空，这是上游公告口径变化，不是解析失败

保存地址：

- 原始层：[`daily_data/hf_raw/pbc_mlf/pbc_mlf.parquet`](/data1/code_git/tick_data_analysis/daily_data/hf_raw/pbc_mlf/pbc_mlf.parquet)
- 标准层：[`daily_data/hf_standard/dataset=pbc_mlf/freq=month/pbc_mlf.parquet`](/data1/code_git/tick_data_analysis/daily_data/hf_standard/dataset=pbc_mlf/freq=month/pbc_mlf.parquet)

读取方式：

```python
import pandas as pd

df = pd.read_parquet("daily_data/hf_standard/dataset=pbc_mlf/freq=month/pbc_mlf.parquet")
print(df[["observation_time", "value", "operation_amount_100m", "tenor_months"]].tail())
```

### 2.3 1Y AAA NCD

来源：

- 中国货币网收盘收益率曲线历史页
- 页面：<https://www.chinamoney.com.cn/chinese/bkcurvclosedyhis/?bondType=CYCC41B&reference=1>

实现方式：

- 使用中国货币网曲线接口 `ClsYldCurvHis`
- 选择曲线 `同业存单(AAA)`
- 抽取 `期限 == 1.0` 的节点
- 输出字段：
  - `date`
  - `ytm_pct`
  - `spot_pct`
  - `forward_pct`

当前下载范围：

- 默认抓取 `2025-12-01` 到 `2026-03-31`
- 这里做了保守窗口控制，因为中国货币网这个接口在大窗口查询时容易超时或拒绝响应

保存地址：

- 原始层：[`daily_data/hf_raw/ncd_aaa_1y/ncd_aaa_1y.parquet`](/data1/code_git/tick_data_analysis/daily_data/hf_raw/ncd_aaa_1y/ncd_aaa_1y.parquet)
- 标准层：[`daily_data/hf_standard/dataset=ncd_aaa_1y/freq=day/ncd_aaa_1y.parquet`](/data1/code_git/tick_data_analysis/daily_data/hf_standard/dataset=ncd_aaa_1y/freq=day/ncd_aaa_1y.parquet)

读取方式：

```python
import pandas as pd

df = pd.read_parquet("daily_data/hf_standard/dataset=ncd_aaa_1y/freq=day/ncd_aaa_1y.parquet")
print(df[["observation_time", "value", "spot_pct"]].tail())
```

### 2.4 DR007

探测来源：

- 中国货币网回购定盘利率页：<https://www.chinamoney.com.cn/chinese/bkfrr/>
- 公开网页可稳定拿到的是 `FR001/FR007/FR014` 和 `FDR001/FDR007/FDR014`
- 另检索了可公开网络源页面：<https://en.macromicro.me/series/5899/cn-dr007>

当前结论：

- 公开可直接程序化的中国货币网页面只公开了 `FR/FDR`，没有直接公开 `DR007` 历史序列
- 其他公开网页要么被 Cloudflare 保护，要么不暴露稳定历史接口
- 因此当前脚本将 `DR007` 记录为 `blocked`

状态记录：

- 查看 [`doc/policy_liquidity_download_status.json`](/data1/code_git/tick_data_analysis/doc/policy_liquidity_download_status.json) 中 `dr007` 项

### 2.5 票据转贴现

探测来源：

- 上海票据交易所公开页面：<https://www.shcpe.com.cn/content/shcpe/market/ycuver.html>

当前结论：

- 当前环境访问该页面会被站点防护拦截，返回 422/防护页
- 尚未拿到无需登录、可稳定重放的历史接口
- 因此当前脚本将 `bill_rediscount_rate` 记录为 `blocked`

状态记录：

- 查看 [`doc/policy_liquidity_download_status.json`](/data1/code_git/tick_data_analysis/doc/policy_liquidity_download_status.json) 中 `bill_rediscount_rate` 项

## 3. 当前落盘结果

截至 `2026-03-18 12:21:37`，脚本实际落盘了 3 组数据：

- `pboc_omo`
- `pbc_mlf`
- `ncd_aaa_1y`

状态文件中也保留了 2 组受阻数据的来源和错误信息：

- `dr007`
- `bill_rediscount_rate`

## 4. 推荐读取入口

如果后续研究只关心标准层，直接读取下列路径即可：

```python
import pandas as pd

omo = pd.read_parquet("daily_data/hf_standard/dataset=pboc_omo/freq=day/pboc_omo.parquet")
mlf = pd.read_parquet("daily_data/hf_standard/dataset=pbc_mlf/freq=month/pbc_mlf.parquet")
ncd = pd.read_parquet("daily_data/hf_standard/dataset=ncd_aaa_1y/freq=day/ncd_aaa_1y.parquet")
```

## 5. 后续建议

如果要把 `DR007` 和 `票据转贴现` 也完全自动化，下一步应优先做两件事：

1. 为 `DR007` 接入商业终端或受控页面的数据接口，而不是继续依赖公开网页
2. 为票据转贴现改用上海票据交易所授权接口、Wind、iFinD 或其他机构级数据源
