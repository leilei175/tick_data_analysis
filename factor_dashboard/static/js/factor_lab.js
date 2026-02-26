/**
 * 因子分析实验室 - 前端逻辑
 */

// 全局状态
const state = {
    sources: [],
    factors: {},
    stockPools: {},
    preprocessMethods: {},
    returnsTypes: {},
    currentResult: null
};

// 预处理方法参数配置
const preprocessParamsConfig = {
    zscore: [
        { name: 'std_threshold', label: 'Z-Score截断阈值', type: 'number', default: 3, min: 1, max: 5, step: 0.5 }
    ],
    winsorize: [
        { name: 'lower_quantile', label: '下侧分位数', type: 'number', default: 0.025, min: 0, max: 0.1, step: 0.005 },
        { name: 'upper_quantile', label: '上侧分位数', type: 'number', default: 0.975, min: 0.9, max: 1, step: 0.005 }
    ],
    mad: [
        { name: 'threshold', label: 'MAD倍数阈值', type: 'number', default: 5, min: 3, max: 10, step: 1 }
    ],
    neutralize: [
        { name: 'market_cap', label: '市值中性化', type: 'checkbox', default: true },
        { name: 'industry', label: '行业中性化', type: 'checkbox', default: true }
    ]
};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    init();
});

async function init() {
    // 加载因子来源
    await loadFactorFactories();

    // 加载收益率类型
    await loadReturnsTypes();

    // 初始化事件监听
    initEventListeners();

    // 设置默认日期
    setDefaultDates();
}

function initEventListeners() {
    // 收益率类型变化时显示/隐藏n参数
    const returnsTypeSelect = document.getElementById('returnsType');
    if (returnsTypeSelect) {
        returnsTypeSelect.addEventListener('change', function() {
            const nGroup = document.getElementById('returnsNGroup');
            if (this.value === 'close2close_n') {
                nGroup.style.display = 'block';
            } else {
                nGroup.style.display = 'none';
            }
        });
    }

    // 因子来源变化时重新加载因子列表
    const factorSourceSelect = document.getElementById('factorSource');
    if (factorSourceSelect) {
        factorSourceSelect.addEventListener('change', async function() {
            const source = this.value;
            const factorSelect = document.getElementById('factorName');
            factorSelect.innerHTML = '<option value="">加载中...</option>';
            await loadFactors(source);
        });
    }
}

function setDefaultDates() {
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const day = String(today.getDate()).padStart(2, '0');

    // 设置起始日期为当月1日
    document.getElementById('startDate').value = `${year}-${month}-01`;
    // 设置结束日期为今天
    document.getElementById('endDate').value = `${year}-${month}-${day}`;
}

// 加载因子工厂信息
async function loadFactorFactories() {
    try {
        const response = await fetch('/api/factor/factories', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        const result = await response.json();

        if (result.status === 'success') {
            state.sources = result.data.sources || [];
            state.factors = result.data.factors || {};
            state.stockPools = result.data.stock_pools || {};
            populateStockPools();

            // 加载第一个来源的因子
            if (state.sources.length > 0) {
                await loadFactors(state.sources[0]);
            }
        } else {
            showToast(result.message || '加载因子列表失败', 'error');
        }
    } catch (error) {
        console.error('加载因子工厂失败:', error);
        showToast('加载因子列表失败', 'error');
    }
}

// 加载指定来源的因子
async function loadFactors(source) {
    if (!source) {
        source = document.getElementById('factorSource')?.value || 'high_frequency';
    }
    try {
        const response = await fetch('/api/factor/factories', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: source })
        });

        const result = await response.json();

        if (result.status === 'success') {
            // 处理返回的数据格式：可能是数组或对象
            let factors = [];
            if (Array.isArray(result.data.factors)) {
                factors = result.data.factors;
            } else if (typeof result.data.factors === 'object' && result.data.factors !== null) {
                factors = result.data.factors[source] || [];
            }

            state.factors[source] = factors;

            // 更新因子下拉框
            const factorSelect = document.getElementById('factorName');
            factorSelect.innerHTML = '';

            factors.forEach(factor => {
                const option = document.createElement('option');
                option.value = factor;
                option.textContent = formatFactorName(factor);
                factorSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error('加载因子列表失败:', error);
    }
}

function populateStockPools() {
    const stockPoolSelect = document.getElementById('stockPool');
    if (!stockPoolSelect) return;

    const previous = stockPoolSelect.value;
    stockPoolSelect.innerHTML = '';

    for (const [key, descList] of Object.entries(state.stockPools || {})) {
        const label = Array.isArray(descList) && descList.length > 0 ? descList[0] : key;
        const option = document.createElement('option');
        option.value = key;
        option.textContent = `${label} (${key.toUpperCase()})`;
        stockPoolSelect.appendChild(option);
    }

    if (previous && state.stockPools && state.stockPools[previous]) {
        stockPoolSelect.value = previous;
    }
}

// 加载收益率类型
async function loadReturnsTypes() {
    try {
        const response = await fetch('/api/returns/types');
        const result = await response.json();

        if (result.status === 'success') {
            state.returnsTypes = result.data.methods || {};

            // 更新收益率类型下拉框
            const returnsSelect = document.getElementById('returnsType');
            returnsSelect.innerHTML = '';

            const methods = result.data.methods || {};
            for (const [key, config] of Object.entries(methods)) {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = `${config.name} - ${config.description}`;
                returnsSelect.appendChild(option);
            }
        }
    } catch (error) {
        console.error('加载收益率类型失败:', error);
    }
}

// 更新预处理参数
function updatePreprocessParams() {
    const method = document.getElementById('preprocessMethod').value;
    const paramsContainer = document.getElementById('preprocessParams');

    if (!method || !preprocessParamsConfig[method]) {
        paramsContainer.innerHTML = '';
        paramsContainer.classList.remove('show');
        return;
    }

    const params = preprocessParamsConfig[method];
    let html = '';

    params.forEach(param => {
        const defaultValue = param.default;

        if (param.type === 'number') {
            html += `
                <div class="form-group">
                    <label class="form-label">${param.label}</label>
                    <input type="number" id="param_${param.name}"
                           class="form-input" value="${defaultValue}"
                           min="${param.min}" max="${param.max}" step="${param.step || 1}">
                </div>
            `;
        } else if (param.type === 'checkbox') {
            html += `
                <div class="form-group">
                    <label class="checkbox-item">
                        <input type="checkbox" id="param_${param.name}" ${defaultValue ? 'checked' : ''}>
                        <span>${param.label}</span>
                    </label>
                </div>
            `;
        }
    });

    paramsContainer.innerHTML = html;
    paramsContainer.classList.add('show');
}

// 获取预处理参数
function getPreprocessParams() {
    const method = document.getElementById('preprocessMethod').value;
    if (!method || !preprocessParamsConfig[method]) {
        return {};
    }

    const params = {};
    const paramConfigs = preprocessParamsConfig[method];

    paramConfigs.forEach(config => {
        const element = document.getElementById(`param_${config.name}`);
        if (element) {
            if (config.type === 'number') {
                params[config.name] = parseFloat(element.value);
            } else if (config.type === 'checkbox') {
                params[config.name] = element.checked;
            }
        }
    });

    return params;
}

// 切换过滤参数显示
function toggleFilterParams() {
    const content = document.getElementById('filterParams');
    content.classList.toggle('show');
}

// 获取过滤条件
function getFilters() {
    return {
        limit_up: document.getElementById('filterLimitUp')?.checked || false,
        min_amount: parseFloat(document.getElementById('minAmount')?.value) || 1000,
        min_turnover: parseFloat(document.getElementById('minTurnover')?.value) || 0.5
    };
}

// 获取分析配置
function getAnalysisConfig() {
    return {
        factor: document.getElementById('factorName').value,
        source: document.getElementById('factorSource').value,
        stock_pool: document.getElementById('stockPool').value,
        start_date: formatDate(document.getElementById('startDate').value),
        end_date: formatDate(document.getElementById('endDate').value),
        preprocess: {
            method: document.getElementById('preprocessMethod').value || null,
            params: getPreprocessParams()
        },
        config: {
            returns_type: document.getElementById('returnsType').value,
            returns_n: parseInt(document.getElementById('returnsN')?.value) || 5,
            quantiles: parseInt(document.getElementById('quantiles').value) || 5,
            filters: getFilters()
        }
    };
}

// 预览预处理效果
async function previewPreprocess() {
    const config = getAnalysisConfig();

    if (!config.factor) {
        showToast('请选择因子', 'warning');
        return;
    }

    showLoading('正在预览预处理效果...');

    try {
        const response = await fetch('/api/factor/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                factor: config.factor,
                source: config.source,
                method: config.preprocess.method || 'zscore',
                params: config.preprocess.params,
                start: config.start_date,
                end: config.end_date
            })
        });

        const result = await response.json();
        hideLoading();

        if (result.status === 'success') {
            const data = result.data;
            const before = data.before_stats;
            const after = data.after_stats;

            showToast(`
                预处理效果预览:
                均值: ${before.mean.toFixed(4)} → ${after.mean.toFixed(4)}
                标准差: ${before.std.toFixed(4)} → ${after.std.toFixed(4)}
            `, 'success');
        } else {
            showToast(result.message || '预览失败', 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('预览失败:', error);
        showToast('预览失败', 'error');
    }
}

// 执行分析
async function runAnalysis() {
    const config = getAnalysisConfig();

    if (!config.factor) {
        showToast('请选择因子', 'warning');
        return;
    }

    // 移除空预处理方法
    if (!config.preprocess.method) {
        config.preprocess = null;
    }

    showLoading('正在执行因子分析...');

    try {
        const response = await fetch('/api/factor/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const result = await response.json();
        hideLoading();

        if (result.status === 'success') {
            state.currentResult = result.data;
            displayResults(result.data);
            updateStepIndicator(4);
            showToast('分析完成!', 'success');
        } else {
            showToast(result.message || '分析失败', 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('分析失败:', error);
        showToast('分析失败: ' + error.message, 'error');
    }
}

// 显示分析结果
function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.add('show');

    // IC统计
    displayICStats(data.ic_stats);

    // 多空组合
    displayLongShortStats(data.long_short_stats);

    // 分层收益
    displayQuantileReturns(data.quantile_returns);

    // 绘制图表
    if (data.charts) {
        drawICChart(data.charts.ic_series);
        drawQuantileChart(data.charts.quantile_returns);
        drawICCumsumChart(data.charts.ic_series);
        drawQuantileNavChart(data.charts.quantile_daily);
    }
}

// 显示IC统计
function displayICStats(icStats) {
    const container = document.getElementById('icStats');

    if (!icStats) {
        container.innerHTML = '<p>暂无数据</p>';
        return;
    }

    const icMean = icStats.ic_mean || 0;
    const icIR = icStats.ic_ir || 0;
    const positiveRatio = (icStats.ic_positive_ratio || 0) * 100;
    const icCount = icStats.ic_count || 0;

    container.innerHTML = `
        <div class="stat-card">
            <div class="stat-value ${icMean >= 0 ? 'positive' : 'negative'}">${icMean.toFixed(4)}</div>
            <div class="stat-label">IC均值</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${icIR.toFixed(4)}</div>
            <div class="stat-label">IC信息比率</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${positiveRatio.toFixed(1)}%</div>
            <div class="stat-label">IC正向比例</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${icCount}</div>
            <div class="stat-label">样本数</div>
        </div>
    `;
}

// 显示多空组合统计
function displayLongShortStats(lsStats) {
    const container = document.getElementById('longShortStats');

    if (!lsStats) {
        container.innerHTML = '<p>暂无数据</p>';
        return;
    }

    const totalReturn = (lsStats.total_return || 0) * 100;
    const sharpe = lsStats.sharpe || 0;
    const winRate = (lsStats.win_rate || 0) * 100;
    const maxDrawdown = (lsStats.max_drawdown || 0) * 100;

    container.innerHTML = `
        <div class="stat-card">
            <div class="stat-value ${totalReturn >= 0 ? 'positive' : 'negative'}">${totalReturn.toFixed(2)}%</div>
            <div class="stat-label">总收益</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${sharpe.toFixed(4)}</div>
            <div class="stat-label">夏普比率</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${winRate.toFixed(1)}%</div>
            <div class="stat-label">胜率</div>
        </div>
        <div class="stat-card">
            <div class="stat-value negative">${maxDrawdown.toFixed(2)}%</div>
            <div class="stat-label">最大回撤</div>
        </div>
    `;
}

// 显示分层收益
function displayQuantileReturns(quantileReturns) {
    const tbody = document.getElementById('quantileTableBody');

    if (!quantileReturns || quantileReturns.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6">暂无数据</td></tr>';
        return;
    }

    let html = '';
    quantileReturns.forEach(q => {
        const meanReturn = (q.mean || 0) * 100;
        const stdReturn = (q.std || 0) * 100;

        html += `
            <tr>
                <td>Q${q.quantile}</td>
                <td>${q.factor_mean?.toFixed(4) || '-'}</td>
                <td class="${meanReturn >= 0 ? 'positive' : 'negative'}">${meanReturn.toFixed(4)}%</td>
                <td>${stdReturn.toFixed(4)}%</td>
                <td>${q.sharpe?.toFixed(4) || '-'}</td>
                <td>${q.count || '-'}</td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
}

// 绘制IC图表
function drawICChart(icSeries) {
    if (!icSeries || icSeries.length === 0) return;

    const dates = icSeries.map(d => d.date);
    const values = icSeries.map(d => d.ic);

    const trace = {
        x: dates,
        y: values,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#3498db', width: 1 },
        marker: { size: 4 }
    };

    const layout = {
        margin: { t: 20, r: 20, b: 40, l: 50 },
        xaxis: { title: '日期' },
        yaxis: { title: 'IC' },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Noto Sans SC, sans-serif' }
    };

    Plotly.newPlot('icChart', [trace], layout, { responsive: true });
}

// 绘制分层收益图表
function drawQuantileChart(quantileReturns) {
    if (!quantileReturns || quantileReturns.length === 0) return;

    const quantiles = quantileReturns.map(q => `Q${q.quantile}`);
    const returns = quantileReturns.map(q => (q.mean || 0) * 100);

    const trace = {
        x: quantiles,
        y: returns,
        type: 'bar',
        marker: {
            color: returns.map(r => r >= 0 ? '#27ae60' : '#e74c3c')
        }
    };

    const layout = {
        margin: { t: 20, r: 20, b: 40, l: 50 },
        xaxis: { title: '分层' },
        yaxis: { title: '平均收益 (%)' },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Noto Sans SC, sans-serif' }
    };

    Plotly.newPlot('quantileChart', [trace], layout, { responsive: true });
}

// 绘制IC累积和图表
function drawICCumsumChart(icSeries) {
    if (!icSeries || icSeries.length === 0) {
        Plotly.purge('icCumsumChart');
        return;
    }

    const dates = icSeries.map(d => d.date);
    const cumsum = [];
    let running = 0;
    icSeries.forEach(point => {
        running += Number(point.ic || 0);
        cumsum.push(running);
    });

    const trace = {
        x: dates,
        y: cumsum,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#0f5c9a', width: 2 }
    };

    const layout = {
        margin: { t: 20, r: 20, b: 40, l: 50 },
        xaxis: { title: '日期' },
        yaxis: { title: 'IC累积和' },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Noto Sans SC, sans-serif' }
    };

    Plotly.newPlot('icCumsumChart', [trace], layout, { responsive: true });
}

// 绘制分组累积净值图表
function drawQuantileNavChart(quantileDaily) {
    if (!quantileDaily || quantileDaily.length === 0) {
        Plotly.purge('quantileNavChart');
        return;
    }

    const dates = quantileDaily.map(row => row.date);
    const quantileKeys = Object.keys(quantileDaily[0]).filter(k => /^Q\d+$/.test(k));
    if (!quantileKeys.length) {
        Plotly.purge('quantileNavChart');
        return;
    }

    const colors = ['#0b3a67', '#1f7a8c', '#4f9d69', '#d4a017', '#b64545', '#7b2cbf', '#3a86ff', '#fb5607', '#4361ee', '#2a9d8f'];
    const traces = quantileKeys.map((q, idx) => {
        let nav = 1.0;
        const navSeries = quantileDaily.map(row => {
            const ret = Number(row[q] || 0);
            nav *= (1 + ret);
            return nav;
        });
        return {
            x: dates,
            y: navSeries,
            type: 'scatter',
            mode: 'lines',
            name: q,
            line: { width: 2, color: colors[idx % colors.length] }
        };
    });

    const layout = {
        margin: { t: 20, r: 20, b: 40, l: 60 },
        xaxis: { title: '日期' },
        yaxis: { title: '累积净值' },
        legend: { orientation: 'h', y: -0.2 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Noto Sans SC, sans-serif' }
    };

    Plotly.newPlot('quantileNavChart', traces, layout, { responsive: true });
}

// 保存结果
async function saveResult() {
    if (!state.currentResult) {
        showToast('请先执行分析', 'warning');
        return;
    }

    const config = getAnalysisConfig();

    showLoading('正在保存结果...');

    try {
        const response = await fetch('/api/factor/save-result', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const result = await response.json();
        hideLoading();

        if (result.status === 'success') {
            showToast(`结果已保存至: ${result.data.save_path}`, 'success');
        } else {
            showToast(result.message || '保存失败', 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('保存失败:', error);
        showToast('保存失败', 'error');
    }
}

// 下载报告
function downloadReport() {
    if (!state.currentResult) {
        showToast('请先执行分析', 'warning');
        return;
    }

    showToast('报告生成中...', 'success');

    // 触发页面刷新以下载新报告
    setTimeout(() => {
        window.location.reload();
    }, 1000);
}

// 更新步骤指示器
function updateStepIndicator(step) {
    const steps = document.querySelectorAll('.step-item');
    steps.forEach((el, index) => {
        const stepNum = index + 1;
        el.classList.remove('active', 'completed');

        if (stepNum < step) {
            el.classList.add('completed');
        } else if (stepNum === step) {
            el.classList.add('active');
        }
    });
}

// 格式化因子名称
function formatFactorName(name) {
    return name
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// 格式化日期
function formatDate(dateStr) {
    if (!dateStr) return null;
    return dateStr.replace(/-/g, '');
}

// 显示加载状态
function showLoading(text) {
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');

    loadingText.textContent = text;
    overlay.classList.add('show');
}

// 隐藏加载状态
function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('show');
}

// 显示提示信息
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast show ${type}`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 5000);
}
