import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from pypfopt import expected_returns, risk_models
from pypfopt.cla import CLA
from pypfopt.hierarchical_portfolio import HRPOpt
# ————————————————————————————————————————————————
# 1) 参数设置（模拟真实策略多样性）
# ————————————————————————————————————————————————
np.random.seed(42)  # 可复现
start_date = "2010-01-01"
end_date   = "2025-12-31"
freq       = "D"  # 日频
N          = 10

# 策略参数（每策略独立）：年化收益、波动率、自相关系数、胜率
strategy_params = [
    {"mu_ann": 0.08, "sigma_ann": 0.12, "rho": 0.1, "win_rate": 0.55},  # 稳健趋势
    {"mu_ann": 0.12, "sigma_ann": 0.20, "rho": 0.3, "win_rate": 0.48},  # 高波动能
    {"mu_ann": 0.05, "sigma_ann": 0.08, "rho": -0.05, "win_rate": 0.62}, # 低波逆向
    {"mu_ann": 0.15, "sigma_ann": 0.25, "rho": 0.4, "win_rate": 0.42},  # 高风险套利
    {"mu_ann": 0.03, "sigma_ann": 0.06, "rho": 0.0, "win_rate": 0.58},   # 现金增强
    {"mu_ann": 0.10, "sigma_ann": 0.18, "rho": 0.25, "win_rate": 0.50}, # 多因子
    {"mu_ann": 0.06, "sigma_ann": 0.10, "rho": 0.15, "win_rate": 0.57}, # 行业轮动
    {"mu_ann": 0.09, "sigma_ann": 0.15, "rho": 0.2, "win_rate": 0.53},  # 波动率套利
    {"mu_ann": 0.04, "sigma_ann": 0.07, "rho": -0.1, "win_rate": 0.65},  # 统计套利
    {"mu_ann": 0.11, "sigma_ann": 0.22, "rho": 0.35, "win_rate": 0.46}, # 高频降频
]

# ————————————————————————————————————————————————
# 2) 生成日频 PnL 序列（T × N）
# ————————————————————————————————————————————————
dates_daily = pd.date_range(start=start_date, end=end_date, freq=freq)
T = len(dates_daily)

# 初始化 PnL 矩阵（全零，后续累加）
pnl_daily = np.zeros((T, N))

for i in range(N):
    params = strategy_params[i]
    
    # 日度参数（年化 → 日度）
    mu_day = params["mu_ann"] / 252
    sigma_day = params["sigma_ann"] / np.sqrt(252)
    
    # 生成带自相关的日度收益（AR(1)）
    eps = np.random.normal(0, sigma_day, T)
    ret = np.empty(T)
    ret[0] = mu_day + eps[0]
    for t in range(1, T):
        ret[t] = mu_day + params["rho"] * (ret[t-1] - mu_day) + np.sqrt(1 - params["rho"]**2) * eps[t]
    
    # 调整胜率（可选：使正收益比例接近 win_rate）
    if params["win_rate"] != 0.5:
        # 简单偏移：调整均值使正收益占比达标（不影响波动率）
        target_pos_frac = params["win_rate"]
        current_pos_frac = np.mean(ret > 0)
        if abs(current_pos_frac - target_pos_frac) > 0.05:
            shift = np.percentile(ret, 50 + 100*(target_pos_frac - 0.5))
            ret = ret - np.mean(ret) + shift
    
    # 累计 PnL（从0开始）
    pnl_daily[:, i] = np.cumsum(ret)

# 构建日频 DataFrame
pnl_daily_df = pd.DataFrame(
    pnl_daily,
    index=dates_daily,
    columns=[f"Strategy_{i+1}" for i in range(N)]
)

pnl_weekly_df = pnl_daily_df.resample("W-FRI").last()

# 清理：删除全 NaN 的首尾行（如起始日非周五）
pnl_weekly_df = pnl_weekly_df.dropna(how="all")

returns_simple = pnl_weekly_df.diff()
returns_simple.iloc[0] = pnl_weekly_df.iloc[0]  # 第一天的收益率就是第一天的累计收益
returns=returns_simple
print("✅ 日频 PnL 生成完成：")
print(f"  • 形状: {pnl_daily_df.shape} ({T} 天 × {N} 策略)")
print(f"  • 时间范围: {pnl_daily_df.index[0]} 至 {pnl_daily_df.index[-1]}")
print(f"  • 示例（前3行）:\n{pnl_daily_df.head(3)}\n")

def calculate_portfolio_performance(weights, expected_returns_vec, cov_matrix, risk_free_rate=0.02):
    """
    计算投资组合绩效指标
    """
    # 确保权重是NumPy数组
    if isinstance(weights, dict):
        weights = np.array(list(weights.values()))
    elif isinstance(weights, list):
        weights = np.array(weights)
    
    # 确保预期收益是NumPy数组
    if isinstance(expected_returns_vec, pd.Series):
        expected_returns_vec = expected_returns_vec.values
    
    # 计算组合收益和风险
    portfolio_return = np.dot(weights, expected_returns_vec)
    
    # 计算组合方差
    portfolio_variance = weights @ cov_matrix @ weights
    portfolio_risk = np.sqrt(portfolio_variance)
    
    if portfolio_risk > 1e-10:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    else:
        sharpe_ratio = 0
    
    # 计算风险贡献度
    if portfolio_risk > 1e-10:
        marginal_risk = cov_matrix @ weights / portfolio_risk
        risk_contribution = weights * marginal_risk
        risk_contribution_pct = risk_contribution / portfolio_risk
    else:
        risk_contribution_pct = np.zeros_like(weights)
    
    return {
        'return': portfolio_return,
        'risk': portfolio_risk,
        'sharpe': sharpe_ratio,
        'weights': weights,
        'risk_contribution': risk_contribution_pct
    }

def run_cla_optimization_from_returns(returns, weight_bounds=(0, 0.1)):
    """
    使用CLA算法进行投资组合优化 - 直接从收益率数据
    """
    # 计算预期收益和协方差
    mu = expected_returns.mean_historical_return(returns,returns_data=True)
    sigma = risk_models.sample_cov(returns,returns_data=True)
    
    # 确保sigma是NumPy数组
    sigma_array = sigma.values if isinstance(sigma, pd.DataFrame) else sigma
    
    try:
        # 初始化CLA对象
        cla = CLA(mu, sigma_array, weight_bounds=weight_bounds)
        
        # 计算最小方差组合
        min_vol_weights = cla.min_volatility()
        
        # 计算最大夏普组合
        max_sharpe_weights = cla.max_sharpe()
        
        # 获取有效前沿
        frontier_points = 100
        mu_list, sigma_list, _ = cla.efficient_frontier(points=frontier_points)
        
        # 计算绩效
        min_vol_perf = calculate_portfolio_performance(min_vol_weights, mu.values, sigma_array)
        max_sharpe_perf = calculate_portfolio_performance(max_sharpe_weights, mu.values, sigma_array)
        
        return {
            'model': cla,
            'min_vol': {'weights': min_vol_weights, 'performance': min_vol_perf},
            'max_sharpe': {'weights': max_sharpe_weights, 'performance': max_sharpe_perf},
            'frontier': {'returns': mu_list, 'risks': sigma_list},
            'mu': mu,
            'sigma': sigma_array
        }
    except Exception as e:
        print(f"CLA优化失败: {e}")
        # 返回一个简化结果用于继续执行
        return {
            'model': None,
            'min_vol': {'weights': {}, 'performance': {'return': 0, 'risk': 0, 'sharpe': 0, 'weights': [], 'risk_contribution': []}},
            'max_sharpe': {'weights': {}, 'performance': {'return': 0, 'risk': 0, 'sharpe': 0, 'weights': [], 'risk_contribution': []}},
            'frontier': {'returns': [], 'risks': []},
            'mu': mu,
            'sigma': sigma_array
        }

def run_hrp_optimization_from_returns(returns):
    """
    使用HRP算法进行投资组合优化 - 直接从收益率数据
    """
    try:
        # 确保收益率数据没有NaN值
        returns = returns.dropna() if returns.isnull().any().any() else returns
        
        # 初始化HRP优化器
        hrp = HRPOpt(returns)
        
        # 运行优化
        hrp_weights = hrp.optimize()
        
        # 获取聚类信息
        corr_matrix = returns.corr()
        # 将相关矩阵转换为距离矩阵
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)
        link_matrix = linkage(dist_matrix, method='single')
        
        # 获取排序索引
        ordered_indices = list(range(len(returns.columns)))
        
        # 计算预期收益和协方差
        mu = expected_returns.mean_historical_return(returns,returns_data=True)
        sigma = risk_models.sample_cov(returns,returns_data=True)
        sigma_array = sigma.values if isinstance(sigma, pd.DataFrame) else sigma
        
        # 计算绩效
        hrp_perf = calculate_portfolio_performance(hrp_weights, mu.values, sigma_array)
        
        return {
            'model': hrp,
            'weights': hrp_weights,
            'performance': hrp_perf,
            'linkage_matrix': link_matrix,
            'ordered_indices': ordered_indices,
            'asset_names': list(returns.columns)
        }
    except Exception as e:
        print(f"HRP优化失败: {e}")
        # 返回一个简化结果
        return {
            'model': None,
            'weights': {},
            'performance': {'return': 0, 'risk': 0, 'sharpe': 0, 'weights': [], 'risk_contribution': []},
            'linkage_matrix': np.array([]),
            'ordered_indices': [],
            'asset_names': list(returns.columns) if hasattr(returns, 'columns') else []
        }
    

hrpv2=run_hrp_optimization_from_returns(returns)



    #%%

import scipy.cluster.hierarchy as sch
pnl_weekly_df = pnl_daily_df.resample("W-FRI").last()
pnl_weekly_df = pnl_weekly_df.dropna(how="all")
weekly_returns = pnl_weekly_df.diff().dropna()
cov_matrix=weekly_returns.cov()
corr_matrix = weekly_returns.corr()
print("✅ 策略间相关系数矩阵:")
print(corr_matrix.round(3))
cob=cov_matrix
corr=corr_matrix
dist=((1-corr)/2.)**.5 # distance matrix 
link=sch.linkage(dist,'single') # linkage matrix  执行层次聚类，将策略分层次聚合为不同集合

#a.2 获取准对角排序
def getQuasiDiag(link):
    """
    对层次聚类的链接矩阵（linkage matrix）进行准对角化排序（quasi-diagonalization）。
    
    该函数通过递归展开聚类树的叶子节点，得到一个资产/策略的重排序列表，
    使得重排后的相关性矩阵呈现“块状对角”结构，便于后续如分层风险平价（HRP）等操作。
    
    参数
    ----------
    link : numpy.ndarray
        来自 scipy.cluster.hierarchy.linkage 的链接矩阵，形状为 (n-1, 4)
        
    返回值
    -------
    list
        按准对角顺序排列的原始资产索引列表（即叶节点的遍历顺序）
    """
    # 复制链接矩阵以避免修改原始数据
    link = link.copy()
    
    # 原始资产总数（链接矩阵最后一行第4列）
    n_items = int(link[-1, 3])
    
    # 从最后一次合并开始：包含两个根子树的索引
    sort_ix = pd.Series([int(link[-1, 0]), int(link[-1, 1])])
    
    # 循环展开所有非叶节点（即索引 >= n_items 的簇）
    while sort_ix.max() >= n_items:
        # 将当前索引间隔拉大一倍（为插入新元素预留空间）
        sort_ix.index = range(0, len(sort_ix) * 2, 2)
        
        # 找出所有代表“簇”（非原始资产）的项
        cluster_mask = sort_ix >= n_items
        df0 = sort_ix[cluster_mask]
        
        # 如果没有簇了，跳出循环（理论上不会发生，但安全起见）
        if df0.empty:
            break
        
        # 获取这些簇在链接矩阵中的行号（需减去 n_items）
        idx_positions = df0.index                    # 当前 Series 中的位置
        cluster_indices = (df0 - n_items).astype(int)  # 对应 link 矩阵的行索引
        
        # 从链接矩阵中取出每个簇的两个子节点
        left_children = link[cluster_indices, 0].astype(int)   # 左子节点
        right_children = link[cluster_indices, 1].astype(int)  # 右子节点
        
        # 用左子节点替换原位置的簇编号
        sort_ix.loc[idx_positions] = left_children
        
        # 在紧邻的下一个位置插入右子节点
        right_series = pd.Series(right_children, index=idx_positions + 1)
        
        # 使用 pd.concat 替代已弃用的 .append()
        sort_ix = pd.concat([sort_ix, right_series])
        
        # 按索引排序，恢复顺序
        sort_ix = sort_ix.sort_index()
        
        # 重置索引为连续整数：0, 1, 2, ...
        sort_ix.index = range(len(sort_ix))
    
    # 确保返回的是整数列表
    return sort_ix.astype(int).tolist()


sorted_indices = getQuasiDiag(link)
sorted_indices=corr.index[sorted_indices].tolist() #恢复label
print("Quasi-diagonal order:", sorted_indices)

#原python2版本
# def getQuasiDiag(link):
# # Sort clustered items by distance
#     link=link.astype(int)
#     sortIx=pd.Series([link[-1,0],link[-1,1]])
#     numItems=link[-1,3] # number of original items
#     while sortIx.max()>=numItems:
#         sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
#         df0=sortIx[sortIx>=numItems] # find clusters
#         i=df0.index;j=df0.values-numItems
#         sortIx[i]=link[j,0] # item 1
#         df0=pd.Series(link[j,1],index=i+1)
#         sortIx =sortIx.append(df0) # item 2
#         sortIx=sortIx.sort_index() # re-sort
#         sortIx.index=range(sortIx.shape[0]) # re-index
#     return sortIx.tolist()

#a.3 计算递归二分权重
def getRecBipart(cov, sortIx):
    """
    基于准对角排序后的资产顺序，递归二分计算分层风险平价（HRP）权重。
    
    该函数从顶层聚类开始，逐层将每个簇二分为左右子簇，
    并根据子簇的风险（方差）反比分配权重，实现“高风险少配，低风险多配”。
    
    参数
    ----------
    cov : pandas.DataFrame
        资产收益率的协方差矩阵，索引和列均为资产名称或编号
    sortIx : list
        经过准对角化排序后的资产索引列表（如 [3, 1, 4, 0, 2]）
        
    返回值
    -------
    w : pandas.Series
        每个资产的 HRP 权重，索引为原始资产标识
    """

    
    # 初始化：所有资产权重设为 1（后续通过乘法逐步缩放）
    w = pd.Series(1.0, index=sortIx)
    
    # 初始时，所有资产在一个簇中
    cItems = [sortIx]
    
    # 只要还有可分割的簇，就继续二分
    while len(cItems) > 0:
        # 对每个簇进行二等分（仅当簇长度 > 1 时）
        # 例如 [A,B,C,D] → [A,B] 和 [C,D]
        cItems = [
            i[j:k] 
            for i in cItems 
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) 
            if len(i) > 1
        ]
        
       
        # 成对处理相邻的两个子簇（左子簇和右子簇）
        for i in range(0, len(cItems), 2):
            cItems0 = cItems[i]     # 左子簇
            cItems1 = cItems[i + 1] # 右子簇
            
            # 计算两个子簇的“组合风险”（即簇内资产按等权组合的方差）
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            
            # 根据风险反比分配权重：
            # 风险越高的簇，分配的权重越小
            alpha = 1 - cVar0 / (cVar0 + cVar1)  # 分配给左簇的比例
            
            # 将当前权重乘以分配比例（递归缩放）
            w[cItems0] *= alpha      # 左子簇权重缩放
            w[cItems1] *= (1 - alpha)  # 右子簇权重缩放
    
    return w

def getIVP(cov,**kargs):
# Compute the inverse-variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp
def getClusterVar(cov,cItems):
# Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar

HRP_weights = getRecBipart(cov_matrix, sorted_indices)
print("HRP配置:", HRP_weights)

'''
Strategy_4     0.009198
Strategy_2     0.017268
Strategy_1     0.067232
Strategy_9     0.245780
Strategy_6     0.020244
Strategy_7     0.090692
Strategy_5     0.318201
Strategy_10    0.011657
Strategy_3     0.186566
Strategy_8     0.033162

'''


# 第一组权重（hrpv2['weights'] 结果）
weights1 = [('Strategy_1', 0.06726624998303406),
             ('Strategy_10', 0.011657747463686032),
             ('Strategy_2', 0.017274772262615142),
             ('Strategy_3', 0.1864112259575438),
             ('Strategy_4', 0.009201850023587761),
             ('Strategy_5', 0.31820538042220353),
             ('Strategy_6', 0.02025507346943731),
             ('Strategy_7', 0.090699377078885),
             ('Strategy_8', 0.033169540588812794),
             ('Strategy_9', 0.2458587827501946)]

# 第二组权重（HRP_weights）
weights2 = [
    ('Strategy_4', 0.009198),
    ('Strategy_2', 0.017268),
    ('Strategy_1', 0.067232),
    ('Strategy_9', 0.245780),
    ('Strategy_6', 0.020244),
    ('Strategy_7', 0.090692),
    ('Strategy_5', 0.318201),
    ('Strategy_10', 0.011657),
    ('Strategy_3', 0.186566),
    ('Strategy_8', 0.033162)
]

# 转为 pandas Series（自动对齐索引）
s1 = pd.Series(dict(weights1))
s2 = pd.Series(dict(weights2))

# 确保两个 Series 按相同顺序对齐（按 Strategy 名称）
s1 = s1.sort_index()
s2 = s2.sort_index()

# 计算差值（第一组 - 第二组）
diff = s1 - s2

# 构建对比 DataFrame
comparison = pd.DataFrame({
    'Weight_Group1': s1,
    'Weight_Group2': s2,
    'Difference (G1 - G2)': diff,
    'Abs_Difference': diff.abs(),
    'Relative_Change (%)': (diff / s2 * 100).round(2)  # 相对于第二组的百分比变化
})

# 按绝对差值降序排序，看哪些策略变动最大
comparison_sorted = comparison.sort_values('Abs_Difference', ascending=False)

print(comparison_sorted)