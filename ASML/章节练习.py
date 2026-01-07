import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import statsmodels.stats.diagnostic as sm
import statsmodels.api as smi
import datetime as dt

#连外网
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# import research as rs

#%%
#注：书里是用python2写的代码。现在用的是python3，有挺多细节是不一样的。

#一些书里用到的函数,书里是基于python2的，需要转为python3，
#使用多核cpu执行函数
# def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs): 
#     ''' 
#     Parallelize jobs, return a DataFrame or Series 
#     + func: function to be parallelized. Returns a DataFrame 
#     + pdObj[0]: Name of argument used to pass the molecule 
#     + pdObj[1]: List of atoms that will be grouped into molecules 
#     + kargs: any other argument needed by func
#     Example: df1=mpPandasObj(func,(’molecule’,df0.index),24,**kargs)
#     '''
#     import pandas as pd 
#     if linMols:
#         parts=linParts(len(pdObj[1]),numThreads*mpBatches) 
#     else:
#         parts=nestedParts(len(pdObj[1]),numThreads*mpBatches) 
#     jobs=[] 
#     for i in range(1,len(parts)): 
#         job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func} 
#         job.update(kargs) 
#         jobs.append(job) 
#     if numThreads==1:
#         out=processJobs_(jobs) 
#     else:
#         out=processJobs(jobs,numThreads=numThreads) 
#     if isinstance(out[0],pd.DataFrame):
#         df0=pd.DataFrame() 
#     elif isinstance(out[0],pd.Series):
#         df0=pd.Series() 
#     else:
#         return out 
#     for i in out:
#         df0=df0.append(i) 
#     return df0.sort_index()

# #mpPandasObj附属函数1
# def nestedParts(numAtoms,numThreads,upperTriang=False):
#     # partition of atoms with an inner loop
#     parts,numThreads_=[0],min(numThreads,numAtoms)
#     for num in xrange(numThreads_):
#         part=1 + 4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
#         part=(-1+part**.5)/2.
#         parts.append(part)
#     parts=np.round(parts).astype(int)
#     if upperTriang: # the first rows are the heaviest
#         parts=np.cumsum(np.diff(parts)[::-1])
#         parts=np.append(np.array([0]),parts)
#     return parts

# #mpPandasObj附属函数2
# def linParts(numAtoms,numThreads):
#     # partition of atoms with a single loop
#     parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
#     parts=np.ceil(parts).astype(int)
#     return parts

#下面是转为python3版本的代码
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# ==============================
# 1. linParts: 线性均匀分块
# ==============================
def linParts(numAtoms, numThreads):
    """将 numAtoms 个原子均匀划分为 numThreads 份（线性分块）"""
    # 至少 1 份，最多 numAtoms 份
    n_parts = min(numThreads, numAtoms)
    parts = np.linspace(0, numAtoms, n_parts + 1)
    parts = np.ceil(parts).astype(int)
    return parts


# ==============================
# 2. nestedParts: 非均匀分块（用于计算负载不均场景）
# ==============================
def nestedParts(numAtoms, numThreads, upperTriang=False):
    """
    嵌套分块：前面的块更大（适用于前段计算更重的情况）
    """
    parts = [0]
    numThreads_ = min(numThreads, numAtoms)
    
    for num in range(numThreads_):  # ← Python 3: xrange → range
        # 注意：原公式中的除法在 Python 3 中已是 float 除法
        part = 1 + 4 * (parts[-1]**2 + parts[-1] + numAtoms * (numAtoms + 1.) / numThreads_)
        part = (-1 + np.sqrt(part)) / 2.
        parts.append(part)
    
    parts = np.round(parts).astype(int)
    
    if upperTriang:
        # 反转块大小：第一个块最重
        diffs = np.diff(parts)[::-1]
        parts = np.cumsum(diffs)
        parts = np.concatenate([[0], parts])
    
    return parts


# ==============================
# 3. 辅助函数：执行单个 job
# ==============================
def _run_job(job_dict):
    """运行单个任务：调用 func(**job_dict)"""
    func = job_dict.pop('func')
    return func(**job_dict)


# ==============================
# 4. 单线程执行器
# ==============================
def processJobs_(jobs):
    """单线程顺序执行任务（用于调试或 numThreads=1）"""
    return [_run_job(job) for job in jobs]


# ==============================
# 5. 多线程执行器
# ==============================
def processJobs(jobs, numThreads=24):
    """多线程并行执行任务"""
    if not jobs:
        return []
    
    max_workers = min(numThreads, multiprocessing.cpu_count(), len(jobs))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_job, job) for job in jobs]
        results = [f.result() for f in futures]
    return results


# ==============================
# 6. 主函数：mpPandasObj（Python 3 兼容版）
# ==============================
def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    """
    并行化处理 pandas 对象（DataFrame/Series），返回合并后的结果。
    不适用于：需要聚合所有数据才能计算的全局统计量（如整体 mean/std/corr）
    是将每个“molecule”独立产出一个结果
    
    参数:
        func: 要并行化的函数，必须接受一个名为 pdObj[0] 的参数（如 'molecule'）
        pdObj: tuple (arg_name: str, atoms: array-like)，例如 ('molecule', df.index)
        numThreads: 并行线程数
        mpBatches: 批次数（增大可提高负载均衡）
        linMols: True=线性分块，False=嵌套分块
        **kargs: 传递给 func 的其他参数
    
    示例:
        result = mpPandasObj(my_func, ('molecule', df.index), numThreads=8, data=df, clf=clf)
    """
    arg_name, atoms = pdObj
    num_atoms = len(atoms)

    # 分块
    if linMols:
        parts = linParts(num_atoms, numThreads * mpBatches)
    else:
        parts = nestedParts(num_atoms, numThreads * mpBatches)

    # 构建任务列表
    jobs = []
    for i in range(1, len(parts)):
        # 每个任务包含：分子（索引子集）+ 函数 + 其他参数
        job = {
            arg_name: atoms[parts[i-1]:parts[i]],
            'func': func
        }
        job.update(kargs)
        jobs.append(job)

    # 执行任务
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)

    # 合并结果
    if not out:
        raise ValueError("No results returned from parallel jobs.")

    first_result = out[0]
    if isinstance(first_result, pd.DataFrame):
        result = pd.concat(out, axis=0)
    elif isinstance(first_result, pd.Series):
        result = pd.concat(out, axis=0)
    else:
        # 非 pandas 对象，直接返回列表
        return out

    return result.sort_index()


#%%
#2-5章是数据分析。对获得到的数据进行分析，清洗，处理至可用
#AFML 第二章
##########制造基础数据
def create_price_data(start_price: float = 1000.00,
                      mu: float = .0,
                      var: float = 1.0,
                      n_samples: int = 1000000):
                      


    i = np.random.normal(mu, var, n_samples)
    df0 = pd.date_range(periods=n_samples,
                        freq=pd.tseries.offsets.Minute(),
                        end=pd.Timestamp.now())
                        
    X = pd.Series(i, index=df0, name = "close").to_frame()
    X.close.iat[0] = start_price
    X.cumsum().plot.line()
    return X.cumsum()
df=create_price_data()
df.to_csv(r'D:\Git\book\ASML\dollar_bars.csv')   #制造基础的数据
df=create_price_data()
df.to_csv(r'D:\Git\book\ASML\volume_bars.csv')   #制造基础的数据
df=create_price_data()
df.to_csv(r'D:\Git\book\ASML\tick_bars.csv')   #制造基础的数据






################# 查询与正态分布的相似程度——通过偏度和峰值的检验。这样就不需要画图了
from scipy import stats

p(stats.jarque_bera(dollar['close'].pct_change().dropna())[0],
stats.jarque_bera(volume['close'].pct_change().dropna())[0],
stats.jarque_bera(tick['close'].pct_change().dropna())[0])




'''
2.1 On a series of E-mini S&P 500 futures tick data:
(a) Form tick, volume, and dollar bars. Use the ETF trick to deal with the roll.
(b) Count the number of bars produced by tick, volume, and dollar bars on a
weekly basis. Plot a time series of that bar count. What bar type produces
the most stable weekly count? Why?
(c) Compute the serial correlation of returns for the three bar types. What bar
method has the lowest serial correlation?
(d) Partition the bar series into monthly subsets. Compute the variance of returns
for every subset of every bar type. Compute the variance of those variances.
What method exhibits the smallest variance of variances?
(e) Apply the Jarque-Bera normality test on returns from the three bar types.
What method achieves the lowest test statistic
'''

#Jarque-Bera normality test 对接近正态分布进行检测
data=pd.read_csv(r'D:\Git\book\ASML\dollar_bars.csv',index_col=[0])

# 使用对数收益率
data['returns'] = np.log(data['close'] / data['close'].shift(1))

# 删除缺失值（第一行会变成 NaN）
returns = data['returns'].dropna()



#c 统计滞后相关性：

returns = data['returns'].dropna()

# 2. 计算序列的自相关系数（可以选择不同滞后期数，这里取1期滞后）
lag = 1  # 滞后期数
serial_corr = returns.autocorr(lag=lag)

print(f"收益率的{lag}期滞后自相关系数: {serial_corr}")

#d ：每月子集的收益率方差
data.index = pd.to_datetime(data.index)
data['month'] = data.index.to_period('M')  # 使用索引中的日期提取月份

# 按月份分组并计算方差
monthly_variance = data.groupby('month')['returns'].var()
# 输出每个月的方差
print(monthly_variance)


# 绘制每个月的收益率方差
plt.figure(figsize=(10, 6))
monthly_variance.plot(kind='bar')
plt.title('Monthly Variance of Returns')
plt.xlabel('Month')
plt.ylabel('Variance')
plt.xticks(rotation=45)
plt.show()

#e ： Jarque-Bera 正态性检验

jb_stat, jb_pvalue = jarque_bera(returns)
print("Jarque-Bera 统计量:", jb_stat)
print("p-value:", jb_pvalue)

# 4. 判断正态性
if jb_pvalue > 0.05:
    print("不能拒绝正态性假设，收益率可以视为近似正态分布")
else:
    print("拒绝正态性假设，收益率不符合正态分布")

'''
2.2 On a series of E-mini S&P 500 futures tick data, compute dollar bars
and dollar imbalance bars. What bar type exhibits greater serial correlation?
Why?

'''



######################## 简单的构建dollar bar  ,交易金额累计达到一定的门槛就resample为新的bar
def dd_bars(data: pd.DataFrame, m: int = None):
    '''
    params: data => dataframe of close series
    params: column => column of data sample; vol, dollar etc  累计阈值门槛，达到就重采样
    '''    
    ts, idx = 0, []
    for i, x in enumerate(data):
        ts += x
        if ts >= m:
            ts = 0; idx.append(i)
            continue
    return data.iloc[idx]
tb = dd_bars(data = data.close, m = 1000000) # assuming 10% of daily transacted volme is 1,000,000


################### 构建复杂的imbalance bar样例，比如考虑涨跌进行加权的情况，还可以加上连续正收益计数
def returns(data, tickers):
    b_t = []
    _ = data[tickers].pct_change()
    _.dropna(inplace=True)
    for i, value in enumerate(_):
        b_t.append(value)
    return b_t

def ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, limit, alpha, T_count):
    weighted_sum_T = limit + (1 - alpha) * weighted_sum_T
    weighted_sum = limit / (1.0 * T_count) + (1 - alpha) * weighted_sum
    weighted_count = 1 + (1 - alpha) * weighted_count
    imbalance = weighted_sum_T * weighted_sum/ weighted_count ** 2
    return imbalance, weighted_count, weighted_sum, weighted_sum_T

def imbalance_bar(data, tickers, set_limit, alpha):
    b_t = returns(data, tickers)
    bt_arr = []
    imb_arr = []
    weighted_sum_T = 0
    weighted_sum = 0
    weighted_count = 0
    bt_count = 0
    bt_up = 0
    b_imb_sum = 0
    b_sum = 0
    imbalance = 0
    for i, value in enumerate(b_t):
        bt_count += 1
        if value >= 0:
            b_sum += b_t[i]
            b_imb_sum += 1
            bt_up += 1
            bt_arr.append(b_sum)
        else:
            b_imb_sum -= 1
            b_sum += b_t[i]
            bt_arr.append(b_sum)
            
        upper_limit = max(b_imb_sum, bt_up)
        if upper_limit >= set_limit:
            imbalance, weighted_count, weighted_sum, weighted_sum_T = ema_tick(imbalance, 
                                                                               weighted_count,
                                                                               weighted_sum,
                                                                               weighted_sum_T,
                                                                               upper_limit,
                                                                               alpha,
                                                                               bt_count)
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            if upper_limit == bt_up:
                bt_up = 0
            else:
                b_imb_sum = 0
        else:
            imb_arr.append(0.0)    
    return bt_arr, imb_arr, b_t  #累计收益率，不平衡信号，收益率list。生成了不平衡收益率触发信号





'''
2.3 On dollar bar series of E-mini S&P 500 futures and Eurostoxx 50 futures:
(a) Apply Section 2.4.2 to compute the { ̂𝜔t} vector used by the ETF trick. (Hint:
You will need FX values for EUR/USD at the roll dates.)
(b) Derive the time series of the S&P 500/Eurostoxx 50 spread.
(c) Confirm that the series is stationary, with an ADF test.

'''
#a 使用pca方法（分解为不相关的主成分）对各个资产的风险进行配置权重，然后使用资产组合方差公式进行配置方差（ax+by如果相关系数为0，则方差为a方*x方差+b方*y方差），可以动态调整从而保持暴露的风险在同一水平
def pcaWeights(cov,riskDist=None,riskTarget=1.):
    # Following the riskAlloc distribution, match riskTarget
    eVal,eVec=np.linalg.eigh(cov) # must be Hermitian
    indices=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[indices],eVec[:,indices]
    if riskDist is None:
        riskDist=np.zeros(cov.shape[0])
        riskDist[-1]=1.
    loads=riskTarget*(riskDist/eVal)**.5
    wghts=np.dot(eVec,np.reshape(loads,(-1,1)))
    #ctr=(loads/riskTarget)**2*eVal # verify riskDist
    return wghts


#b
# 计算一阶差分
data['close_diff'] = data['close'].diff()

# 删除 NaN 值（差分后第一个值会为 NaN）
data = data.dropna()

#c ADF检验，用于检验时间序列是否平稳。（常用于统计套利-价差回归）
# 执行 ADF 检验  平稳的数据有稳定的统计特性：均值，方差，协方差等等是稳定的，能够预测数据
adf_result = adfuller(data['close_diff'])  #大数据很卡，很吃内存

# 输出检验结果
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
print('Critical Values:', adf_result[4])

'''
2.4 Form E-mini S&P 500 futures dollar bars:
(a) Compute Bollinger bands of width 5% around a rolling moving average.
Count how many times prices cross the bands out (from within the bands
to outside the bands).
(b) Now sample those bars using a CUSUM filter, where {yt} are returns and
h = 0.05. How many samples do you get?
(c) Compute the rolling standard deviation of the two-sampled series. Which
one is least heteroscedastic? What is the reason for these results

2.5 Using the bars from exercise 4:
(a) Sample bars using the CUSUM filter, where {yt} are absolute returns and
h = 0.05.
(b) Compute the rolling standard deviation of the sampled bars.
(c) Compare this result with the results from exercise 4. What procedure delivered the least heteroscedastic sample? Why?

'''



#a 5%布林带及其内穿外
window = 20  # 计算布林带所用的移动窗口
std_multiplier = 0.05  # 5%标准差倍数

tb=pd.DataFrame(tb)
tb['close'] = pd.to_numeric(tb['close'], errors='coerce')  # 强制转换为数值类型

# tb=tb.iloc[:-2,:]
# 计算简单移动平均 (SMA) 和标准差
tb['SMA'] = tb['close'].rolling(window=window).mean()  # 计算SMA
tb['Std'] = tb['close'].rolling(window=window).std()  # 计算标准差

# 计算布林带的上轨和下轨
tb['Upper_Band'] = tb['SMA'] + (std_multiplier * tb['Std'])
tb['Lower_Band'] = tb['SMA'] - (std_multiplier * tb['Std'])

# 检查穿越次数
cross_up = 0  # 从内穿到外上轨
cross_down = 0  # 从内穿到外下轨

# 遍历时间序列，检查每个点是否穿越了布林带
for i in range(1, len(tb)):
    # 判断是否从内到外穿越
    if tb['close'][i] > tb['Upper_Band'][i] and tb['close'][i-1] <= tb['Upper_Band'][i-1]:
        cross_up += 1
    elif tb['close'][i] < tb['Lower_Band'][i] and tb['close'][i-1] >= tb['Lower_Band'][i-1]:
        cross_down += 1

# 输出穿越次数
print(f"从内到外穿越上轨的次数: {cross_up}")
print(f"从内到外穿越下轨的次数: {cross_down}")

#b   CUSUM filter. 两边正负对称版。（正负累加值可以不对称阈值，还可以加预期收益值进行介入，见章节2.5.2.1）
#起到了滤波器的效果，对震荡行情能够过滤。
#CUSUM filter 到底是对价格的变化，还是对收益率的变化累计出来的结果更有效呢？需要在这里的2.4，2.5进行检验一下。
#所以这里有三个：价格的变化，收益变化，收益率的差变化。
#对收益率的差变化进行CUSUM filter还可以进行区分市场是出于回归还是趋势类状态。

# 基于价格差的CUSUM filter
def cumsum_events(df: pd.Series, limit: float):
    idx, _up, _dn = [], 0, 0
    diff = df.diff()
    for i in diff.index[1:]:
        _up, _dn = max(0, _up + diff.loc[i]), min(0, _dn + diff.loc[i])
        if _up > limit:
            _up = 0; idx.append(i)
        elif _dn < - limit:
            _dn = 0; idx.append(i)
        
    return pd.DatetimeIndex(idx)

# 基于收益率差的CUSUM filter
def cumsum_events1(df: pd.Series, limit: float):
    idx, _up, _dn = [], 0, 0
    diff = df.pct_change()
    for i in diff.index[1:]:
        _up, _dn = max(0, _up + diff.loc[i]), min(0, _dn + diff.loc[i])
        if _up > limit:
            _up = 0; idx.append(i)
        elif _dn < - limit:
            _dn = 0; idx.append(i)
        
    return pd.DatetimeIndex(idx)


#事件构建的阈值也可以基于标准偏差来构建，而不是主观臆断
#按照百分比和标准偏差构建的事件能够通过怀特检验，是同方差，而benchmark这样主观固定数值构建的事件无法通过。
# 使用对数收益率
tb['returns'] = np.log(tb['close'] / tb['close'].shift(1))
tb = tb.dropna()


event = cumsum_events(tb['close'], limit = 0.005) # benchmark
event_pct = cumsum_events1(tb['close'], limit = 0.005) #基于百分比构建事件
event_abs = cumsum_events(tb['close'], limit = tb['Std'].mean()) # 基于标准+标准差阈值
event_pct2 = cumsum_events(tb['returns'], limit = 0.005) #基收益率差

tb.index = pd.to_datetime(tb.index)

event_count0 = tb.reindex(event)
event_count1 = tb.reindex(event_abs)
event_count2 = tb.reindex(event_pct)
event_count3 = tb.reindex(event_pct2)


#White Test 同方差检验
def white_test(data: pd.DataFrame, window: int = 21):
    data['std1'] = data['close'].rolling(21).std()
    data.dropna(inplace= True)
    X = smi.tools.tools.add_constant(data['close'])
    results = smi.regression.linear_model.OLS(data['std1'], X).fit()
    resid = results.resid
    exog = results.model.exog
    print("White-Test p-Value: {0}".format(sm.het_white(resid, exog)[1]))
    if sm.het_white(resid, exog)[1] > 0.05:
        print("White test outcome at 5% signficance: 同方差")
    else:
        print("White test outcome at 5% signficance: 异方差")


white_test(event_count0)  #异方差
white_test(event_count1)  #同方差，p值0.35
white_test(event_count2)  #异方差
white_test(event_count3)  #同方差,p值0.66



#%%
'''
第三章：元标签（meta label）
在有下注方向有如何确定是否下注

'''

import numpy as np
import pandas as pd
# import research as rs
import matplotlib.pyplot as plt



'''
3.1 Form dollar bars for E-mini S&P 500 futures:
(a) Apply a symmetric CUSUM filter (Chapter 2, Section 2.5.2.1) where the
threshold is the standard deviation of daily returns (Snippet 3.1).
(b) Use Snippet 3.4 on a pandas series t1, where numDays=1.
(c) On those sampled features, apply the triple-barrier method, where
ptSl=[1,1] and t1 is the series you created in point 1.b.
(d) Apply getBins to generate the labels.

'''
#a  找到事件
dollar = pd.read_csv(r'D:\Git\book\ASML\dollar_bars.csv'   ,
                     parse_dates=True,      # 解析日期列
                     index_col=[0]  # 将 'date_time' 列作为索引
                     )
tb = dd_bars(data = dollar.close, m = 1000000)
#a 5%布林带及其内穿外
window = 20  # 计算布林带所用的移动窗口
tb=pd.DataFrame(tb)
tb['close'] = pd.to_numeric(tb['close'], errors='coerce')  # 强制转换为数值类型
tb['returns'] = np.log(tb['close'] / tb['close'].shift(1))
tb = tb.dropna()
# 计算移动标准差
def getDailyVol(close,span0=100):
# daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0 - 1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
    df0=df0.ewm(span=span0).std()
    return df0
std=getDailyVol(tb.close,span0=100)
std=pd.DataFrame(std).rename(columns={'close': 'daily_vol'})
# 按 index 合并
tb = tb.join(std)
def cumsum_events3(df: pd.Series, limit: pd.Series):
    idx, _up, _dn = [], 0, 0
    diff = df.diff()
    
    # 确保 limit 和 df 有相同的索引
    if not df.index.equals(limit.index):
        raise ValueError("The index of 'limit' must match the index of 'df'.")
    
    for i in diff.index[1:]:
        # 使用与 df 当前索引对应的 limit 值
        current_limit = limit.loc[i]
        
        _up = max(0, _up + diff.loc[i])
        _dn = min(0, _dn + diff.loc[i])
        
        # 如果累计值超过 limit，重置累计值
        if _up > current_limit:
            _up = 0
            idx.append(i)
        elif _dn < -current_limit:
            _dn = 0
            idx.append(i)
        
    return pd.DatetimeIndex(idx)

event_pct3=cumsum_events3(tb.returns, tb.daily_vol)

#b 给事件加上1天的时间长度
numDays=1
close=tb.close
tEvents=event_pct3
t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays)) #返回第一个大于等于每个调整后的事件时间的位置，即大于1天的第一个close的时间
t1=t1[t1<close.shape[0]]
t1=pd.Series(close.index[t1],index=tEvents[:t1.shape[0]])
t1.name = 't1'

#c the triple-barrier method ，找到事件在指定箱体内先碰到哪条边，然后获取对应的时间，
#得根据起点的状态构建目标上下限，否则有未来函数
#原文是基于收益的，而不是基于价格。原文还有一个买入方向来判断多空，我这里也没判断，就默认为全是做多。
def applyPtSlOnT1(close:pd.Series, events:pd.DataFrame, ptSl:list, daily_vol:pd.Series):
    out = events[['t1']].copy()  # 复制事件数据框，只保留t1列
    out['pt_time'] = pd.NaT  # 初始化pt_time列
    out['sl_time'] = pd.NaT  # 初始化sl_time列
    
    for loc, row in events.iterrows():
        t1 = row['t1']
        
        # 获取事件的起始时间和终止时间
        start_time = loc
        end_time = t1
        
        # 获取事件的价格数据（起始时间到终止时间之间）
        price_data = close[start_time:end_time]
        
        # 计算上下限
        upper_limit = close[start_time] * (1 + daily_vol[start_time]) * ptSl[0]
        lower_limit = close[start_time] * (1 - daily_vol[start_time]) * ptSl[1]
        
        # 查找首次触及上限和下限的时间点
        pt_time = price_data[price_data >= upper_limit].index.min()  # 上限触及时间
        sl_time = price_data[price_data <= lower_limit].index.min()  # 下限触及时间
        
        # 如果触及时间存在，则记录，若没有触及则保持为NaT
        out.loc[loc, 'pt_time'] = pd.to_datetime(pt_time, errors='coerce') if pd.notna(pt_time) else pd.NaT
        out.loc[loc, 'sl_time'] =  pd.to_datetime(sl_time, errors='coerce') if pd.notna(sl_time) else pd.NaT
    
    return out

ptSl=[1,1]
events=pd.DataFrame(t1)
result = applyPtSlOnT1(tb.close, events, ptSl, tb.daily_vol)


#d 获取事件对应的箱型标签,就不仅仅是获取时间了。到这一步就是将波动率较低的事件给剔除，然后再剩余的事件里面执行箱型规则，并且记录到每个事件的最终终点时间（不管是箱型规则三边的那一边），以及对应的trgt
#trgt是事件的水平障碍目标（止盈止损），绝对收益，这里可以用std替代
#指最小收益率，在这里的作用是将波动/收益较小的事件给过滤掉
def getEvents(close,tEvents,ptSl,trgt,minRet,t1=False):
    #1) get target
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet
    #2) get t1 (max holding period)
    if t1 is False:
        t1=pd.Series(pd.NaT,index=tEvents)
    #3) form events object, apply stop loss on t1
    side_=pd.Series(1.,index=trgt.index)
    events=pd.concat({'t1':t1,'trgt':trgt,'side':side_},axis=1).dropna(subset=[('trgt')])
    #df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),numThreads=numThreads,close=close,events=events,ptSl=[ptSl,ptSl])
    df0=applyPtSlOnT1(close, events, ptSl, events.trgt)
    events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    events=events.drop('side',axis=1)
    return events

trgt=std['daily_vol']
minRet=0.035
result_getEvents= getEvents(close,tEvents,ptSl,trgt,minRet,t1)

#获取对应事件的实际收益与实际要执行的正确方向。
#也就是这个事件最终应该是label为什么样的方向
def getBins(events,close):
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin']=np.sign(out['ret'])  #数据根据原来的数值转为【-1，0，1】中的一个
    return out

result_getBins=getBins(events,close)

'''
3.2 From exercise 3.1, use Snippet 3.8 to drop rare labels.

'''
#去掉出现频率低于0.05的标签。
#去掉极端标签，有利于机器学习识别
def dropLabels(events,minPtc=0.05):
# apply weights, drop labels with insufficient examples
    while True:
        df0=events['bin'].value_counts(normalize=True)
        if df0.min()>minPtc or df0.shape[0]<3:
            break
        print ('dropped label',df0.argmin(),df0.min())
        events=events[events['bin']!=df0.argmin()]
    return events

result_getBins=dropLabels(events=result_getBins,minPtc=0.05)



'''
3.3 Adjust the getBins function (Snippet 3.5) to return a 0 whenever the vertical
barrier is the one touched first
'''
#在获取到result_getEvents的基础上进行改造，即起点和终点。
def getBins2(events,t1,close):
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin']=np.sign(out['ret'])  #数据根据原来的数值转为【-1，0，1】中的一个
    #如果事件终点的值等于垂直障碍，则赋值为0 
    out=out.merge(events_['t1'],how='inner', left_index=True, right_index=True)
    out=out.merge(t1,how='inner', left_index=True, right_index=True)
    out['bin']= np.where(out['t1_x'] == out['t1_y'], 0, out['bin'])
    return out

result_getBins2=getBins2(events=result_getEvents,t1=t1,close=close)




'''
3.4 Develop a trend-following strategy based on a popular technical analysis statistic
(e.g., crossing moving averages). For each observation, the model suggests a side,
but not a size of the bet.
(a) Derive meta-labels for ptSl=[1,2] and t1 where numDays=1. Use as
trgt the daily standard deviation as computed by Snippet 3.1.
(b) Train a random forest to decide whether to trade or not. Note: The decision
is whether to trade or not, {0,1}, since the underlying  model (the crossing moving average) has decided the side, {−1,1}.

'''

#meta-labels:对模型发出的信号进行确认。负责解决“当主模型发出信号时，我到底该不该相信它？“,通过分类学习主模型信号是否盈利来提升概率/置信。即信号发出后置信区间大于阈值再执行，如概率大于0.6再执行
#可以独立的优化主模型与元模型（meta模型），主模型是无论是统计模型、机器学习模型还是基于规则的系统都可以应用，元模型也可以继续使用市场特征（价格，交易量等），或者因子等作为特征集。
#还可以使用元模型来确定买入的头寸大小，这样就构成了主模型确定方向，元模型确定头寸。
# 具体：Deepseek说下注的大小是跟下注成功概率挂钩，可以加入预期盈亏比引入凯利公式，也可以将这个概率结合波动率，相关性等结合，是风险管理止盈止损都可以用的上的。都是在metalabel获取信号成功概率基础上进行的。


#a
#使用dollar bar金叉死叉构建简单的交易。
# 计算移动平均线
short_ma = close.rolling(window=5).mean()
long_ma = close.rolling(window=60).mean()

# 创建信号DataFrame
df = pd.DataFrame(index=close.index)
df['close'] = close
df['short_ma'] = short_ma
df['long_ma'] = long_ma
# 生成交易信号
# 金叉: 短均线上穿长均线 (买入信号)
# 死叉: 短均线下穿长均线 (卖出信号)
df['signal'] = 0  # 0表示无信号
# 计算金叉和死叉
golden_cross = (df['short_ma'] > df['long_ma']) & (df['short_ma'].shift(1) <= df['long_ma'].shift(1))
death_cross = (df['short_ma'] < df['long_ma']) & (df['short_ma'].shift(1) >= df['long_ma'].shift(1))
# 标记信号
df.loc[golden_cross, 'signal'] = 1  # 买入信号
df.loc[death_cross, 'signal'] = -1  # 卖出信号
# 提取所有交易信号的时间点
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]

print("买入信号发生时间:")
print(len(buy_signals.index))
print("\n卖出信号发生时间:")
print(len(sell_signals.index))

# # 可视化
# plt.figure(figsize=(12, 8))
# plt.plot(df.index, df['close'], label='Close Price', alpha=0.5)
# plt.plot(df.index, df['short_ma'], label='5-period MA', alpha=0.7)
# plt.plot(df.index, df['long_ma'], label='60-period MA', alpha=0.7)
# # 标记买入信号
# plt.scatter(buy_signals.index, buy_signals['close'], 
#             color='green', marker='^', s=100, label='Buy Signal')

# # 标记卖出信号
# plt.scatter(sell_signals.index, sell_signals['close'], 
#             color='red', marker='v', s=100, label='Sell Signal')
# plt.title('Golden Cross/Death Cross Trading Strategy')
# plt.legend()
# plt.grid(True)
# plt.show()

# 计算相邻时间点的时间差
time_diffs = df[df['signal'] != 0].index.to_series().diff().dropna()

avg_interval_seconds = time_diffs.dt.total_seconds().mean()
avg_interval = avg_interval_seconds/86400 # 转换为天数

numDays = avg_interval
# 
ptSl=[1,2]
std = getDailyVol(df['close'], span0=100)
std = pd.DataFrame(std).rename(columns={'close': 'daily_vol'})
# 将结果合并回原DataFrame
df = df.join(std)

t1 = df['close'].index.searchsorted(df['close'].index + pd.Timedelta(days=numDays))
t1 = t1[t1 < df.shape[0]]  # 确保不超出范围
t1 = pd.Series(df['close'].index[t1], index=df['close'].index[:t1.shape[0]])
t1.name = 't1'
# 将t1列添加到DataFrame
df = df.join(t1)
df_events = df[df['signal'] != 0]
df_events.head()

def generate_metalabels(df_events: pd.DataFrame, close: pd.Series, daily_vol: pd.Series, ptSl: list) -> pd.DataFrame:
    """
    生成三重障碍法的meta-labels
    注：可以改进为向量化操作再叠加分布式计算应用于大数据集。
    
    参数:
        df_events: 包含t1列的事件DataFrame
        close: 收盘价Series
        daily_vol: 每日波动率Series
        ptSl: [止盈倍数, 止损倍数]
    
    返回:
        添加了metallabel列的DataFrame
    """
    # 复制一份避免修改原数据
    df = df_events.copy()
    df['metallabel'] = 0  # 初始化为0
    
    for idx, row in df.iterrows():
        start_time = idx
        end_time = row['t1']
        
        # 获取价格序列
        price_series = close[start_time:end_time]
        if price_series.empty:
            continue
            
        # 计算上下限
        if ptSl[0]==0:
            upper_limit = np.inf  # 如果止盈为0，则设为无穷大
        else:
            upper_limit = close[start_time] * (1 + ptSl[0] * daily_vol[start_time])
        
        if ptSl[1]==0:
            lower_limit = -np.inf  # 如果止损为0，则设为负无穷大
        else:
            lower_limit = close[start_time] * (1 - ptSl[1] * daily_vol[start_time])
        
        # 检查是否触及止盈
        pt_touch = price_series[price_series >= upper_limit]
        # 检查是否触及止损
        sl_touch = price_series[price_series <= lower_limit]
        
        # 确定最先触及的障碍
        if not pt_touch.empty and not sl_touch.empty:
            # 两者都触及，看哪个先发生
            if pt_touch.index[0] < sl_touch.index[0]:
                df.loc[idx, 'metallabel'] = 1
            else:
                df.loc[idx, 'metallabel'] = -1
        elif not pt_touch.empty:
            df.loc[idx, 'metallabel'] = 1
        elif not sl_touch.empty:
            df.loc[idx, 'metallabel'] = -1
        # 如果都没触及，保持为0
        
    return df


df_events_with_labels = generate_metalabels(df_events, df['close'], df_events['daily_vol'], ptSl)


#b 写一个随机森林模型学习是否该下注
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. 准备特征和目标变量
# 创建目标变量：signal是否等于metallabel（正确方向）
df_events_with_labels['correct_direction'] = (df_events_with_labels['signal'] == df_events_with_labels['metallabel']).astype(int)

# 2. 特征工程
def prepare_features(df):
    """准备用于训练的特征数据"""
    features = pd.DataFrame(index=df.index)
    
    # 1. 价格动量特征
    features['returns'] = df['close'].pct_change() 
    features['returns_5'] = df['close'].pct_change(5)
    features['returns_20'] = df['close'].pct_change(20)
    
    # 2. 移动平均特征
    features['ma_ratio'] = df['short_ma'] / df['long_ma'] - 1
    features['ma_dist'] = (df['close'] - df['long_ma']) / df['long_ma']
    
    # 3. 波动率特征
    features['volatility'] = df['daily_vol']
    features['volatility_ratio'] = df['daily_vol'] / df['daily_vol'].rolling(20).mean()
    
    # 4. 信号特征
    features['signal'] = df['signal']

    #原始特征
    features['short_ma'] = df['short_ma']
    features['long_ma'] = df['long_ma']
    features['close'] = df['close']
    
    # 删除缺失值
    features = features.dropna()
    
    return features

features = prepare_features(df)
features=features.loc[df_events_with_labels.index]  # 只保留事件对应的特征

X = features
y = df_events_with_labels['correct_direction']

# 3. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. 训练随机森林模型
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'  # 处理类别不平衡
)
rf.fit(X_train, y_train)

# 5. 评估模型
print(classification_report(y_test, rf.predict(X_test)))

df_x_test = X_test.copy()


df_x_test['correct_direction'] = y_test  # 实际正确方向
df_x_test['metallabel'] = df_events_with_labels.loc[y_test.index, 'metallabel']  # 原始metallabel

# 3. 添加预测结果
df_x_test['predicted_direction'] = rf.predict(X_test)  # 预测方向

# 4. 计算预测准确率
accuracy = (df_x_test['predicted_direction'] == df_x_test['correct_direction']).mean()
print(f"模型预测准确率: {accuracy:.2%}")

# 5. 可选：添加预测概率
df_x_test['probability'] = rf.predict_proba(X_test)[:, 1]  # 预测为1的概率

# 6. 查看结果
display(df_x_test.head())  

#结论，这个信号模型效果极差，meta模型反馈出来的结果都是不进行下注，不管多空方向都是。




'''
3.5 Develop a mean-reverting strategy based on Bollinger bands. For each observation, the model suggests a side, but not a size of the bet.
(a) Derive meta-labels for ptSl=[0,2] and t1 where numDays=1. Use as trgt the daily standard deviation as computed by Snippet 3.1.
(b) Train a random forest to decide whether to trade or not. Use as features: volatility, serial correlation, and the crossing moving averages from
exercise 2.
(c) What is the accuracy of predictions from the primary model (i.e., if the secondary model does not filter the bets)? What are the precision, recall, and
F1-scores?
(d) What is the accuracy of predictions from the secondary model? What are the precision, recall, and F1-scores?
'''

#a

def bollinger_strategy(close, window=20, num_std=2):
    """
    基于布林带的均值回归策略
    :param close: 收盘价序列
    :param window: 移动平均窗口
    :param num_std: 标准差倍数
    :return: 包含信号列的DataFrame
    """
    # 计算布林带
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    
    # 生成信号
    signal = pd.Series(0, index=close.index)
    signal[close < lower_band] = 1    # 低于下轨，买入信号
    signal[close > upper_band] = -1   # 高于上轨，卖出信号
    
    # 创建结果DataFrame
    result = pd.DataFrame({
        'close': close,
        'rolling_mean': rolling_mean,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'signal': signal
    })
    
    return result

# 使用示例
df = bollinger_strategy(df['close'], window=20, num_std=2)
display(df.head())


import matplotlib.pyplot as plt
# 创建图表
plt.figure(figsize=(12, 6))

# 绘制收盘价和布林带
plt.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
plt.plot(df.index, df['rolling_mean'], label='Moving Average', color='blue', linestyle='--')
plt.plot(df.index, df['upper_band'], label='Upper Band', color='red', linestyle=':')
plt.plot(df.index, df['lower_band'], label='Lower Band', color='green', linestyle=':')

# 标记买入信号 (signal == 1)
buy_signals = df[df['signal'] == 1]
plt.scatter(buy_signals.index, buy_signals['close'], 
           label='Buy Signal', color='green', marker='^', s=100)

# 标记卖出信号 (signal == -1)
sell_signals = df[df['signal'] == -1]
plt.scatter(sell_signals.index, sell_signals['close'], 
           label='Sell Signal', color='red', marker='v', s=100)

# 填充布林带区域
plt.fill_between(df.index, df['upper_band'], df['lower_band'], 
               color='gray', alpha=0.2, label='Bollinger Band')

# 添加图例和标题
plt.title('Bollinger Band Strategy Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# 显示图表
plt.show()


# 计算相邻时间点的时间差 作为平均持仓时间
time_diffs = df[df['signal'] != 0].index.to_series().diff().dropna()

avg_interval_seconds = time_diffs.dt.total_seconds().mean()
avg_interval = avg_interval_seconds/86400 # 转换为天数

numDays = avg_interval

ptSl=[0,2]
std = getDailyVol(df['close'], span0=100)
std = pd.DataFrame(std).rename(columns={'close': 'daily_vol'})
# 将结果合并回原DataFrame
df = df.join(std)

t1 = df['close'].index.searchsorted(df['close'].index + pd.Timedelta(days=numDays))
t1 = t1[t1 < df.shape[0]]  # 确保不超出范围
t1 = pd.Series(df['close'].index[t1], index=df['close'].index[:t1.shape[0]])
t1.name = 't1'
# 将t1列添加到DataFrame
df = df.join(t1)
df_events = df[df['signal'] != 0]
df_events.head()

df_events_with_labels = generate_metalabels(df_events, df['close'], df_events['daily_vol'], ptSl)

#b 使用随机森林模型去验证信号效果。貌似训练题里面提到的特征是不包含主模型signal的。
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def prepare_features(close, window_short=5, window_long=60, corr_window=20):
    """
    扩展特征工程函数
    :param close: 收盘价序列
    :param window_short: 短期移动平均窗口
    :param window_long: 长期移动平均窗口
    :param corr_window: 序列相关性计算窗口
    :return: 包含新特征的DataFrame
    """
    # 基础特征
    features = pd.DataFrame(index=close.index)
    features['returns'] = close.pct_change()
    
    # 1. 序列相关性特征
    features['autocorr_1'] = features['returns'].rolling(corr_window).apply(
        lambda x: x.autocorr(lag=1), raw=False)
    features['autocorr_5'] = features['returns'].rolling(corr_window).apply(
        lambda x: x.autocorr(lag=5), raw=False)
    
    # 2. 移动平均特征
    features['ma_short'] = close.rolling(window_short).mean()
    features['ma_long'] = close.rolling(window_long).mean()
    
    # 3. 移动平均比率
    features['ma_ratio'] = features['ma_short'] / features['ma_long']
    
    # 4. 价格与移动平均偏离度
    features['dev_short'] = close / features['ma_short'] - 1
    features['dev_long'] = close / features['ma_long'] - 1
    
    # 5. 移动平均交叉信号
    features['ma_cross'] = np.where(features['ma_short'] > features['ma_long'], 1, -1)
    
    return features.dropna()

# 使用示例
features = prepare_features(df['close'])
# display(features.head())
merged_df = df_events_with_labels.join(features, how='inner')


# 准备特征和目标变量
X = merged_df[['close', 'rolling_mean', 'upper_band', 'lower_band', 'signal',
       'daily_vol', 'returns', 'autocorr_1', 'autocorr_5',
       'ma_short', 'ma_long', 'ma_ratio', 'dev_short', 'dev_long', 'ma_cross']]
y = merged_df['metallabel']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train)


print(classification_report(y_test, rf.predict(X_test)))

df_x_test = X_test.copy()

df_x_test['correct_direction'] = y_test  # 实际正确方向
df_x_test['metallabel'] = merged_df.loc[y_test.index, 'metallabel']  # 原始metallabel

# 3. 添加预测结果
df_x_test['predicted_direction'] = rf.predict(X_test)  # 预测方向

# 4. 计算预测准确率
accuracy = (df_x_test['predicted_direction'] == df_x_test['correct_direction']).mean()
print(f"模型预测准确率: {accuracy:.2%}")

# 5. 可选：添加预测概率
df_x_test['probability'] = rf.predict_proba(X_test)[:, 1]  # 预测为1的概率

# 6. 查看结果
display(df_x_test.head())  

df_x_test[['signal','metallabel','predicted_direction','probability']]


#c  主模型的准确率，精确率，召回率和f1score，即没有元模型进行过滤的话

print("主模型评估报告:")
print(classification_report(y_test, X_test['signal']))



#d  加入元模型后的准确率，精确率，召回率和f1score
print("加入元模型评估报告:")
print(classification_report(y_test, df_x_test['predicted_direction']))

# 结论：准确率，精确率，召回率和f1score这些指标都有回升，提高了信号的可信度

#画图
# 1. 获取X_test时间段内的收盘价
start_date = df_x_test.index.min()
end_date = df_x_test.index.max()
closes_in_period = close.loc[start_date:end_date]

# 2. 创建图表
plt.figure(figsize=(14, 7))

# 绘制收盘价
plt.plot(closes_in_period.index, closes_in_period, 
        label='Close Price', color='black', linewidth=1.5)

# 3. 标注原始signal信号
signal_mask = df_x_test['signal'] != 0
buy_signals = df_x_test[(df_x_test['signal'] == 1) & signal_mask]
sell_signals = df_x_test[(df_x_test['signal'] == -1) & signal_mask]

plt.scatter(buy_signals.index, 
           closes_in_period.loc[buy_signals.index],
           label='Original Buy Signal', 
           color='blue', marker='^', s=100)
plt.scatter(sell_signals.index, 
           closes_in_period.loc[sell_signals.index],
           label='Original Sell Signal', 
           color='red', marker='v', s=100)

# 4. 标注预测predicted_direction信号
pred_buy = df_x_test[df_x_test['predicted_direction'] == 1]
pred_sell = df_x_test[df_x_test['predicted_direction'] == -1]

plt.scatter(pred_buy.index, 
           closes_in_period.loc[pred_buy.index],
           label='Predicted Buy', 
           color='green', marker='*', s=150, alpha=0.7)
plt.scatter(pred_sell.index, 
           closes_in_period.loc[pred_sell.index],
           label='Predicted Sell', 
           color='orange', marker='*', s=150, alpha=0.7)

# 5. 添加图表元素
plt.title(f'Price and Signals from {start_date.date()} to {end_date.date()}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. 显示图表
plt.show()
# 结论：可以看到元模型过滤后的信号点明显减少，且大部分集中在价格的极端位置，正确率比较高，而且收益极为恐怖
#3.5好像还忘记过滤最小收益率了，可以选择在信号side生成后用CUSUM filter过滤掉收益较低的信号。效果可能会更好

#改进方向：
#First, we build a model that achieve high recall, even if precision is not particularly high.
#Second correct for low precision by applying meta-label to the positives predicted by the primary model."
#Advances in Financial Machine Learning, page 52
# 也就是先让模型尽可能多的预测出正类（高召回率），然后再用元模型去过滤掉错误的正类（高精准率），从而提升整体的精确率



#%%
#4.样本权重
#这章比较偏数学，看的我云里雾里的

'''
4.1 In Chapter 3, we denoted as t1 a pandas series of timestamps where the first
barrier was touched, and the index was the timestamp of the observation. This
was the output of the getEvents function.
(a) Compute a t1 series on dollar bars derived from E-mini S&P 500 futures
tick data.
(b) Apply the function mpNumCoEvents to compute the number of overlapping
outcomes at each point in time.
(c) Plot the time series of the number of concurrent labels on the primary axis,
and the time series of exponentially weighted moving standard deviation of
returns on the secondary axis.
(d) Produce a scatterplot of the number of concurrent labels (x-axis) and the
exponentially weighted moving standard deviation of returns (y-axis). Can
you appreciate a relationship

'''

#a 获取事件起止时间 t1
dollar = pd.read_csv(r'D:\Git\book\ASML\dollar_bars.csv'   ,
                     parse_dates=True,      # 解析日期列
                     index_col=[0]  # 将 'date_time' 列作为索引
                     )
close = dd_bars(data = dollar.close, m = 100000) #dollar bar的series


short_ma = close.rolling(window=5).mean()
long_ma = close.rolling(window=30).mean()

# 创建信号DataFrame
df = pd.DataFrame(index=close.index)
df['close'] = close
df['short_ma'] = short_ma
df['long_ma'] = long_ma
# 生成交易信号
# 金叉: 短均线上穿长均线 (买入信号)
# 死叉: 短均线下穿长均线 (卖出信号)
df['signal'] = 0  # 0表示无信号
# 计算金叉和死叉
golden_cross = (df['short_ma'] > df['long_ma']) & (df['short_ma'].shift(1) <= df['long_ma'].shift(1))
death_cross = (df['short_ma'] < df['long_ma']) & (df['short_ma'].shift(1) >= df['long_ma'].shift(1))
# 标记信号
df.loc[golden_cross, 'signal'] = 1  # 买入信号
df.loc[death_cross, 'signal'] = -1  # 卖出信号
# 提取所有交易信号的时间点
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]

print("买入信号发生时间:")
print(len(buy_signals.index))
print("\n卖出信号发生时间:")
print(len(sell_signals.index))

# 计算相邻时间点的时间差
time_diffs = df[df['signal'] != 0].index.to_series().diff().dropna()

avg_interval_seconds = time_diffs.dt.total_seconds().mean() *1.5
avg_interval = avg_interval_seconds/86400 # 转换为天数

numDays = avg_interval
# 
ptSl=[1,1]
std = getDailyVol(df['close'], span0=100)
std = pd.DataFrame(std).rename(columns={'close': 'daily_vol'})
# 将结果合并回原DataFrame
df = df.join(std)

t1 = df['close'].index.searchsorted(df['close'].index + pd.Timedelta(days=numDays))
t1 = t1[t1 < df.shape[0]]  # 确保不超出范围
t1 = pd.Series(df['close'].index[t1], index=df['close'].index[:t1.shape[0]])
t1.name = 't1'
# 将t1列添加到DataFrame
df = df.join(t1)
df_events = df[df['signal'] != 0]
df_events_with_labels = generate_metalabels(df_events, df['close'], df_events['daily_vol'], ptSl)
# df_events_with_labels.head()

#b 计算每个时间点（bar）的重叠事件数量  需要在a中生成重叠的事件
def mpNumCoEvents(event,close):
    close=close[event.index.min():event.index.max()] # align
    count=pd.Series(0,index=close.index)
    for loc,row in event.iterrows():
        count.loc[loc:row['t1']]+=1 # count events
    return count
event_count_cloes=mpNumCoEvents(df_events_with_labels,close)
event_count_cloes.head()


#c 绘制重叠事件数量与收益波动率的时间序列图
# 计算收益率和指数加权移动标准差
returns = close.pct_change()  # 收益率
ewm_std = returns.ewm(span=20).std()  # 20天半衰期的指数加权移动标准差
# 确保时间索引对齐
ewm_std = ewm_std[event_count_cloes.index.min():event_count_cloes.index.max()] 

# 创建画布和主坐标轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制主坐标轴（并发标签数量）
ax1.plot(event_count_cloes, color='blue', label='Concurrent Labels')
ax1.set_xlabel('Time')
ax1.set_ylabel('Number of Concurrent Labels', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 创建次坐标轴并绘制波动率
ax2 = ax1.twinx()
ax2.plot(ewm_std, color='red', label='EWM Std of Returns')
ax2.set_ylabel('EWM Std of Returns', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 添加图例和标题
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Concurrent Labels vs. Returns Volatility')
plt.show()


#d 画散点图，以并发数量为x轴，收益波动率为y轴
plt.figure(figsize=(10, 6))
plt.scatter(event_count_cloes, ewm_std, alpha=0.5)  # alpha设置透明度
plt.xlabel('Number of Concurrent Labels (event_count_cloes)')
plt.ylabel('EWM Std of Returns')
plt.title('Scatter Plot: Concurrent Labels vs Returns Volatility')
plt.grid(True)  # 添加网格线
plt.show()

'''
4.2 Using the function mpSampleTW, compute the average uniqueness of each label.
What is the first-order serial correlation, AR(1), of this time series? Is it statistically significant? Why?

'''
#计算每个标签的平均唯一性。平均唯一性是指在某个时间点上，所有覆盖该时间点的事件的唯一性加权的平均值。大于0，小于1.
# mpSampleTW这里只计算了每个事件的平均唯一性加权。
#上述得到了每个时间的唯一值uniqueness，是事件级别的，而不是bar级别的。所以对于每个bar， 将覆盖该bar的所有事件的uniqueness取平均值 ，就得到了该bar 的唯一值
def mpSampleTW(event,close):
    '''
    #计算每个事件的唯一性加权，返回每个事件的权重
    '''
    close=close[event.index.min():event.index.max()] # align
    count=pd.Series(0,index=close.index)
    for loc,row in event.iterrows():
        count.loc[loc:row['t1']]+=1 # count events
    
    # 计算每个事件的权重（持续期间的平均并发事件倒数）
    wght = pd.Series(index=event.index)
    for tIn, tOut in event['t1'].items():
        wght.loc[tIn] = (1./count.loc[tIn:tOut]).mean()
    
    return wght

uniqueness = mpSampleTW(df_events_with_labels, close)
uniqueness.head()
def calculate_bar_uniqueness(events, uniqueness, bar_timestamps):
    """
    计算每个bar的唯一值（覆盖该bar的所有事件的uniqueness平均值）
    
    参数:
        events: 事件DataFrame，包含't1'列（事件结束时间）
        uniqueness: Series，事件级别的唯一值，索引为事件开始时间
        bar_timestamps: bar的时间戳列表/Index
    """
    bar_uniqueness = pd.Series(index=bar_timestamps, dtype=float)
    
    for bar_time in bar_timestamps:
        # 找出覆盖当前bar的所有事件（事件开始时间 <= bar_time <= 事件结束时间）
        overlapping_events = events[(events.index <= bar_time) & (events['t1'] >= bar_time)]
        
        if not overlapping_events.empty:
            # 计算这些事件的uniqueness平均值
            bar_uniqueness[bar_time] = uniqueness[overlapping_events.index].mean()
        else:
            bar_uniqueness[bar_time] = 1  # 无覆盖事件时设为1
    
    return bar_uniqueness
# 计算每个标签的平均唯一性

bar_uniqueness = calculate_bar_uniqueness(df_events_with_labels, uniqueness, close.index)


# 计算AR(1)
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# 对bar_uniqueness进行单位根测试 
#平稳说明序列的统计特性（如均值和方差）不会随着时间发生变化
result = adfuller(bar_uniqueness.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1]) ## p<0.05则平稳
print('Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value}')


'''
4.3 Fit a random forest to a financial dataset where  (∑I i=1 ūi)I ≪ 1. #I是事件数量，ūi是事件的平均唯一性
(a) What is the mean out-of-bag accuracy?
(b) What is the mean accuracy of k-fold cross-validation (without shuffling) on
the same dataset?
(c) Why is out-of-bag accuracy so much higher than cross-validation accuracy?
Which one is more correct / less biased? What is the source of this bias?
'''

#a 当前使用的数据集就满足 (∑I i=1 ūi)I ≪ 1，因为所以的bar平均唯一性都是大于0小于1的
#袋外精确度均值 ：在训练中没有选择该数据点的所有决策树对该数据点的预测准确度，所有数据点的袋外准确度的平均值就是袋外精确度均值
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

#延续使用4.1的数据集
# 准备特征和目标变量 'metallabel'作为目标值
X = df_events_with_labels[['close', 'short_ma', 'long_ma', 'signal', 'daily_vol']]
y = df_events_with_labels['metallabel']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
# 初始化随机森林分类器，启用袋外评分
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
# 训练模型
rf.fit(X_train, y_train)
# 打印袋外精确度均值
print("袋外精确度均值:", rf.oob_score_)  #0.528344671201814

#b k折交叉验证精确度均值 ：在k折交叉验证中，每次将数据集分为k个子集，每次使用k-1个子集训练模型，用剩下的子集验证模型，最后取k次验证结果的平均值作为模型的精确度均值。
# 初始化随机森林分类器，启用k折交叉验证
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 设置k-fold交叉验证（不洗牌）
k = 5  # 可以根据需要调整折数
kf = KFold(n_splits=k, shuffle=False, random_state=None)

# 计算交叉验证准确率
cv_scores = cross_val_score(rf_classifier, X, y, 
                           cv=kf, scoring='accuracy', n_jobs=-1)

# 输出结果
print(f"K-fold交叉验证准确率 (k={k}): {cv_scores}")
print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})") #0.4784

#c 袋外精确度均值比 k折交叉验证精确度均值高。
#Kfold 只是将数据拆成样本，不会替换或重新选择他们。而随机森林oob际上选择和替换样本（Bootstrap = True默认），当数据集的唯一性不高时，由于并发事件，这将使OOB分数与in-bag样本非常相同，并且彼此冗余。
#参见《金融机器学习进展》，第62 - 63页，第4.5节。


'''
4.4 Modify the code in Section 4.7 to apply an exponential time-decay factor.
'''
#除了对平均唯一性的应用外，样本效果随时间衰减也是重要的应用
def getTimeDecay(tW,clfLastW=1.):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW=tW.sort_index().cumsum()
    if clfLastW>=0:
        slope=(1.-clfLastW)/clfW.iloc[-1]
    else:
        slope=1./((clfLastW+1)*clfW.iloc[-1])
    const=1.-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    print (const,slope)
    return clfW

bar_uniqueness_decay1=getTimeDecay(bar_uniqueness, clfLastW=1)
bar_uniqueness_decay2=getTimeDecay(bar_uniqueness, clfLastW=0.6)
bar_uniqueness_decay3=getTimeDecay(bar_uniqueness, clfLastW=0.3)
bar_uniqueness_decay4=getTimeDecay(bar_uniqueness, clfLastW=0)
bar_uniqueness_decay5=getTimeDecay(bar_uniqueness, clfLastW=-0.5)
bar_uniqueness_decay6=getTimeDecay(bar_uniqueness, clfLastW=-0.9)
#画图
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
# plt.plot(bar_uniqueness.index, bar_uniqueness, label='Original')
plt.plot(bar_uniqueness.index, bar_uniqueness_decay1, label='Decay 1')
plt.plot(bar_uniqueness.index, bar_uniqueness_decay2, label='Decay 0.6')
plt.plot(bar_uniqueness.index, bar_uniqueness_decay3, label='Decay 0.3')
plt.plot(bar_uniqueness.index, bar_uniqueness_decay4, label='Decay 0')
plt.plot(bar_uniqueness.index, bar_uniqueness_decay5, label='Decay -0.5')
plt.plot(bar_uniqueness.index, bar_uniqueness_decay6, label='Decay -0.9')
plt.legend()
plt.title('Bar Uniqueness with Different Decay Factors')
plt.xlabel('Date')
plt.ylabel('Uniqueness')
plt.show()


'''
4.5 Consider you have applied meta-labels to events determined by a trend-following
model. Suppose that two thirds of the labels are 0 and one third of the labels
are 1.
(a) What happens if you fit a classifier without balancing class weights?
(b) A label 1 means a true positive, and a label 0 means a false positive. By
applying balanced class weights, we are forcing the classifier to pay more
attention to the true positives, and less attention to the false positives. Why
does that make sense?
(c) What is the distribution of the predicted labels, before and after applying
balanced class weights?
'''


#a 不平衡类权重的分类器会将更多的权重分配给多数类（0），从而导致模型对少数类（1）的预测能力下降。

#b 通过应用平衡类权重，模型会更加关注少数类（1），从而提高对正类的识别能力。
#而真正类真是能够带来盈利的点，假整正类并不会提高盈利水平，只是提高了模型综合得分。
#所以在metalabel的识别要既要将焦点聚焦到真正类的提高就够了。
#应该如何平衡类权重呢？ 1.class_weight='balanced'（或者'balance_subsample'）,  # 自动平衡权重 设置class_weight就是要求ML模型更加关注少数类
# #2.class_weight={0:1, 1:3}  # 手动设置权重 3.对样本进行重采样

#c  分布会更加均衡，少数类（1）的比例会增加，从而使模型在预测时更有可能输出1。


'''
4.6 Update the draw probabilities for the final draw in Section 4.5.3
4.7 In Section 4.5.3, suppose that number 2 is picked again in the second draw. What
would be the updated probabilities for the third draw
'''
#纯数学，概率再平衡的顺序抽样（sequential bootstrap） 做不出，先跳过

'''
第四章总结：
1.本章数学含量极大，终于知道为什么做量化需要找清北数学系的了。 但是没关系，不需要做到那么极限的数学也足够了
2.这章主要是关注了样本权重的问题。一是样本的唯一性（事件使用的bar重叠），二是样本的时间衰减性，三是样本（事件）类别不平衡性


'''


#%%
#第五章 Fractionally Differentiated Features 分数阶微分特征 即分数级差分
'''
#金融数据短期内由于套利等高频操作使得信噪比很低，价格序列通常是非平稳的，而且通常具有记忆性。相比之下，整数差分后的序列，如收益率，其记忆是有限的，也就是说，历史数据在有限样本窗口之外将被完全忽略
#一旦平稳性变换抹去了数据中的所有记忆，就会求助于复杂的数学指标来提取信号
#而分数阶查分在确保数据平稳的同时尽可能保留记忆。

两个特性很重要：平稳性与记忆性
1.平稳性指数据的统计特性（如均值和方差）不会随着时间发生变化。也就是不管起点在哪都具备类似的统计特性。
    通过数据转变是能够使数据变为平稳序列的，比如对数，差分等，但是这些操作很容易就将记忆性抹除了
2.记忆性指数据中存在长期依赖关系，过去的事件会影响未来的事件。记忆性能够帮助我们捕捉数据中的模式和趋势，从而提高预测的准确性。
    一般来说收益是平稳的，但没有记忆；价格是有记忆的，但不平稳。


分数差分会将滞后阶的数据系数赋值为非0 ，所以具有记忆性。当为整数阶差分的时候，后面的滞后项的系数会置0.

本章精髓：
确定最小的 d（分数差分阶），使得 FFD(d) 上 ADF 统计量的 p 值低于 5%，即95%显著平稳。
然后，使用 FFD(d) 序列作为预测特征。
'''


'''
5.1 Generate a time series from an IID Gaussian random process. This is a memoryless, stationary series:
(a) Compute the ADF statistic on this series. What is the p-value?
(b) Compute the cumulative sum of the observations. This is a non-stationary
series without memory.
(i) What is the order of integration of this cumulative series?
(ii) Compute the ADF statistic on this series. What is the p-value?
(c) Differentiate the series twice. What is the p-value of this over-differentiated
series?

'''

#a 生成一个独立同分布的高斯随机过程时间序列，并计算其ADF统计量和p值
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
def create_price_data51(start_price: float = 1000.00,
                      mu: float = .0,
                      var: float = 1.0,
                      n_samples: int = 1000):
                      
    i = np.random.normal(mu, var, n_samples)
    df0 = pd.date_range(periods=n_samples,
                        freq=pd.tseries.offsets.Minute(),
                        end=pd.Timestamp.now())
                        
    X = pd.Series(i, index=df0, name = "close").to_frame()
    # X.close.iat[0] = start_price
    return X
data51 = create_price_data51()
# 计算ADF统计量和p值
adf_result = adfuller(data51)
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])


#b 计算观察值的累积和，并分析其平稳性
cumsum_series = data51['close'].cumsum()
# 计算ADF统计量和p值
adf_result = adfuller(cumsum_series)
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])

#变成了非平稳了。data51相当于日收盘价的差，而累计和类似于每日收盘价。每日收盘价是非平稳的，所以必须要进行处理

#b1 该累积序列的积分阶数为1，因为它是通过对原始序列进行一次累积得到的。
#积分阶指使一个非平稳过程变为平稳所需的累积（或差分）的次数，所以这里是1

#c 对原生成序列进行两次差分，并计算其ADF统计量和p值
diff_series = data51.diff().diff().dropna()
# 计算ADF统计量和p值
adf_result = adfuller(diff_series)
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])

#仍旧是平稳的，差分没有改变平稳性。


'''
5.2 Generate a time series that follows a sinusoidal function. This is a stationary
series with memory.
(a) Compute the ADF statistic on this series. What is the p-value?
(b) Shift every observation by the same positive value. Compute the cumulative
sum of the observations. This is a non-stationary series with memory.
(i) Compute the ADF statistic on this series. What is the p-value?
(ii) Apply an expanding window fracdiff, with 𝜏 = 1E − 2. For what minimum d value do you get a p-value below 5%?
(iii) Apply FFD, with 𝜏 = 1E − 5. For what minimum d value do you get a p-value below 5%?

'''

#a 生成一个遵循正弦函数的时间序列，并计算其ADF统计量和p值
def create_sinusoidal_data52(amplitude: float = 1.0,
                             frequency: float = 1.0,
                             phase: float = 0.0,
                             n_samples: int = 50000,
                             n_periods: int = 5):  # n_periods 用来表示周期数
    # 计算时间 t，跨越多个周期
    t = np.linspace(0, 2 * np.pi * n_periods, n_samples)
    
    # 生成正弦波数据
    X = amplitude * np.sin(frequency * t + phase)
    
    # 生成日期时间索引
    df0 = pd.date_range(periods=n_samples,
                        freq=pd.tseries.offsets.Minute(),
                        end=pd.Timestamp.now())
    
    # 将正弦波数据转换为 pandas DataFrame
    X = pd.Series(X, index=df0, name="close").to_frame()
    
    return X

# 生成包含多个周期的正弦波数据
data52 = create_sinusoidal_data52(n_periods=100)  

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(data52.index, data52['close'], label='Sinusoidal Wave', color='b')
plt.title('Sinusoidal Wave with Multiple Periods')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
# 计算ADF统计量和p值
adf_result = adfuller(data52)
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])

#b 将每个观察值平移一个正值，并计算其累积和的ADF统计量和p值
shifted_series = data52 + 3  # 平移一个正值
shifted_series = shifted_series['close'].cumsum()
# 计算ADF统计量和p值
adf_result = adfuller(shifted_series)
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])
#经过转换后数据变成了非平稳的。假设价格是在某个均价上下波动，而且具有记忆性。但是合成的累计和价格序列是非平稳的

#b2 应用扩展窗口分数差分，找到使p值低于5%的最小d值，丢弃阈值0.01
#只需要找到显著平稳的最小d值即可，d越大，越接近于1，记忆性是越差的。在【0,1】这个范围内
def getWeights(d,size):
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_=-w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1)
    return w
def fracDiff_ew(series,d,thres=.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeights(d,series.shape[0])
    #2) Determine initial calcs to be skipped based on weight-loss threshold
    w_=np.cumsum(abs(w))
    w_/=w_[-1]
    skip=w_[w_>thres].shape[0]
    #3) Apply weights to values
    df={}
    series=pd.DataFrame(series)
    for name in series.columns:
        seriesF,df_=series[[name]].ffill().dropna(),pd.Series()  
        for iloc in range(skip,seriesF.shape[0]):
            loc=seriesF.index[iloc]
            if not np.isfinite(series.loc[loc,name]):continue # exclude NAs
            df_[loc]=np.dot(w[-(iloc+1):,:].T,seriesF.loc[:loc])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

expending_windows=fracDiff_ew(shifted_series,d=0.1,thres=1E-2)
# 计算ADF统计量和p值
adf_result = adfuller(expending_windows)
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])
#以0.01为间隔，逐步减少d值，直到p值低于0.05
d_values = np.arange(0, 0.1, 0.005)
for d in d_values:
    expending_windows=fracDiff_ew(shifted_series,d=d,thres=1E-2)
    # 计算ADF统计量和p值
    adf_result = adfuller(expending_windows)
    print(f'd={d:.3f}, ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}')
    if adf_result[1] <= 0.05:
        print(f'最小d值: {d:.3f}')
        break
    #一个周期时：
#最小d值: 0-0.5区间0.1就是平稳了，但是expending_windows只有40%的数据了。在0.5-1区间，0.69是最小平稳值。数据样本还有95%
#不同的阈值thres也影响最小分数阶d值
#100个周期时：
#最小d值0.005，直接出

#b31 书中代码
def getWeights_FFD(d,thres):
    w,k=[1.],1
    while True:
        w_=-w[-1]/k*(d-k+1)
        if abs(w_)<thres:break
        w.append(w_);k+=1
    return np.array(w[::-1]).reshape(-1,1)
def fracDiff_FFD(series,d,thres=1e-5):
    # Constant width window (new solution)
    w=getWeights_FFD(d,thres)
    width=len(w)-1
    df={}
    series=pd.DataFrame(series)
    for name in series.columns:
        seriesF,df_=series[[name]].ffill().dropna(),pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df,width

ffd_book,width=fracDiff_FFD(shifted_series,d=0.8,thres=1E-5)
print(f'窗口宽度: {width}')
# 计算ADF统计量和p值
adf_result = adfuller(ffd_book)
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])
#以0.01为间隔，逐步减少d值，直到p值低于0.05
d_values = np.arange(0.995, 1, 0.00002)
for d in d_values:
    ffd_book,width=fracDiff_FFD(shifted_series,d=d,thres=1E-5)
    # 计算ADF统计量和p值
    adf_result = adfuller(ffd_book)
    print(f'd={d:.5f}, ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}, 窗口宽度: {width}')
    if adf_result[1] <= 0.05:
        print(f'最小d值: {d:.5f}')
        break

#1个周期时：
#以1E-4的阈值都难以在【0,1】区间找到平稳的，最终在d=1时达到平稳。减少精度为1e-2也找不到合适的d值。只是接近1的时候就是平稳的。0.9999999995-0.99999999995之间才达到平稳，非常麻烦
#阈值是为了控制系数的权重阈值，越小的阈值能够截取更长的系数列，精度更高，即窗口更大。窗口指每个x与后面width长度的值产生递推关系
#书本里面的窗口长度是自适应阈值的，这样要处理的系数就少了一个。即分数阶d，精度阈值thres，和窗口长度width只需要确定两个就行了。
#所以最后还是使用Fracdiff这个库更快更方便。
#100个周期时：
#还是难以找到合适的d值

#b32
#Fracdiff这个库是专门计算分数阶差分的库，是专门为这本书开发的一个库，甚至支持了5%下显著的平稳性最大记忆获取。即最小d值获取  https://github.com/fracdiff/fracdiff
#但是这个库最高只支持python3.9 
#使用miniforge装了环境，需要时切换到该环境装包使用。
#conda activate py39 激活 然后conda install fracdiff
#Fracdiff 这个库只实现了固定窗口法。以后使用都用这个，但是联系还得写拓展窗口法练习。
#mode必须选择'valid'才行，same模式是填充的，仅适用于可视化画图
#window是确定精度，即与后面多少位产生递推关系。precision是输出的分数阶精度
#使用fracdiff对单次求微分，使用FracdiffStat寻找显著水平下的最小分数阶，只需要确定window这个参数即可
from fracdiff.sklearn import FracdiffStat,fracdiff
series_2d = shifted_series.to_numpy().reshape(-1, 1)
ffd = FracdiffStat(window=5000 , precision=1e-6,lower=0,upper=1.0, pvalue=0.05,mode = 'valid')  
y=ffd.fit_transform(series_2d)
# 计算ADF统计量和p值
adf_result = adfuller(y)
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])
print(f"最小d值: {ffd.d_[0]:.9f}")  
#最小d值0.999999958，直接得到最后的结果。显著的比课本里的代码快

#一开始生成的数据只有正弦函数一个周期，后续改为100个周期，数据结构不一样了，结论也不同
#100个周期： 只能找到d=1了。将窗口扩大也一样。

'''
5.3 Take the series from exercise 2.b:
(a) Fit the series to a sine function. What is the R-squared?
(b) Apply FFD(d=1). Fit the series to a sine function. What is the R-squared?
(c) What value of d maximizes the R-squared of a sinusoidal fit on FFD(d).
Why?

'''


#5.3a 拟合为正弦函数并计算r方
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def fit_sine_y_only(y, sampling_rate=1):
    """
    只有y数据时的正弦函数拟合
    
    参数:
    y: y值数据数组
    sampling_rate: 采样率，默认为1（假设等间距采样）
    
    返回:
    popt: 最优参数 [A, f, phi, C]
    r_squared: R²值
    y_pred: 预测值
    x: 生成的x坐标
    """
    
    # 假设x是等间距的索引
    x = np.arange(len(y)) / sampling_rate
    
    # 正弦函数模型
    def sin_func(x, A, f, phi, C):
        return A * np.sin(2 * np.pi * f * x + phi) + C
    
    # 提供初始参数估计
    # 振幅估计
    A0 = (np.max(y) - np.min(y)) / 2
    if A0 == 0:
        A0 = 1  # 避免除零
    
    # 频率估计（使用FFT找到主要频率）
    fft = np.fft.fft(y - np.mean(y))  # 去均值
    freqs = np.fft.fftfreq(len(y), 1/sampling_rate)
    
    # 找到正频率部分的最大振幅对应的频率
    positive_freq_idx = np.where(freqs > 0)[0]
    if len(positive_freq_idx) > 0:
        idx = positive_freq_idx[np.argmax(np.abs(fft[positive_freq_idx]))]
        f0 = freqs[idx]
    else:
        f0 = 1/len(y)  # 默认频率
    
    # 相位和偏移估计
    phi0 = 0
    C0 = np.mean(y)
    
    initial_guess = [A0, f0, phi0, C0]
    
    try:
        # 设置参数边界以避免不合理的值
        bounds = ([0, 0, -np.pi, -np.inf], 
                 [2*A0, sampling_rate/2, np.pi, np.inf])
        
        # 使用curve_fit进行非线性最小二乘拟合
        popt, pcov = curve_fit(sin_func, x, y, p0=initial_guess, 
                              bounds=bounds, maxfev=5000)
        
        # 计算预测值
        y_pred = sin_func(x, *popt)
        
        # 计算R²
        r_squared = r2_score(y, y_pred)
        
        return popt, r_squared, y_pred, x
        
    except Exception as e:
        print(f"拟合过程中出现错误: {e}")
        # 尝试不使用边界
        try:
            popt, pcov = curve_fit(sin_func, x, y, p0=initial_guess, maxfev=5000)
            y_pred = sin_func(x, *popt)
            r_squared = r2_score(y, y_pred)
            return popt, r_squared, y_pred, x
        except:
            return None, None, None, x


# 如果您有自己的y数据，请使用这个函数
def fit_your_data(y_data, sampling_rate=1):
    """
    对您的y数据进行正弦拟合
    
    参数:
    y_data: 您的y数据数组
    sampling_rate: 采样率（如果知道的话），默认为1
    """
    popt, r_squared, y_pred, x = fit_sine_y_only(y_data, sampling_rate)
    
    if popt is not None:
        A, f, phi, C = popt
        
        print("=" * 50)
        print("您的数据拟合结果")
        print("=" * 50)
        print(f"振幅 (A): {A:.4f}")
        print(f"频率 (f): {f:.4f} Hz")
        print(f"相位 (φ): {phi:.4f} rad")
        print(f"偏移 (C): {C:.4f}")
        print(f"R²: {r_squared:.4f}")
        print("=" * 50)
        print(f"拟合方程: y = {A:.4f} * sin(2π*{f:.4f}*t + {phi:.4f}) + {C:.4f}")
        
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(x, y_data, 'bo', alpha=0.7, label='您的数据', markersize=4)
        plt.plot(x, y_pred, 'r-', label='拟合曲线', linewidth=2)
        plt.xlabel('样本索引' if sampling_rate == 1 else '时间')
        plt.ylabel('y')
        plt.legend()
        plt.title('您的数据正弦拟合结果')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return popt, r_squared, y_pred, x
    else:
        print("拟合失败，请检查数据")
        return None, None, None, None

y=y.flatten()
fit_your_data(y, sampling_rate=1) 


#5.3b 应用FFD(d=1)并拟合为正弦函数
ffd_book,width=fracDiff_FFD(shifted_series,d=1,thres=1E-2)
ffd_book=ffd_book['close']
fit_your_data(ffd_book, sampling_rate=1) 

'''
d=1
振幅 (A): 1.0000
频率 (f): 0.0000 Hz
相位 (φ): 0.0001 rad
偏移 (C): 3.0000
R²: 1.0000

d=0.9999999995
振幅 (A): 1.0000
频率 (f): 0.0000 Hz
相位 (φ): 0.0124 rad
偏移 (C): 3.0000
R²: 1.0000

整体差别不大
'''

'''
5.4 Take the dollar bar series on E-mini S&P 500 futures. Using the code
in Snippet 5.3, for some d ∈ [0, 2], compute fracDiff_FFD(fracDiff
_FFD(series,d),-d). What do you get? Why?
'''
from fracdiff import fdiff
dollar = pd.read_csv(r'D:\Git\book\ASML\dollar_bars.csv'   ,
                     parse_dates=True,      # 解析日期列
                     index_col=[0]  # 将 'date_time' 列作为索引
                     )
tb = dd_bars(data = dollar.close, m = 100000)
d = 0.2
window = 100

d0 = fdiff(tb.values, d, window=window, mode="same")
d1 = fdiff(d0, -d, window=window, mode="same")
spx=tb
spxd = pd.Series(d0, index=spx.index)
spxi = pd.Series(d1, index=spx.index)

plt.figure(figsize=(24, 6))

plt.subplot(1, 3, 1)
plt.title("原始")
plt.plot(spx, linewidth=0.6)

plt.subplot(1, 3, 2)
plt.title("d^{} 原始".format(d))
plt.plot(spxd, linewidth=0.6)

plt.subplot(1, 3, 3)
plt.title("d^{} d^{} 原始".format(-d, d))
plt.plot(spxi, linewidth=0.6)

plt.show()

#不能百分百还原，因为窗口截断是有误差的

'''
5.5 Take the dollar bar series on E-mini S&P 500 futures.
(a) Form a new series as a cumulative sum of log-prices.
(b) Apply FFD, with 𝜏 = 1E − 5. Determine for what minimum d ∈ [0, 2] the
new series is stationary.
(c) Compute the correlation of the fracdiff series to the original (untransformed)
series.
(d) Apply an Engel-Granger cointegration test on the original and fracdiff series.
Are they cointegrated? Why?
(e) Apply a Jarque-Bera normality test on the fracdiff series.

'''
#5.5a 
log_prices = np.log(tb)
log_prices_cumsum = log_prices.cumsum()

#5.5b
#根据分数阶和阈值确定窗口大小
from fracdiff.sklearn.tol import window_from_tol_coef
window = window_from_tol_coef(0.5, 1e-5)
print('合适窗口大小:',window)

X_log_cumsum = np.array(log_prices).reshape(-1, 1)
f = FracdiffStat(window=window, mode="valid", upper=2)
diff = f.fit_transform(X_log_cumsum)

print("* Order: {:.2f}".format(f.d_[0]))
adf_result = adfuller(diff)
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])

#5.5c
#计算fracdiff系列与原始系列的相关性
corr = np.corrcoef(diff.flatten(), X_log_cumsum[window-1:].flatten())[0, 1]
print("* Correlation: {:.4f}".format(corr))
#画图 diff需要向右移动window-1个单位，而且diff使用右轴使得图形差不多重叠
# 创建图形和坐标轴
fig, ax1 = plt.subplots(figsize=(12, 6))
# 确保数据格式适合画图
if hasattr(X_log_cumsum, 'flatten'):
    X_flat = X_log_cumsum.flatten()
else:
    X_flat = X_log_cumsum
if hasattr(diff, 'flatten'):
    diff_flat = diff.flatten()
else:
    diff_flat = diff
# 向右移动diff数据window-1个单位
shifted_diff = np.full_like(X_flat, np.nan)
shifted_diff[window-1:] = diff_flat
# 在左轴上绘制原始序列
ax1.plot(X_flat, label='原始对数累计序列', color='blue')
ax1.set_xlabel('时间')
ax1.set_ylabel('原始序列值', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
# 创建右轴
ax2 = ax1.twinx()
# 在右轴上绘制移动后的diff
ax2.plot(shifted_diff, label='分数差分序列 (shifted)', color='red', alpha=0.7)
ax2.set_ylabel('分数差分序列值', color='red')
ax2.tick_params(axis='y', labelcolor='red')
# 设置标题和图例
plt.title(f'原始对数累计序列与分数差分序列对比 (window={window}, correlation={corr:.4f})')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
# 显示网格线
ax1.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# 显示图形
plt.show()


#5.5d 应用协整检验。即两个序列是否存在线性数学关系使得新组合是平稳的
from statsmodels.tsa.stattools import coint
# 进行Engel-Granger cointegration test
score, pvalue, _ = coint(X_log_cumsum[window-1:].flatten(), diff.flatten())
print("* Cointegration Test p-value: {:.4f}".format(pvalue))
# 解释结果
if pvalue < 0.05:
    print("* 拒绝原假设：序列是 协整关系")
else:
    print("* 不能拒绝原假设：序列不是 协整关系")


#5.5e 应用Jarque-Bera检验。即检验序列是否服从正态分布
from statsmodels.stats.stattools import jarque_bera
# 进行Jarque-Bera test
jb_stat, pvalue, _, _ = jarque_bera(diff.flatten())
print("* Jarque-Bera Test p-value: {:.4f}".format(pvalue))
# 解释结果
if pvalue < 0.05:
    print("* 拒绝原假设：序列不是 正态分布")
else:
    print("* 不能拒绝原假设：序列是 正态分布")

#非正态分布。所以时间序列关系里面，平稳性与记忆性更为重要，正态性是更高一级的要求，不一定能达到。



'''
5.6 Take the fracdiff series from exercise 5.
(a) Apply a CUSUM filter (Chapter 2), where h is twice the standard deviation
of the series.
(b) Use the filtered timestamps to sample a features’ matrix. Use as one of the
features the fracdiff value.
(c) Form labels using the triple-barrier method, with symmetric horizontal barriers of twice the daily standard deviation, and a vertical barrier of 5 days.
(d) Fit a bagging classifier of decision trees where:
(i) The observed features are bootstrapped using the sequential method
from Chapter 4.
(ii) On each bootstrapped sample, sample weights are determined using the
techniques from Chapter 4.
'''

#5.5a 应用CUSUM过滤器。即筛选出序列中显著的变化点

#计算标准差
diff_std=diff.std()
diff_Series=pd.Series(diff.flatten()) #转pd.Series
#将索引改为日期格式，从2025年1月1日往前倒推
end_date = pd.Timestamp('2025-01-01')
date_index = pd.date_range(end=end_date, periods=len(diff_Series), freq='h')
diff_Series.index = date_index

#原序列已经是平稳且有记忆了，直接用原序列计算CUSUM filter，不需要再加百分比变化了
event_diff = cumsum_events(diff_Series, limit = 2*diff_std) 

#5.6b
sample_diff=diff_Series[event_diff]

#5.6c 使用三重障碍法形成标签
# 定义参数
diff_Series=pd.DataFrame(diff_Series)
diff_Series.columns=['close']
ptSl = [2, 2]  # 对称水平 barriers
std = getDailyVol(diff_Series['close'], span0=100)
std = pd.DataFrame(std).rename(columns={'close': 'daily_vol'})
# 将结果合并回原DataFrame
diff_Series = diff_Series.join(std)
numDays=5
t1 = diff_Series['close'].index.searchsorted(diff_Series['close'].index + pd.Timedelta(days=numDays))
t1 = t1[t1 < diff_Series.shape[0]]  # 确保不超出范围
t1 = pd.Series(diff_Series['close'].index[t1], index=diff_Series['close'].index[:t1.shape[0]])
t1.name = 't1'
# 将t1列添加到DataFrame
diff_Series = diff_Series.join(t1)
diff_Series_event=diff_Series.loc[event_diff]
diff_Series_label=generate_metalabels(diff_Series_event, diff_Series['close'], diff_Series_event['daily_vol'], ptSl)



##########搞不懂这里############
#5.6d1 应用bagging分类器  
#要求的是使用顺序引导法进行抽样



#5.6d2 对上一步应用的顺序引导法样本确定权重  ，有可能抽样重复，所以需要重新计算权重

'''
第五章总结：
1.本章关注的是数据的平稳性与记忆性。关于记忆性的问题，切分样本的时候是不是尽量要按顺序切分，否则就破坏的记忆性（趋势性）
2.直接使用fracdiff这个包找到指定置信度（一般5%）下的最小差分阶d，就是具有最高记忆性且平稳的差分数据了。
3.from fracdiff.sklearn import FracdiffStat 就可以直接找到对应的最小d值了，使用前先用from fracdiff.sklearn.tol import window_from_tol_coef 确定指定阈值下的窗口大小。这个包的应用例子都在上面
'''


#%%
#模型 ：6-9章是介绍模型的使用
#第六章

#模型设置关键参数：
'''
随机森林模型：
1.max_features 设置小一点，可以增加树的差异性
2.将正则化参数 min_weight_fraction_leaf 设置为足够大的值（例如 5%），以使袋外准确率收敛到样本外（k 折）准确率
3.修改 RF 类，将样本取样从标准自助法改为顺序自助法  （见第四章代码）
4.可以先对特征进行主成分分析（pca），降低过拟合。
5.class_weight='balanced_subsample' 降低样本不平衡性带来的影响。
6.criterion='entropy' 提升模型性能
clf0=RandomForestClassifier(n_estimators=1000,class_weight='balanced_subsample',
criterion='entropy') 
'''

'''
6.1 Why is bagging based on random sampling with replacement? Would bagging
still reduce a forecast’s variance if sampling were without replacement?
'''


#使用有放回的随机抽样是为了增加每棵树之间的差异性，从而提高整体模型的泛化能力。如果使用无放回抽样，样本之间的差异性会减少，可能导致模型过拟合，从而无法有效降低预测的方差。

'''
6.2 Suppose that your training set is based on highly overlap labels (i.e., with low
uniqueness, as defined in Chapter 4).
(a) Does this make bagging prone to overfitting, or just ineffective? Why?
(b) Is out-of-bag accuracy generally reliable in financial applications? Why?

'''

#a 高度重叠的标签会使得袋装方法变得无效，因为模型难以学习到有意义的模式，从而无法有效降低预测的方差。
#Bagging 的核心机制是“降低方差”，但前提是模型本身能学到一些模式.历史数据本身没有规律,模型在训练集和测试集上都表现平平，无法超越随机猜测.而不是模型在训练集上表现很好，但在测试集上很差的过拟合
#b 金融数据具有强时间依赖性和非平稳性，OOB 样本虽然是“未参与训练”的，但由于是从同一时间段随机抽取的，它们与训练样本在时间上是混合的。这导致 OOB 样本并非真正独立同分布，也破坏了时间循序，带来未来函数的问题。
#使用K折来评估效果比较适宜

'''
6.3 Build an ensemble of estimators, where the base estimator is a decision tree.
(a) How is this ensemble different from an RF?
(b) Using sklearn, produce a bagging classifier that behaves like an RF. What
parameters did you have to set up, and how?

'''
#随机森林 = Bagging + 决策树 + 随机特征子集（feature subsampling）
#a 随机森林在每棵决策树的训练过程中引入了随机特征子集选择（feature subsampling），而普通的决策树集成（如Bagging）通常使用所有特征进行训练。这种随机特征选择增加了树之间的差异性，从而提高了整体模型的泛化能力。

#b
# 定义基础决策树
base_tree = DecisionTreeClassifier(
    criterion='gini',           # 分裂标准，RF 默认为 'gini'
    splitter='best',            # 最佳分裂方式
    random_state=None           # 让每棵树随机化
)

# 构建一个“类随机森林”的 Bagging 分类器
bagging_rf_clone = BaggingClassifier(
    base_estimator=base_tree,
    n_estimators=100,                     # 树的数量
    max_samples=1.0,                      # 使用 100% 的样本（有放回）
    max_features=0.5,                     # ⭐ 关键：每次分裂时随机选择 50% 的特征
    bootstrap=True,                       # 对样本进行有放回抽样
    bootstrap_features=False,             # 不对特征抽样（由 max_features 控制）
    oob_score=True,                       # 启用袋外误差评估
    random_state=42
)


'''
6.4 Consider the relation between an RF, the number of trees it is composed of, and the number of features utilized:
(a) Could you envision a relation between the minimum number of trees needed in an RF and the number of features utilized?
(b) Could the number of trees be too small for the number of features used?
(c) Could the number of trees be too high for the number of observations available?
'''
#特征数量增加，所需要树数量也会增加。树数量过少时会出现重要特征未充分学习，模型性能未达上限；未能平均掉噪音等问题
#树不会过多，理论上不会导致过拟合。但是树太多容易导致训练时间过长，预测延迟等性能与硬件上的问题。


'''
6.5 How is out-of-bag accuracy different from stratified k-fold (with shuffling) cross validation accuracy?
'''

#OOB 准确率是一种高效、专用的评估方法，适用于 bagging 集成模型；而分层 k 折交叉验证是一种通用、严谨的标准方法，适用于所有模型
#OOB不需要额外的训练，但是类别平衡保障是没有的，是有偏的，而且只能在随机森林里使用。适合做快速筛选
#k 折需要耗费时间训练，但是效果较好，所有模型都可以使用。适合做最终确认。


'''
第六章总结与要点：
1.bagging适用于处理过拟合问题，因为结果是多个树投票决定，会降低拟合度。boost用于解决欠拟合问题，因为训练事会把拟合度低的估计器丢弃，会增加拟合度。
金融数据适用于bagging，过拟合的后果往往是灾难性的。而且信噪比低，很容易过拟合
'''

#%%
#第七章 交叉验证

'''
 7.1 Why is shuffling a dataset before conducting k-fold CV generally a bad idea in
 finance? Whatis the purpose of shuffling? Why does shuffling defeat the purpose of k-fold CV in financial datasets?

'''
#打乱数据会破坏记忆性，使得模型效果变差。而且金额数据也不满足独立同分布的特性，k-FOLD的假设不能满足。
#打乱是让数据集分布更均匀，而提高泛化能力

'''
 7.2 Take a pair of matrices (X,y), representing observed features and labels. These
 could be one of the datasets derived from the exercises in Chapter 3.
(a) Derive the performance from a 10-foldCV of a RFclassifier on (X,y),without shuffling.
 (b) Derive the performance from a 10-fold CVofanRFon(X,y),with shuffling.
 (c) Why are both results so different?
 (d) How does shuffling leak information?
'''
#a 不打乱数据集进行10折交叉验证
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import make_classification

import yfinance as yf
import talib 

# ----------------------------
# 1. 准备数据 (X, y)
# ----------------------------

ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2023-12-31")
ohlcv = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
ohlcv.columns = ['open', 'high', 'low', 'close', 'volume'] # talib 习惯小写

df = ohlcv.copy()

# --- 基础价格比率 ---
df['close_to_open'] = df['close'] / df['open']
df['high_to_low'] = df['high'] / df['low']

# --- 移动平均线 ---
window_short = 20
window_long = 50
df[f'ma_{window_short}'] = talib.SMA(df['close'], timeperiod=window_short)
df[f'ma_{window_long}'] = talib.SMA(df['close'], timeperiod=window_long)
df['ma_ratio'] = df[f'ma_{window_short}'] / df[f'ma_{window_long}']

# --- 指数移动平均线 ---
df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
df['ema_diff'] = df['ema_12'] - df['ema_26']

# --- MACD (Moving Average Convergence Divergence) ---
macd, macd_signal, macd_hist = talib.MACD(df['close'])
df['macd'] = macd
df['macd_signal'] = macd_signal
# df['macd_hist'] = macd_hist # 可选

# --- 布林带 (Bollinger Bands) ---
upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=20)
df['bb_upper'] = upperband
df['bb_lower'] = lowerband
df['bb_width'] = (upperband - lowerband) / middleband # 布林带宽度

# --- 相对强弱指数 RSI ---
df['rsi'] = talib.RSI(df['close'], timeperiod=14)

# --- 随机指标 Stochastic Oscillator ---
slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
df['stoch_k'] = slowk
df['stoch_d'] = slowd

# --- 波动率 (使用 ATR - Average True Range) ---
df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

# --- 成交量指标 ---
df['volume_sma'] = talib.SMA(df['volume'].astype(float), timeperiod=20)
df['volume_ratio'] = df['volume'] / df['volume_sma']

# --- 删除因 talib 计算产生的 NaN ---
df.dropna(inplace=True)

# --- 定义特征列 ---
# 选择最终用于训练的特征 (可以根据需要调整)
feature_cols = [
    'close_to_open', 'high_to_low',
    'ma_ratio', 'ema_diff',
    'macd', 'macd_signal',
    'bb_width',
    'rsi',
    'stoch_k', 'stoch_d',
    'atr',
    'volume_ratio'
]
X = df[feature_cols].copy()

horizon = 5
df['future_close'] = df['close'].shift(-horizon) # 未来 horizon 天的收盘价
df['future_return'] = (df['future_close'] - df['close']) / df['close'] # 未来收益
df['label'] = (df['future_return'] > 0).astype(int) # 1: 上涨, 0: 下跌或持平

#生成事件以及对应的metalabel
#计算标准差  这里也可以用滑动的标准差来计算
diff_std=df['future_return'].std()

#收盘价变动超过1个标准差的抓出来当事件
event_diff = cumsum_events1(df['close'], limit = diff_std) 

#使用三重障碍法形成标签
# 定义参数
ptSl = [1, 1]  # 对称水平 barriers
std = getDailyVol(df['close'], span0=100)
std = pd.DataFrame(std).rename(columns={'close': 'daily_vol'})
# 将结果合并回原DataFrame
df = df.join(std)
numDays=15
t1 = df['close'].index.searchsorted(df['close'].index + pd.Timedelta(days=numDays))
t1 = t1[t1 < df.shape[0]]  # 确保不超出范围
t1 = pd.Series(df['close'].index[t1], index=df['close'].index[:t1.shape[0]])
t1.name = 't1'
# 将t1列添加到DataFrame
df = df.join(t1)
df_event=df.loc[event_diff]
df_label=generate_metalabels(df_event, df['close'], df_event['daily_vol'], ptSl)

# --- 对齐 X 和 y，删除包含 NaN 的行 ---
# 注意：由于未来收益的 shift(-horizon)，最后 horizon 行的 future_* 会是 NaN
# 以及 talib 计算引入的 NaN
df_final = df_label.dropna(subset=['label']) # dropna on label also handles feature NaNs from alignment

# 确保 X 和 y 索引对齐
X_final = X.loc[df_final.index]
y_final = df_final['metallabel']

# # 转换为 DataFrame（可选，便于查看）
# X = pd.DataFrame(X_final)
# y = pd.Series(y_final)

# 2. 设置 10 折交叉验证（不洗牌）

cv = KFold(n_splits=10, shuffle=False, random_state=None)  # shuffle=False 是关键！

model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. 执行交叉验证并获取性能
scores = cross_val_score(
    estimator=model,
    X=X_final,
    y=y_final,
    cv=cv,
    scoring='accuracy',  # 可改为 'roc_auc', 'f1' 等
    n_jobs=-1            # 并行加速
)

# ----------------------------
# 5. 输出结果
# ----------------------------
print("10-Fold CV Accuracy Scores (no shuffling):")
print(scores)
print(f"\nMean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

#b 打乱数据集进行10折交叉验证
cv = KFold(n_splits=10, shuffle=True, random_state=32)  # shuffle=True 是关键！
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 执行交叉验证并获取性能
scores = cross_val_score(
    estimator=model,
    X=X_final,
    y=y_final,
    cv=cv,
    scoring='accuracy',  # 可改为 'roc_auc', 'f1' 等
    n_jobs=-1            # 并行加速
)

print("10-Fold CV Accuracy Scores (with shuffling):")
print(scores)
print(f"\nMean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

#c1 打乱后数据的Mean Accuracy下降了，为什么？从0.92降低到0.91，不是很明显的变化。这个变化感觉用训练误差就能说的过去————这是一开始使用sklearn自带的数据集

#c2 使用股票数据后发现打乱后数据Mean Accuracy上升了40%，从0.5到0.7，非常夸张。这是为什么？出现了信息泄露。

#c3 使用股票数据+metalabel，打乱后Mean Accuracy从0.5198下降到了 0.4942，为什么?不是应该出现未来函数导致准确率上升么？还是说数据的记忆性被破坏导致准确率下降了？不对，打乱后是信息泄露，导致准确率上升才对。数据处理可能哪里出问题了。-------修改了random_state=36后，打乱后Mean Accuracy从0.4997 上升到了0.5198 ，符合逾期————修改为30后又与预期相反了,改为26后又符合了，看来是样本数据分布影响准确性？

#d 打乱会泄露信息，因为时间的顺序不一样了，导致未来函数的信息泄露。

'''
 7.3 Take the same pair of matrices (X,y) you used in exercise 2.
 (a) Derive the performance from a 10-fold purged CV of an RF on (X,y), with
 1%embargo.
 (b) Why is the performance lower?
 (c) Why is this result more realistic
'''

#a with 1%embargo  清除和禁止都需要手动写代码实现，没有现成的python库
#需要在这里写成清除和禁止的函数，以及应用了这样清洗数据的k-fold CV
#清除+禁止数据处理 测试集可以在中间但是前后都去掉重叠（清除），后面时间的数据还要去掉禁止
#下面的PurgedKFold和cvScore是出自书本原文，直接用就行。
from sklearn.model_selection._split import _BaseKFold
class PurgedKFold(_BaseKFold): 
    '''
        Extend KFold to work with labels that span intervals 
        The train is purged of observations overlapping test-label intervals 
        Test set is assumed contiguous (shuffle=False), w/o training examples in between
    ''' 
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.): 
            if not isinstance(t1,pd.Series): 
                raise ValueError('Label Through Dates must be a pandas series') 
            super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None) 
            self.t1=t1 
            self.pctEmbargo=pctEmbargo 

    def split(self,X,y=None,groups=None): 
        if (X.index==self.t1.index).sum()!=len(self.t1): 
            raise ValueError('X and ThruDateValues must have the same index') 
        indices=np.arange(X.shape[0]) 
        mbrg=int(X.shape[0]*self.pctEmbargo) 
        test_starts=[(i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),self.n_splits)] 
        for i,j in test_starts: 
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j] 
            maxT1Idx=self.t1.index.searchsorted(self.t1.iloc[test_indices].max()) 
            train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index) 
            train_indices=np.concatenate((train_indices,indices[maxT1Idx+mbrg:]))  #将测试集后面的禁运期（mbrg）后面的一部分数据也包含到训练集中
            yield train_indices,test_indices

def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',t1=None,cv=None,cvGen=None,pctEmbargo=0.01):
    '''
    sample_weight是与xy对应的事件级别的权重，而不是bar级别的权重，在使用第四章的重叠事件修正权重后的输入，需要在cvScore中传递


    '''
    
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    
    from sklearn.metrics import log_loss,accuracy_score
    # from clfSequential import PurgedKFold #就是上述的函数

    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    score=[]
    for train,test in cvGen.split(X=X):
        fit=clf.fit(X=X.iloc[train,:],y=y.iloc[train],sample_weight=sample_weight.iloc[train].values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X.iloc[test,:])
            score_=-log_loss(y.iloc[test],prob,sample_weight=sample_weight.iloc[test].values,labels=clf.classes_)
        else:
            pred=fit.predict(X.iloc[test,:])
            score_=accuracy_score(y.iloc[test],pred,sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)

# 生成样本权重 根据df_final这个事件生成，观察来看是有重叠的
sample_weight = pd.Series(np.ones(len(X_final)), index=X_final.index)
event_uniqueness = mpSampleTW(event=df_final, close=df['close'])
# bar_uniqueness = calculate_bar_uniqueness(events=df_final, uniqueness=event_uniqueness, bar_timestamps=df['close'].index)

model = RandomForestClassifier(n_estimators=100, random_state=42)

#执行交叉验证并获取性能
pcv_score=cvScore(model,X_final,y_final,sample_weight=event_uniqueness,scoring='accuracy',t1=df_final['t1'],cv=10,pctEmbargo=0.01)
print("10-Fold CV Accuracy Scores (with purging and embargo):")
print(pcv_score)
print(f"\nMean Accuracy: {pcv_score.mean():.4f} (+/- {pcv_score.std() * 2:.4f})")

#b #c
#7.4aMean Accuracy 在random_state=42 是0.4804，比较低  random_state=26 是 0.4807
#random_state=42  7.3a的准确率为0.5198   ,random_state=26 是 0.4997
#random_state=42  7.3b打乱后的7.3b是0.5204 ,random_state=26 是 0.5117 
#明显看到在清除+禁止后数据的稳定性上升了，收数据切分的波动减少，更加反映真实情况

#延伸：是否需要shuffle？——不行，清除和禁止都是默认按照时间循序的。

'''
 7.4 In this chapter we have focused on one reason why k-fold CV fails in financial
 applications, namely the fact that some information from the testing set leaks into
 the training set. Can you think of a second reason for CV’s failure?
'''



#时间序列的非独立同分布性，具有时间依赖


'''
 7.5 Suppose you try one thousand configurations of the same investment strategy,
 and perform a CV on each of them. Some results are guaranteed to look good,
 just by sheer luck. If you only publish those positive results, and hide the rest,
 your audience will not be able to deduce that these results are false positives, a
 statistical fluke. This phenomenon is called “selection bias.”
 (a) Can you imagine one procedure to prevent this?
 (b) What if we split the dataset in three sets: training, validation, and testing?
 The validation set is used to evaluate the trained parameters, and the testing
 is run only on the one configuration chosen in the validation phase. In what
 case does this procedure still fail?
 (c) What is the key to avoiding selection bias
'''

#这样的情况就像是在freqtrade做参数调节进行回测，然后很容易过拟合。今年就中枪了，过拟合而不自知。
#a 可能的避免方法：
#拉长时间回测，看长期的年化，最大回撤，夏普。看切片每年的收益情况 （时间）
#小资金先试盘，看看是否符合回测预期  ，或者设置验证集，进行样本外二次验证。 （样本外）
#经济逻辑，策略应有合理金融理论支撑，而非纯数据拟合 这样是有长期保证的必要条件。
#在后续第十章会深入的探讨这些问题。

#b
#未来函数，信息泄露，数据牛熊分布不均，样本重叠，数据量太少（测试集不具代表性）

#c 
#数据划分必须严格按时间顺序：训练集 → 验证集 → 测试集，禁止随机打乱；在事件驱动标签中，还需使用 清除（Purging）和禁运（Embargo） 技术，防止样本间的时间重叠污染。  
#在上一个的基础上，测试集只能使用一次，作为最终的裁判


'''
第七章要点与总结：
1.k折叠交叉验证在金融领域，不管是模型开发还是回测，都是失败的。因为K折是要求数据独立同分布的，金融数据不符合要求。第5章讲到金融数据是具有记忆性的，直接硬性划分数据会导致信息泄露——测试集的部分信息在训练集中，这就导致过拟合
第二个原因是，在模型开发过程中测试集被多次使用，从而导致多次检验和选择偏差。（这个暂未理解，后面会讲到）
2.处理一（Purging）：根据测试集的时间区间，剔除所有与测试区间存在时间重叠的训练样本，以防止“未来信息泄露”（look-ahead bias）。
3.处理二（ Embargo）：
原因：剔除可能的未来函数。

例子：
feature = MA_20 / MA_50
划分训练/测试集
测试集时间：2023-07-01 至 2023-07-31
你已经做了 Purging：确保所有训练样本的标签结束时间 < 2023-07-01
例如，最后一个训练样本的观察日是 2023-06-30，其标签覆盖 2023-07-01 到 2023-07-05 → ❌ 被 Purging 剔除
所以你保留的最后一个训练样本是 2023-06-23（标签覆盖 6/26–6/30，完全在7月前）
✅ 看似安全！

 但 Embargo 要解决的问题出现了！
考虑一个训练样本：观察日 = 2023-06-23

它的标签：基于 2023-06-26 到 2023-06-30 的价格 → ✅ 在测试前，没问题
它的特征 MA_20 / MA_50：需要 2023-04-14 到 2023-06-23 的价格数据 → ✅ 看起来也没问题？
⚠️ 关键来了：

测试集从 2023-07-01 开始
而你的训练样本用到了 2023-06-23 的价格 —— 这是测试前最后一个交易日
在实盘中，2023-06-23 的价格在 2023-06-23 收盘后才确定，而你要在 2023-06-23 盘中或之前 做出预测

处理二结论：在加一个小窗口进行Embargo即可防止未来函数。

#其实还有个小疑惑，如果事件的长度是5个bar，但是用到了比如52个bar的均值指标，这个时候清除+禁止能够防止信息泄露吗————这就是禁止所其左右的场景。在训练集1（间隔1） 测试集 （间隔2）训练集2  这样的场景中，严格来说禁止期需要大于等于（测试集+训练集2）中间的间隔2，这样测试集的信息才不会泄露到训练集2.而且由于是使用ema这样的权重方式，禁止去掉了靠近训练集2的间隔2数据，也能够有效防止信息泄露。至于在间隔1的特征数据，是不存在信息泄露的，测试集使用了部分训练集的信息是完全OK的。——为了防止数据缺失过多，只使用禁止就够了，数据量足够+需要严格防止信息泄露才需要增大禁止期。——为了提高禁止的有效性，移动平均这些指标计算要使用ema这样的越近权重越大的指标。


4.结论：每次进行训练集和测试集的划分时必须要使用清除和禁止来划分。这里有新增的k-fold类来替代sklearn自带的划分方法。
'''

#%%

#第八章 特征

'''
 8.1 Using the code presented in Section 8.6:
 (a) Generate a dataset (X,y).
 (b) Apply a PCA transformation on X, which we denote ̇ X.
 (c) Compute MDI, MDA, and SFI feature importance on ( ̇ X,y), where the base
 estimator is RF.
 (d) Do the three methods agree on what features are important? Why?
'''

#a
import pandas as pd
from sklearn.datasets import make_classification
import datetime

# def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
#     """
#     生成一个带时间索引的合成分类数据集，用于金融机器学习实验（如 AFML 框架）。

#     参数:
#         n_features: 总特征数量（默认 40）
#         n_informative: 有信息量的特征数量（真正与标签相关的特征，默认 10）
#         n_redundant: 冗余特征数量（由有信息量特征线性组合生成，默认 10）
#         n_samples: 样本数量（默认 10000）

#     返回:
#         trnsX (pd.DataFrame): 特征矩阵，索引为工作日日期
#         cont (pd.DataFrame): 包含以下列：
#             - 'bin': 二分类标签（0 或 1）
#             - 'w': 样本权重（初始为均匀权重）
#             - 't1': 事件结束时间（此处设为与样本时间相同，模拟瞬时事件）
#     """
#     # 使用 sklearn 生成合成分类数据
#     trnsX, cont = make_classification(
#         n_samples=n_samples,          # 样本数
#         n_features=n_features,        # 总特征数
#         n_informative=n_informative,  # 真实有效特征数
#         n_redundant=n_redundant,      # 冗余特征数（由有效特征派生）
#         random_state=0,               # 随机种子，确保结果可复现
#         shuffle=False                 # 不打乱顺序，保留“时间”结构（尽管数据本身无时序依赖）
#     )
    
#     # 创建以今天为终点的工作日日期索引（共 n_samples 个交易日）
#     end_date = pd.Timestamp.today().normalize()  # 获取今天的日期（去掉时分秒）
#     date_index = pd.date_range(
#         end=end_date,
#         periods=n_samples,
#         # freq=pd.tseries.offsets.BDay()  # 工作日频率（跳过周末和节假日）
#         freq='H'  # 按小时频率生成时间戳
#     )
    
#     # 转换为 pandas DataFrame 和 Series，并设置时间索引
#     trnsX = pd.DataFrame(trnsX, index=date_index)
#     cont = pd.Series(cont, index=date_index).to_frame('bin')  # 'bin' 表示二元标签
    
#     # 构造特征列名：
#     # - I_0, I_1, ... : 有信息量的特征（Informative）
#     # - R_0, R_1, ... : 冗余特征（Redundant）
#     # - N_0, N_1, ... : 噪声特征（Noise，既不相关也不冗余）
#     informative_cols = [f'I_{i}' for i in range(n_informative)]
#     redundant_cols = [f'R_{i}' for i in range(n_redundant)]
#     noise_cols = [f'N_{i}' for i in range(n_features - n_informative - n_redundant)]
#     trnsX.columns = informative_cols + redundant_cols + noise_cols
    
#     # 添加样本权重：初始设为均匀权重（后续可替换为基于唯一性的权重）
#     cont['w'] = 1.0 / cont.shape[0]
    
#     # 添加 t1 列（事件结束时间）：
#     # 在真实 triple-barrier 标签中，t1 是未来某个时间点；
#     # 此处为简化，设为当前样本时间（即每个事件只覆盖一个 bar）
#     cont['t1'] = cont.index.copy()
    
#     return trnsX, cont
def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000, target_oob_range=(0.5, 0.8)):
    """
    生成一个带时间索引的合成分类数据集，用于金融机器学习实验（如 AFML 框架）。
    改进版本：生成的数据更接近真实金融数据，OOB分数在50%-80%范围内。

    参数:
        n_features: 总特征数量（默认 40）
        n_informative: 有信息量的特征数量（真正与标签相关的特征，默认 10）
        n_redundant: 冗余特征数量（由有信息量特征线性组合生成，默认 10）
        n_samples: 样本数量（默认 10000）
        target_oob_range: 目标OOB分数范围（默认 (0.6, 0.8)）

    返回:
        trnsX (pd.DataFrame): 特征矩阵，索引为工作日日期
        cont (pd.DataFrame): 包含以下列：
            - 'bin': 二分类标签（0 或 1）
            - 'w': 样本权重（初始为均匀权重）
            - 't1': 事件结束时间（此处设为与样本时间相同，模拟瞬时事件）
    """
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # 调整参数以降低模型性能，使其更接近真实金融数据
    # 1. 降低特征与标签的相关性
    # 2. 增加类别不平衡
    # 3. 增加噪声
    
    # 初始尝试使用默认参数
    flip_y = 0.05  # 5%的标签随机翻转
    class_sep = 0.8 # 类别分离度，默认为1.0，降低此值使类别更难区分
    
    # 迭代调整参数直到达到目标OOB范围
    max_attempts = 5
    for attempt in range(max_attempts):
        # 使用 sklearn 生成合成分类数据
        trnsX, cont = make_classification(
            n_samples=n_samples,          # 样本数
            n_features=n_features,        # 总特征数
            n_informative=n_informative,  # 真实有效特征数
            n_redundant=n_redundant,      # 冗余特征数（由有效特征派生）
            flip_y=flip_y,                # 随机翻转的标签比例
            class_sep=class_sep,          # 类别分离度
            random_state=42+attempt,      # 每次尝试使用不同的随机种子
            shuffle=False                 # 不打乱顺序，保留"时间"结构
        )
        
        # 创建一个简单的随机森林来测试OOB分数
        rf_test = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            oob_score=True,
            random_state=42
        )
        
        # 拟合模型
        rf_test.fit(trnsX, cont)
        oob_score = rf_test.oob_score_
        
        print(f"尝试 {attempt+1}: flip_y={flip_y}, class_sep={class_sep}, OOB={oob_score:.4f}")
        
        # 如果OOB分数在目标范围内，跳出循环
        if target_oob_range[0] <= oob_score <= target_oob_range[1]:
            break
            
        # 调整参数
        if oob_score > target_oob_range[1]:  # OOB太高，增加难度
            flip_y = min(0.2, flip_y + 0.03)  # 增加标签噪声
            class_sep = max(0.5, class_sep - 0.1)  # 降低类别分离度
        else:  # OOB太低，降低难度
            flip_y = max(0.01, flip_y - 0.02)  # 减少标签噪声
            class_sep = min(2.0, class_sep + 0.1)  # 增加类别分离度
    
    # 创建以今天为终点的工作日日期索引（共 n_samples 个交易日）
    end_date = pd.Timestamp.today().normalize()  # 获取今天的日期（去掉时分秒）
    date_index = pd.date_range(
        end=end_date,
        periods=n_samples,
        # freq=pd.tseries.offsets.BDay()  # 工作日频率（跳过周末和节假日）
        freq='h'  # 按小时频率生成时间戳
    )
    
    # 转换为 pandas DataFrame 和 Series，并设置时间索引
    trnsX = pd.DataFrame(trnsX, index=date_index)
    cont = pd.Series(cont, index=date_index).to_frame('bin')  # 'bin' 表示二元标签
    
    # 构造特征列名：
    # - I_0, I_1, ... : 有信息量的特征（Informative）
    # - R_0, R_1, ... : 冗余特征（Redundant）
    # - N_0, N_1, ... : 噪声特征（Noise，既不相关也不冗余）
    informative_cols = [f'I_{i}' for i in range(n_informative)]
    redundant_cols = [f'R_{i}' for i in range(n_redundant)]
    noise_cols = [f'N_{i}' for i in range(n_features - n_informative - n_redundant)]
    trnsX.columns = informative_cols + redundant_cols + noise_cols
    
    # 添加样本权重：初始设为均匀权重（后续可替换为基于唯一性的权重）
    cont['w'] = 1.0 / cont.shape[0]
    
    # 添加 t1 列（事件结束时间）：
    # 在真实 triple-barrier 标签中，t1 是未来某个时间点；
    # 此处为简化，设为当前样本时间（即每个事件只覆盖一个 bar）
    cont['t1'] = cont.index.copy()
    
    print(f"最终OOB分数: {oob_score:.4f}")
    
    return trnsX, cont

trnsX,cont=getTestData()


#b 应用pca降维
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca_to_features(trnsX, n_components=None, variance_threshold=0.95):
    """
    对特征矩阵 trnsX 应用 PCA 降维。

    参数:
        trnsX (pd.DataFrame): 原始特征矩阵，索引为时间
        n_components (int or None): 指定保留的主成分数量。
            - 若为 None，则根据 variance_threshold 自动选择
        variance_threshold (float): 累积方差解释比例阈值（仅当 n_components=None 时生效）

    返回:
        trnsX_pca (pd.DataFrame): 降维后的特征矩阵，列名为 'PC_0', 'PC_1', ...
    """
    # 1. 标准化：PCA 对量纲敏感，必须先标准化（均值为0，方差为1）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(trnsX)
    
    # 2. 初始化 PCA
    if n_components is None:
        # 自动选择能解释至少 variance_threshold 方差的最少主成分
        pca = PCA(n_components=variance_threshold)
    else:
        pca = PCA(n_components=n_components)
    
    # 3. 拟合并转换
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. 构造新的列名：PC_0, PC_1, ...
    n_final = X_pca.shape[1]
    pca_columns = [f'PC_{i}' for i in range(n_final)]
    
    # 5. 转换为 DataFrame，保留原始时间索引
    trnsX_pca = pd.DataFrame(X_pca, index=trnsX.index, columns=pca_columns)
    
    # （可选）打印信息
    print(f"原始特征数: {trnsX.shape[1]}")
    print(f"降维后特征数: {n_final}")
    print(f"累积解释方差比例: {pca.explained_variance_ratio_.sum():.4f}")
    if n_components is None:
        print(f"自动选择主成分数量以保留 ≥{variance_threshold:.0%} 的方差")
    
    return trnsX_pca, pca, scaler  # 返回变换器以便后续用于新数据


# 2. 应用 PCA（保留 95% 方差）
trnsX_pca, pca_model, scaler_model = apply_pca_to_features(
    trnsX, 
    variance_threshold=0.95
)

# （可选）查看各主成分解释的方差比例
print("\n各主成分解释的方差比例:")
print(pca_model.explained_variance_ratio_)


#c 基于随机森林模型计算MDI, MDA, and SFI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

def featImpMDI(fit, featNames):
    """
    基于样本内（In-Sample）平均不纯度减少（MDI）计算特征重要性。
    
    注意：适用于每棵树只使用一个特征的 Bagging 模型（如 AFML 推荐设置），
          此时每棵树的 feature_importances_ 中只有一个非零值，其余为 0。
    
    参数:
        fit: 已训练的 BaggingClassifier 或类似集成模型（需有 .estimators_）
        featNames: 特征名称列表（与输入特征顺序一致）
    
    返回:
        pd.DataFrame: 包含 'mean' 和 'std' 两列，已归一化（mean 列总和为 1）
    """
    # 提取每棵树的特征重要性（每棵树是一个数组）
    # 当 max_features=1 时，每棵树只有一个特征的重要性非零，其余为 0
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    
    # 转为 DataFrame：行=树，列=特征
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    
    # 将 0 替换为 NaN，以便后续统计忽略未被使用的特征
    # （因为每棵树只用一个特征，其他都是 0，不代表“不重要”，而是“未使用”）
    df0 = df0.replace(0, np.nan)
    
    # 计算均值和标准误（Standard Error = std / sqrt(n)）
    mean_imp = df0.mean()
    std_err = df0.std() / np.sqrt(df0.count())  # 使用 count() 避免 NaN 影响
    
    # 构造结果 DataFrame
    imp = pd.DataFrame({
        'mean': mean_imp,
        'std': std_err
    })
    
    # 归一化：使 mean 列的总和为 1（便于解释为相对重要性）
    imp['mean'] /= imp['mean'].sum()
    
    return imp

def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    """
    基于样本外（OOS）性能下降计算 MDA（Mean Decrease Accuracy）特征重要性。
    
    参数:
        clf: 已定义但未训练的分类器（需支持 fit/predict/predict_proba）
        X: 特征 DataFrame
        y: 标签 Series
        cv: CV 折数
        sample_weight: 样本权重 Series
        t1: 事件结束时间 Series（用于 PurgedKFold）
        pctEmbargo: Embargo 比例（防止信息泄露）
        scoring: 评分方式，支持 'neg_log_loss' 或 'accuracy'
    
    返回:
        imp (pd.DataFrame): 各特征的 MDA 重要性（mean 和 std）
        oos_score (float): 原始 OOS 平均得分
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise ValueError("scoring 必须是 'neg_log_loss' 或 'accuracy'")

    # from crossValidation import PurgedKFold  # 确保该模块已实现且兼容 Python 3

    # 初始化 Purged K-Fold 交叉验证生成器
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)

    # 存储原始 OOS 分数（每折一个值）
    scr0 = pd.Series(dtype=float)
    # 存储打乱每个特征后的 OOS 分数（每折 × 每特征）
    scr1 = pd.DataFrame(columns=X.columns, dtype=float)

    # 执行交叉验证
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        # 训练集和测试集划分
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]

        # 在训练集上拟合模型
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)

        # 计算原始 OOS 分数
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            score_orig = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
        else:  # accuracy
            pred = fit.predict(X1)
            score_orig = accuracy_score(y1, pred, sample_weight=w1.values)
        scr0.loc[i] = score_orig

        # 对每个特征进行打乱测试
        for j in X.columns:
            # 深拷贝测试集特征
            X1_permuted = X1.copy()
            # 打乱第 j 列（注意：必须操作 .values 以避免 pandas 警告）
            shuffled_values = X1_permuted[j].values.copy()
            np.random.shuffle(shuffled_values)
            X1_permuted[j] = shuffled_values

            # 用打乱后的数据预测
            if scoring == 'neg_log_loss':
                prob_perm = fit.predict_proba(X1_permuted)
                score_perm = -log_loss(y1, prob_perm, sample_weight=w1.values, labels=clf.classes_)
            else:
                pred_perm = fit.predict(X1_permuted)
                score_perm = accuracy_score(y1, pred_perm, sample_weight=w1.values)
            
            scr1.loc[i, j] = score_perm

    # 计算每个特征的重要性：原始分数 - 打乱后分数（越大越重要）
    # 注意：scr0 是 Series (n_splits,), scr1 是 DataFrame (n_splits, n_features)
    imp = scr0.values[:, None] - scr1.values  # 广播相减
    imp = pd.DataFrame(imp, index=scr1.index, columns=scr1.columns)

    # 归一化（相对下降比例）
    if scoring == 'neg_log_loss':
        # 避免除零：使用打乱后的分数作为分母（AFML 原始做法）
        imp = imp / (-scr1 + 1e-10)  # 加小量防止除零
    else:
        # accuracy: 最大可能下降是 (1 - 打乱后准确率)
        imp = imp / (1.0 - scr1 + 1e-10)

    # 计算均值和标准误（Standard Error = std / sqrt(n)）
    mean_imp = imp.mean()
    std_err = imp.std() / np.sqrt(imp.shape[0])

    # 构造结果 DataFrame
    result_imp = pd.DataFrame({
        'mean': mean_imp,
        'std': std_err
    })

    return result_imp, scr0.mean()

def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    """
    计算单特征重要性（SFI）：对每个特征单独训练模型，评估其 OOS 预测能力。
    
    参数:
        featNames: 要评估的特征名称列表（或索引）
        clf: 分类器（每次会用单特征重新训练）
        trnsX: 完整特征 DataFrame
        cont: 包含 'bin'（标签）和 'w'（样本权重）的 DataFrame
        scoring: 评分指标（如 'accuracy', 'neg_log_loss'）
        cvGen: 交叉验证生成器（如 PurgedKFold）
    
    返回:
        pd.DataFrame: 每行一个特征，包含 'mean'（平均得分）和 'std'（标准误）
    """
    # 初始化结果 DataFrame
    imp = pd.DataFrame(index=featNames, columns=['mean', 'std'], dtype=float)

    from sklearn.base import clone

    

    for featName in featNames:
        clf_copy = clone(clf) # 克隆分类器模板（保留参数，但未训练）
        # 只使用当前特征进行交叉验证评分
        scores = cvScore(
            clf_copy,
            X=trnsX[[featName]],          # 注意：双括号保持 DataFrame 结构
            y=cont['bin'],
            sample_weight=cont['w'],
            scoring=scoring,
            cvGen=cvGen
        )
        
        # 计算均值和标准误（Standard Error = std / sqrt(n)）
        mean_score = scores.mean()
        std_err = scores.std() / (len(scores) ** 0.5) if len(scores) > 1 else 0.0
        
        imp.loc[featName, 'mean'] = mean_score
        imp.loc[featName, 'std'] = std_err
        # print(f"特征 {featName}: 平均分={mean_score:.4f}, 标准误={std_err:.4f}, 分数={scores}")
    return imp

def featImportance(trnsX, cont, clf=None, n_estimators=1000, cv=10, max_samples=1.,
                   numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='MDI',
                   minWLeaf=0., random_state=38):
    """
    计算特征重要性（支持传入自定义分类器），如果未提供分类器，则创建 AFML 推荐的“无偏”Bagging 模型


    参数:
        trnsX (pd.DataFrame): 特征矩阵
        cont (pd.DataFrame): 包含标签和权重的 DataFrame，必须包含 'bin'（标签）和 'w'（样本权重）列
        clf: 可选，自定义分类器实例；若为 None，则使用 AFML 推荐的 BaggingClassifier
        n_estimators (int): Bagging 中基学习器的数量
        cv (int): 交叉验证折数
        max_samples (float): Bagging 中每个基学习器使用的样本比例
        numThreads (int): 并行线程数
        pctEmbargo (float): 禁止期比例
        scoring (str): 评分方法，支持 'neg_log_loss' 或 'accuracy'
        method (str): 特征重要性计算方法，支持 'MDI' 或 'MDA'
        minWLeaf (float): 决策树叶节点的最小权重分数
        random_state (int): 随机种子
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.base import clone
    # from crossValidation import PurgedKFold

    # 如果未提供分类器，则创建 AFML 推荐的“无偏”Bagging 模型
    if clf is None:
        tree = DecisionTreeClassifier(
            criterion='entropy',
            max_features=1,
            class_weight='balanced',
            min_weight_fraction_leaf=minWLeaf,
        )
        clf = BaggingClassifier(
            estimator=tree,
            n_estimators=n_estimators,
            max_features=1.0,
            max_samples=max_samples,
            oob_score=True,
            n_jobs=(-1 if numThreads > 1 else 1)
        )

    clf = clone(clf) #重置为未训练状态
     # === DEBUG START ===
    assert not hasattr(clf, 'estimators_'), "ERROR: clf 已训练！必须传未训练模板"
    print("✅ 输入模型未训练")
    
    # 检查是否有常数特征
    const_cols = trnsX.columns[trnsX.nunique() <= 1]
    if len(const_cols) > 0:
        print(f"⚠️ 警告：存在常数特征 {list(const_cols)}，将导致 MDA=0")
    # === DEBUG END ===

    # 克隆分类器模板（保留参数，但未训练）
    clf_oob = clone(clf)

    # 拟合模型 这里会导致clf参数被训练到，下方计算传入的模型必须是未训练的,所以要另起一个同样初始配置的模型
    fit = clf_oob.fit(X=trnsX, y=cont['bin'], sample_weight=cont['w'].values)
    oob = fit.oob_score_

    # 准备 CV 生成器
    cvGen = PurgedKFold(n_splits=cv, t1=cont['t1'], pctEmbargo=pctEmbargo)

    if method == 'MDI':
        imp = featImpMDI(fit, featNames=trnsX.columns)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], sample_weight=cont['w'],scoring=scoring,t1=cont['t1'], cvGen=cvGen).mean()
    elif method == 'MDA':
        imp, oos = featImpMDA(clf, X=trnsX, y=cont['bin'], cv=cv,sample_weight=cont['w'], t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring)
    elif method == 'SFI':
        oos = cvScore(clf, X=trnsX, y=cont['bin'], sample_weight=cont['w'],scoring=scoring,t1=cont['t1'], cvGen=cvGen).mean()
        
        # SFI 使用并行计算每个特征的重要性
        clf.n_jobs = 1  # 将并行交给 mpPandasObj，而非 sklearn
        #如果没传入clf ,auxFeatImpSFI 结果貌似只跟random_state有关，使用决策树和bagging集成的话，数据一点作用都没有
        imp = mpPandasObj(
            func=auxFeatImpSFI,
            pdObj=('featNames', trnsX.columns),
            numThreads=numThreads,
            clf=clf,
            trnsX=trnsX,
            cont=cont,
            scoring=scoring,
            cvGen=cvGen
        )
    else:
        raise ValueError("method 必须是 'MDI', 'MDA' 或 'SFI'")
    print(method,' finish！')
    return imp, oob, oos


#需要对正交处理后的特征进行特征重要性计算
# 初始化随机森林分类器
rf = RandomForestClassifier(
    n_estimators=50,        # 树的数量
    max_depth=10,             # 树的最大深度
    # random_state=0,           # 随机种子
    # max_features=1,    # 关键设置，防止遮蔽效应  树模型强制max_features ≥ 1
    n_jobs=-1,                # 使用所有 CPU 核心并行计算
    oob_score=True
)

MDI_imp,MDI_oob,MDI_oos=featImportance(trnsX_pca, cont, clf=rf, n_estimators=50, cv=3, max_samples=1.,numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='MDI',minWLeaf=0., random_state=42)
MDA_imp,MDA_oob,MDA_oos=featImportance(trnsX_pca, cont, clf=rf, n_estimators=50, cv=10, max_samples=1.,numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='MDA',minWLeaf=0., random_state=42)
SFI_imp,SFI_oob,SFI_oos=featImportance(trnsX_pca, cont, clf=rf, n_estimators=50, cv=3, max_samples=1., numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='SFI',minWLeaf=0., random_state=42)
# print(MDA_oob,MDA_oos) 
# 合并三种方法的结果（确保索引对齐）
imp_df = pd.DataFrame({
    'MDI': MDI_imp['mean'] if isinstance(MDI_imp, pd.DataFrame) else MDI_imp,
    'MDA': MDA_imp['mean'] if isinstance(MDA_imp, pd.DataFrame) else MDA_imp,
    'SFI': SFI_imp['mean'] if isinstance(SFI_imp, pd.DataFrame) else SFI_imp
})
imp_df_sorted = imp_df.sort_values(by='MDA', ascending=False)
print(imp_df_sorted.round(4))
print("\n📈 OOS 性能:")
print(f"MDI OOS: {MDI_oos:.4f} | MDA OOS: {MDA_oos:.4f} | SFI OOS: {SFI_oos:.4f}")


#使用未正交的特征训练，对比噪音特征都处于低值，mda效果不错。SFI差距都不大，有难难区分
MDI_imp2,MDI_oob2,MDI_oos2=featImportance(trnsX, cont, clf=rf, n_estimators=50, cv=3, max_samples=1.,numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='MDI',minWLeaf=0., random_state=42)
MDA_imp2,MDA_oob2,MDA_oos2=featImportance(trnsX, cont, clf=rf, n_estimators=50, cv=3, max_samples=1.,numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='MDA',minWLeaf=0., random_state=42)
SFI_imp2,SFI_oob2,SFI_oos2=featImportance(trnsX, cont, clf=rf, n_estimators=50, cv=3, max_samples=1., numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='SFI',minWLeaf=0., random_state=42)

# 合并三种方法的结果（确保索引对齐）
imp_df2 = pd.DataFrame({
    'MDI': MDI_imp2['mean'] if isinstance(MDI_imp2, pd.DataFrame) else MDI_imp2,
    'MDA': MDA_imp2['mean'] if isinstance(MDA_imp2, pd.DataFrame) else MDA_imp2,
    'SFI': SFI_imp2['mean'] if isinstance(SFI_imp2, pd.DataFrame) else SFI_imp2
})
imp_df_sorted2 = imp_df2.sort_values(by='MDA', ascending=False)
print(imp_df_sorted2.round(4))
print("\n📈 OOS 性能:")
print(f"MDI OOS: {MDI_oos2:.4f} | MDA OOS: {MDA_oos2:.4f} | SFI OOS: {SFI_oos2:.4f}")

#结论以mda为主，sfi为辅筛选特征重要性。


'''
 8.2 From exercise 1, generate a new dataset (̈ X,y), where ̈ X is a feature union of X
 and ̇ X.
 (a) Compute MDI, MDA, and SFI feature importance on (̈ X,y), where the base
 estimator is RF.
 (b) Do the three methods agree on the important features? Why?
'''

import random

# 设置随机种子（可选，用于复现）
random.seed(42)

# 从 trnsX 中随机抽取 n1 个特征
n1 = 10  # 例如抽取 10 个原始特征
selected_from_trnsX = random.sample(list(trnsX.columns), k=min(n1, trnsX.shape[1]))

# 从 trnsX_pca 中随机抽取 n2 个特征
n2 = 10   # 例如抽取 5 个 PCA 特征
selected_from_trnsX_pca = random.sample(list(trnsX_pca.columns), k=min(n2, trnsX_pca.shape[1]))

# 按选定的列提取子集，并横向拼接（确保行索引对齐）
trnsX_union = pd.concat([
    trnsX[selected_from_trnsX],
    trnsX_pca[selected_from_trnsX_pca]
], axis=1)

print("新数据集 trnsX_union 的形状:", trnsX_union.shape)
print("包含的列:", list(trnsX_union.columns))

#使用合成的特征训练
MDI_imp3,MDI_oob3,MDI_oos3=featImportance(trnsX_union, cont, clf=rf, n_estimators=50, cv=3, max_samples=1.,numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='MDI',minWLeaf=0., random_state=42)
MDA_imp3,MDA_oob3,MDA_oos3=featImportance(trnsX_union, cont, clf=rf, n_estimators=50, cv=3, max_samples=1.,numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='MDA',minWLeaf=0., random_state=42)
SFI_imp3,SFI_oob3,SFI_oos3=featImportance(trnsX_union, cont, clf=rf, n_estimators=50, cv=3, max_samples=1., numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='SFI',minWLeaf=0., random_state=42)

# 合并三种方法的结果（确保索引对齐）
imp_df3 = pd.DataFrame({
    'MDI': MDI_imp3['mean'] if isinstance(MDI_imp3, pd.DataFrame) else MDI_imp3,
    'MDA': MDA_imp3['mean'] if isinstance(MDA_imp3, pd.DataFrame) else MDA_imp3,
    'SFI': SFI_imp3['mean'] if isinstance(SFI_imp3, pd.DataFrame) else SFI_imp3
})
imp_df_sorted3 = imp_df3.sort_values(by='MDA', ascending=False)
print(imp_df_sorted3.round(4))
print("\n📈 OOS 性能:")
print(f"MDI OOS: {MDI_oos3:.4f} | MDA OOS: {MDA_oos3:.4f} | SFI OOS: {SFI_oos3:.4f}")


#结果：1.MDA和MDI都会有跳跃下跌的特征，倒序排列，差一行的差别大约有3倍这样，反正差了好几倍的。只使用跳跃下降前的数据即可。
# 2.经过两个模型的结果对比，在混合模型MDA中表现较好的，在原模型也标准较好。但是有MDA误杀的特征，在MDI中没有，但是错杀的都是冗余字段。用MDA校准MDI（例如：MDI排名前10但MDA不显著 → 删除），这样更严格，找到的重要性更有效。



'''
 8.3 Take the results from exercise 2:
 (a) Drop the most important features according to each method, resulting in a
 features matrix ⃛ X.
 (b) Compute MDI, MDA, and SFI feature importance on (⃛ X,y), where the base
 estimator is RF.
 (c) Do you appreciate significant changes in the rankings of important features,
 relative to the results from exercise 2?
'''

#假设我们从上一步的结果中选择每种方法得分最高的特征进行删除

selected_features = imp_df_sorted3.MDA.nlargest(1).index.tolist() + imp_df_sorted3.MDI.nlargest(1).index.tolist() + imp_df_sorted3.SFI.nlargest(1).index.tolist()

# 从 trnsX_union 中删除选中的特征
trnsX_union_dropped = trnsX_union.drop(columns=selected_features)

print("删除后的 trnsX_union 形状:", trnsX_union_dropped.shape)
print("删除的特征:", selected_features)

#计算得分
MDI_imp4,MDI_oob4,MDI_oos4=featImportance(trnsX_union_dropped, cont, clf=rf, n_estimators=50, cv=3, max_samples=1.,numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='MDI',minWLeaf=0., random_state=42)
MDA_imp4,MDA_oob4,MDA_oos4=featImportance(trnsX_union_dropped, cont, clf=rf, n_estimators=50, cv=3, max_samples=1.,numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='MDA',minWLeaf=0., random_state=42)
SFI_imp4,SFI_oob4,SFI_oos4=featImportance(trnsX_union_dropped, cont, clf=rf, n_estimators=50, cv=3, max_samples=1., numThreads=24, pctEmbargo=0.01, scoring='accuracy', method='SFI',minWLeaf=0., random_state=42)

# 合并三种方法的结果（确保索引对齐）
imp_df4 = pd.DataFrame({
    'MDI': MDI_imp4['mean'] if isinstance(MDI_imp4, pd.DataFrame) else MDI_imp4,
    'MDA': MDA_imp4['mean'] if isinstance(MDA_imp4, pd.DataFrame) else MDA_imp4,
    'SFI': SFI_imp4['mean'] if isinstance(SFI_imp4, pd.DataFrame) else SFI_imp4
})
imp_df_sorted4 = imp_df4.sort_values(by='MDA', ascending=False)
print(imp_df_sorted4.round(4))
print("\n📈 OOS 性能:")
print(f"MDI OOS: {MDI_oos4:.4f} | MDA OOS: {MDA_oos4:.4f} | SFI OOS: {SFI_oos4:.4f}")


#c
#特征R_7在实验2里重要性接近于0，但是在实验3里重要性一跃成为第一位。其余跳跃前的特征仍旧保留在跳跃前的位置，只是顺序略有改变。
#去掉重要特征可以减少替代效应，但是会导致oob，oos降低，整体的模型性能也会下降。所以更多的是考虑PCA方法来降低替代效益，从结果上看，效果差不多，而不需要人工筛选哪些是重要特征。


'''
 8.4 Using the code presented in Section 8.6:
 (a) Generate a dataset (X,y) of 1E6 observations, where 5 features are informa
tive, 5 are redundant and 10 are noise.
 (b) Split (X,y) into 10 datasets {(Xi,yi)}i=1,…,10, each of 1E5 observations.
 (c) Compute the parallelized feature importance (Section 8.5), on each of the 10
 datasets, {(Xi, yi)}i=1,…,10.
 (d) Compute the stacked feature importance on the combined dataset (X,y).
 (e) What causes the discrepancy between the two? Which one is more reliable?
 '''

#a
trnsX,cont=getTestData(n_features=20, n_informative=5, n_redundant=5, n_samples=1000000)

#b
# 将数据集拆分为 10 个子集
trnsX_subsets = np.array_split(trnsX, 10)
cont_subsets = np.array_split(cont, 10)
#将cont_subsets的w这列的权重修正为1/len(cont_subset)
cont_subsets = [cont_subset.assign(w=1/len(cont_subset)) for cont_subset in cont_subsets]
# MDA_imp_subset1, _oob, _oos = featImportance(trnsX_subsets[1], cont_subsets[1],n_estimators=100, cv=10, pctEmbargo=0.01, scoring='accuracy', method='MDA')

#c 并行计算每个子集的特征重要性 只使用MDA 
MDA_imp_subsets = []
MDA_imp_oobs=[]
MDA_imp_ooss=[]
for trnsX_subset, cont_subset in zip(trnsX_subsets, cont_subsets):
    MDA_imp_subset, MDA_imp_oob, MDA_imp_oos = featImportance(trnsX_subset, cont_subset, clf=None,n_estimators=100, cv=10, pctEmbargo=0.01, scoring='neg_log_loss', method='MDA')
    MDA_imp_subsets.append(MDA_imp_subset)
    MDA_imp_oobs.append(MDA_imp_oob)
    MDA_imp_ooss.append(MDA_imp_oos)




# 计算每个子集的重要性均值
MDA_imp_mean = pd.concat(MDA_imp_subsets).groupby(level=0).mean()
MDA_imp_oobs_mean = np.mean(MDA_imp_oobs)
MDA_imp_ooss = np.mean(MDA_imp_ooss)
print(MDA_imp_mean,'oob:',MDA_imp_oobs_mean,'oos:',MDA_imp_ooss)

#d 在整个数据集上计算堆叠的特征重要性
MDA_imp_full, MDA_imp_oob_full, MDA_imp_oos_full = featImportance(trnsX, cont,  clf=None,n_estimators=100, cv=10, pctEmbargo=0.01, scoring='neg_log_loss', method='MDA')
print(MDA_imp_full,'oob:',MDA_imp_oob_full,'oos:',MDA_imp_oos_full)
#e 区别
#特征重要性在堆叠和并行上都能够将I,R,N进行区分，ir与N有明显的区别。

##堆叠oob: 0.814751 oos: -0.4946819712192533
#并行 oob: 0.9005750000000001 oos: -0.3266109903063382
#在OOB和OOS上并行的效果明显偏大，是有偏的，所以预测能力上应该使用堆叠更反映真实效果
#此外，堆叠的鲁棒性更好，因为训练的数据集更大，更少受极端值影响。缺点是需要的硬件条件要更高

'''
 8.5 Repeat all MDI calculations from exercises 1–4, but this time allow for masking
 effects. That means, do not set max_features=int(1) in Snippet 8.2. How do
 results differ as a consequence of this change? Why?
'''

#太费时了，就不重跑了
#遮蔽效应实例如下。所以一般情况下要使用featImportance(trnsX, cont,  clf=None）里面的clf=None，进行剔除遮蔽效应
'''
特征		标准 RF（有掩蔽）MDA	max_features=1（无掩蔽）MDA
MA5		0.12	                         0.11
MA10	0.01（被掩蔽）	                0.10
'''






'''
第八章总结：
1.要认识到回测是验证手段而不是探索发现真理的方法，特征重要性才是探索的工具。
2.如何评估特征重要性？PCA正交降维+MDI（随机森林的feature_import，会有替代效应，所以必须使用pca）/MDA（平均准确性下降,使用袋外样本，不限于随机森林模型，更泛用，也会有替代效应）/SFI（单个特征的重要性，可能会丢失特征交互效应）
3.为了避免书里列出的部分缺陷（section 8.2  8.3 遮蔽效应），需要使用改进后的模型（max_features=1,且是决策树分类器）+改进后的特征重要性计算函数，而不是使用sklearn自带的。 此外数据的处理也需要执行清楚和禁运，事件等内容，这些内容是累积上去的，而不是割裂存在的。——————所以计算MDA时，就不要使用传入的clf了，使用featImportance里面设置的集成决策树模型!!!!! 这一步很关键
4.对第三点的衍生：由于使用max_features=1的树模型会导致所有特征的SFI都相同，而MDA，SFI是可以应用于所有分类器的，所以传入一个clf模型进去，防止SFI计算失效
5.MDI只能用于随机森林模型，由于遮蔽效应，替代效益等，导致偏差，所以仅适用于初筛，应该以MDA作为基准，SFI进行辅助为佳。
6.MDA和MDI都会有跳跃下跌的特征，倒序排列，差一行的差别大约有3倍这样，反正差了好几倍的。只使用跳跃下降前的数据即可。
7.经过两个模型（trnsX_union）的结果对比，在混合模型MDA中表现较好的，在原模型也标准较好。但是有MDA误杀的特征，在MDI中没有，但是错杀的都是冗余字段。用MDA校准MDI（例如：MDI排名前10但MDA不显著 → 删除），这样更严格，找到的重要性更有效。
8.在MDA的计算中，如果模型的准确率达到95%等以上，会导致MDA无法计算，特征重要性，所有的特征重要性均值和方差都会只得到0。这个时候要改用neg_log_loss作为scoring指标。当然，在实际数据中，达到70%以上准确率都非常少见了。
'''

#%%

#第九章 交叉验证的超参优化
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold,RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier

'''
 9.1 Using the function getTestData from Chapter 8, form a synthetic dataset of
 10,000 observations with 10 features, where 5 are informative and 5 are noise.
 (a) Use GridSearchCV on 10-fold CV to find the C, gamma optimal hyper
parameters on a SVC with RBF kernel, where param_grid={'C':[1E
2,1E-1,1,10,100],'gamma':[1E-2,1E-1,1,10,100]} and the scor
ing function is neg_log_loss.
 (b) How many nodes are there in the grid?
 (c) How many fits did it take to find the optimal solution?
 (d) How long did it take to find this solution?
 (e) How can you access the optimal result?
 (f) What is the CV score of the optimal parameter combination?
 (g) How can you pass sample weights to the SVC?
'''
#a
class MyPipeline(Pipeline):
    def fit(self,X,y,sample_weight=None,**fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0]+'__sample_weight']=sample_weight
        return super(MyPipeline,self).fit(X,y,**fit_params)

#适应AFML体系的超参优化
def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0, None, 1.], 
                rndSearchIter=0, n_jobs=-1, pctEmbargo=0.01, **fit_params):
    '''
    注意：
    1.样本权重必须要在**fit_params里面传入，而且要以{Pipeline[-1][0]}__sample_weight的形式传入,Pipeline[-1][0]就是构造的Pipeline最终的分类器名称。   ————这样样本权重已经被正确的传入了网格搜索和模型里


    '''
    # 1) 设置评分标准
    # if set(lbl.values) == {0, 1}: 
    #     scoring = 'f1'  # F1分数用于二分类问题
    # else:
    #     scoring = 'neg_log_loss'  # 对数损失，用于多分类问题
    scoring = 'neg_log_loss'  # 对9.1的临时使用
    # scoring = 'accuracy'

    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  

    # 3) 超参数搜索
    if rndSearchIter == 0:
        # 使用网格搜索
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid, scoring=scoring, 
                          cv=inner_cv, n_jobs=n_jobs, ) #iid=False 已弃用，新版的sklearn默认就是False
    else:
        # 使用随机搜索
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid, 
                                scoring=scoring, cv=inner_cv, n_jobs=n_jobs, 
                                n_iter=rndSearchIter)

    # 训练并获得最佳模型
    gs = gs.fit(feat, lbl, **fit_params)  #样本权重通过{Pipeline[-1][0]}__sample_weight 在参数里传入了， 因为是使用双下划线进行传入的样本权重参数，pipeline能够调用这个参数

    best_estimator = gs.best_estimator_ 
    print("最优 CV 得分:", gs.best_score_)
    # 4) 如果需要，使用 Bagging 集成方法
    if bagging[1] > 0:
        gs = BaggingClassifier(base_estimator=MyPipeline(gs.steps), 
                               n_estimators=int(bagging[0]), max_samples=float(bagging[1]), 
                               max_features=float(bagging[2]), n_jobs=n_jobs)
        
        # 在 Bagging 中训练模型
        gs = gs.fit(feat, lbl, sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
        final_model  = Pipeline([('bag', gs)])
    else:
        final_model = best_estimator
    # 5) 返回最终的模型,搜索的结果
    return final_model,gs

trnsX,cont=getTestData(n_features=10,n_informative=5,n_redundant=0,n_samples=10000)

# 2. 构建 Pipeline：必须包含标准化（SVC 对尺度极度敏感！）
pipe = Pipeline([
    ('scaler', StandardScaler()),          
    ('svc', SVC(kernel='rbf', probability=True, random_state=42))  # probability=True 才能用 log_loss
])

# 3. 定义超参数网格
param_grid = {
    'svc__C': [1e-2, 1e-1, 1, 10, 100],      # 注意：你写的是 1E2=100，但通常从更小开始
    'svc__gamma': [1e-2, 1e-1, 1, 10, 100]
}

#网格搜索
print('开始时间：',dt.datetime.now() )
svc_model_GS,svc_GS_gs=clfHyperFit(feat=trnsX, lbl=cont['bin'], t1=cont['t1'], pipe_clf=pipe, param_grid=param_grid, cv=10, bagging=[0, 0, 1.], rndSearchIter=0,pctEmbargo=0.01, **{'svc__sample_weight':cont['w']})
print('结束时间：',dt.datetime.now() )


#b 网格搜索的节点数量
n_nodes = len(param_grid['svc__C']) * len(param_grid['svc__gamma'])
 
#c 网格搜索的拟合次数  节点数量*cv
n_fits = n_nodes * 10

#d 花费时间 每记录，大约1小时吧

#e 访问最优结果
best_params =svc_model_GS.named_steps['svc'].get_params() 

#f 最优参数组合的CV分数 函数没有返回？
best_score = svc_model_GS.best_score_



'''
 9.2 Using the same dataset from exercise 1,
 (a) Use RandomizedSearchCV on 10-fold CV to find the C,
 gamma optimal hyper-parameters on an SVC with RBF kernel,
 where
 param_distributions={'C':logUniform(a=1E-2,b=
 1E2),'gamma':logUniform(a=1E-2,b=1E2)},n_iter=25 and
 neg_log_loss is the scoring function.
 (b) How long did it take to find this solution?
 (c) Is the optimal parameter combination similar to the one found in exercise 1?
 (d) What is the CV score of the optimal parameter combination? How does it
 compare to the CV score from exercise 1?
 '''

#a
from scipy.stats import loguniform
import numpy as np,pandas as pd,matplotlib.pyplot as mpl
from scipy.stats import rv_continuous,kstest

# # 官方
# dist1 = loguniform(0.01, 100)

#差异不大，直接用官方的就行，更方便
# class logUniform_gen(rv_continuous):
# # random numbers log-uniformly distributed between 1 and e
#     def _cdf(self,x):
#         return np.log(x/self.a)/np.log(self.b/self.a)
# def logUniform(a=1,b=np.exp(1)):
#     return logUniform_gen(a=a,b=b,name='logUniform')

# dist2 = logUniform_gen(a=0.01, b=100, name='logUniform')



# 构建 Pipeline：必须包含标准化（SVC 对尺度极度敏感！）
pipe = Pipeline([
    ('scaler', StandardScaler()),          
    ('svc', SVC(kernel='rbf', probability=True, random_state=42))  # probability=True 才能用 log_loss
])

#定义超参数网格
param_grid = {
    'svc__C': loguniform(a=1E-2,b=1E2),    
    'svc__gamma': loguniform(a=1E-2,b=1E2)
}


print('开始时间：',dt.datetime.now() )
svc_model_RS,svc_RS_gs=clfHyperFit(feat=trnsX, lbl=cont['bin'], t1=cont['t1'], pipe_clf=pipe, param_grid=param_grid, cv=10, bagging=[0, 0, 1.], rndSearchIter=25,pctEmbargo=0.01, **{'svc__sample_weight':cont['w']})
print('结束时间',dt.datetime.now() )

#b 花费时间只有25分钟，明显变快了

#c 'C': 0.20446553804452852,'gamma': 0.1902583355975827  在上一份里面'C': 1，'gamma': 0.1,数据是接近的，但是不是一样
best_params =svc_model_RS.named_steps['svc'].get_params() 

#d 最优 CV 得分: -0.6300694759525192 ，在上一份练习 -0.6569474181788084.比之前的快，而且得分更高


'''
 9.3 From exercise 1,
 (a) Compute the Sharperatio of the resulting in-sample forecasts, from point 1.a
 (see Chapter 14 for a definition of Sharpe ratio).
 (b) Repeat point 1.a, this time with accuracy as the scoring function. Compute
 the in-sample forecasts derived from the hyper-tuned parameters.
 (c) What scoring method leads to higher (in-sample) Sharpe ratio?
'''

#计算样本内的夏普  这不是标准夏普，只是类似夏普的一个均值/方差得分稳定性 参考性不高
def in_sample_sharpe_ratio(clf):
    sharpe_ratio = []
    for i in np.arange(len(clf.cv_results_['mean_test_score'])):
        if clf.cv_results_['mean_test_score'][i] < 0:
            sharpe_ratio.append(-1 * clf.cv_results_['mean_test_score'][i]/ clf.cv_results_['std_test_score'][i])
        else:
            sharpe_ratio.append(clf.cv_results_['mean_test_score'][i]/ clf.cv_results_['std_test_score'][i])
    print("IS Best Score Sharpe Ratio: {0:.6f}".format(sharpe_ratio[clf.best_index_]))
    print("Best IS Sharpe ratio: {0:.6f}\nLowest IS Sharpe Ratio: {1:.6f}\nMean Sharpe Ratio: {2:.6f}".format(max(sharpe_ratio), min(sharpe_ratio), np.mean(sharpe_ratio)))

in_sample_sharpe_ratio(svc_GS_gs)
#IS Best Score Sharpe Ratio: 8.743683
# Best IS Sharpe ratio: 24.950627
# Lowest IS Sharpe Ratio: 1.285766
# Mean Sharpe Ratio: 13.120860


#b accuracy 
in_sample_sharpe_ratio(svc_GS_gs)
# IS Best Score Sharpe Ratio: 4.153074
# Best IS Sharpe ratio: 4.153074
# Lowest IS Sharpe Ratio: 0.881693
# Mean Sharpe Ratio: 1.033248

#c neg_log_loss得到的结果更好，更稳定。这是为啥呢？

'''
 9.4 From exercise 2,
 (a) Compute the Sharpe ratio of the resulting in-sample forecasts, from point
 2.a.
 (b) Repeat point 2.a, this time with accuracy as the scoring function. Compute
 the in-sample forecasts derived from the hyper-tuned parameters.
 (c) What scoring method leads to higher (in-sample) Sharpe ratio?
 '''

# a neg_log_loss
in_sample_sharpe_ratio(svc_RS_gs)
# IS Best Score Sharpe Ratio: 6.121556
# Best IS Sharpe ratio: 19.323110
# Lowest IS Sharpe Ratio: 1.344132
# Mean Sharpe Ratio: 11.597977

#b accuracy 
in_sample_sharpe_ratio(svc_RS_gs)
# IS Best Score Sharpe Ratio: 0.881693
# Best IS Sharpe ratio: 0.881693
# Lowest IS Sharpe Ratio: 0.881693
# Mean Sharpe Ratio: 0.881693

#c neg_log_loss得到的结果更好，更稳定。






'''
 9.5 Read the definition of log loss, L[Y,P].
 (a) Why is the scoring function neg_log_loss defined as the negative log loss,
 −L[Y,P]?
 (b) What would be the outcome of maximizing the log loss, rather than the neg
ative log loss?
'''
#(a) 因为sklearn的超参优化函数都是最大化评分函数，所以为了最小化log_loss，需要取负值，使得最小化log_loss等价于最大化-neg_log_loss。

# 最大化negative log loss 改为最大化log loss会导致模型故意学坏


'''
 9.6 Consider an investment strategy that sizes its bets equally, regardless of the fore
cast’s confidence. In this case, what is a more appropriate scoring function for
 hyper-parameter tuning, accuracy or cross-entropy loss?
'''


#准确率比较合适，不管置信度高低，反正每次下注都一样。只考虑这次的信息是否正确。F1-score 也是考虑的一个方向，发出的信号可能是不平衡的，所以F1-score 更好。
#进一步，对于这样的等额下注，可以自定义 scorer：score = mean(returns[y_pred == y_true] - returns[y_pred != y_true])  关注长期期望收益




'''
第九章总结：
1.介绍了优化后的超参优化函数，能够适配上面提到的清除和禁止。提出使用随机超参，我觉得比网格搜索好。
2.本章核心。使用meta-labeling时使用f1评分进行估计，而其他情况要使用neg_log_loss进行评分。使用accuracy进行评分会容易导致预测错误而亏损。因为投资的收益来源于对高置信的正确预测，而准确率对于高置信度与低置信度的预测是没有区分的，都是1-0，这样。而使用！！！！！
3.根据第二点的衍生：仓位，风险度依赖于置信度。比如预测上涨概率为0.6时仓位为1W，概率为0.8时仓位为5W，这样的一个基于置信度的动态调整。
4.对于meta-labeling时可以使用f1或者accuracy进行评分,meta-labeling标签已代表“经济结果”,是直接决策做还是不做，无需概率校准，而且不是使用在主模型上的。所以仅有这个例外。
'''



#%%

#第十章 下注大小  根据机器学习结果调整下注的大小

from scipy.stats import norm

def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kargs):

    '''
    注：
    1.由于使用了t-value of OvR，所以一个潜在的前提是假设每个类别的概率是相等的，即所有类别的先验概率都是1/numClasses。 所以输入的分类是要基本平衡的。如果分类不均衡，比如一个类别的概率是0.9，另一个类别的概率是0.1，那么这个类别的信号会被放大，而另一个类别的信号会被缩小。导致对小类别的信号下注变少，减少了recall率，从而降低盈利。   当然经过主模型事件式处理后是接近于均衡的——若使用RF/XGB，必须显式加 CalibratedClassifierCV(..., method='isotonic')能够减少不平衡？未验证
    '''

    # Get signals from predictions
    if prob.shape[0] == 0:
        return pd.Series()
    # 1) Generate signals from multinomial classification (one-vs-rest, OvR)
    signal0 = (prob - 1. / numClasses) / (prob * (1. - prob)) ** 0.5  # t-value of OvR 预测概率转换为标准正态置信度分布的分位数
    signal0 = pred * (2 * norm.cdf(signal0) - 1)  # Signal = side * size  转为[0,1]的信号，方向看pred

    if 'side' in events:
        signal0 *= events.loc[signal0.index, 'side']  # Meta-labeling
    # 2) Compute average signal among those concurrently open
    df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
    df0 = avgActiveSignals(df0, numThreads)  # 计算每个事件起始时间点上投注的平均值

    signal1 = discreteSignal(signal0=df0, stepSize=stepSize) # 离散化信号，将连续信号转换为离散信号。减少平均化导致的小额交易，得平均投注额增长到一定额度再执行交易
    return signal1

def avgActiveSignals(signals, numThreads):
    # compute the average signal among those active
    # 1) time points where signals change (either one starts or one ends)
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    
    # Convert set to a sorted list
    tPnts = list(tPnts)
    tPnts.sort()

    out = mpPandasObj(mpAvgActiveSignals, ('molecule', tPnts), numThreads, signals=signals)
    
    return out

def mpAvgActiveSignals(signals, molecule):
    '''
    At time loc, average signal among those still active.
    在时间重叠的事件中，将重叠部分的信号进行平均。直到事件结束。
    '''
    
    out = pd.Series(dtype=float)  # 创建一个空的 Series，用于存储结果

    for loc in molecule:
        # 筛选出在 loc 时间点上仍然活跃的信号
        df0 = (signals.index.values <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
        
        act = signals[df0].index  # 获取活跃信号的索引

        if len(act) > 0:
            # 如果有活跃信号，计算它们的平均值
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            # 如果没有活跃信号，设置为 0
            out[loc] = 0
    
    return out

def discreteSignal(signal0, stepSize):
    """
    离散化下注信号，将连续信号转换为离散信号。减少平均化导致的小额交易，得平均投注额增长到一定额度再执行交易。
    
    Parameters:
    signal0 : array-like
        The input signal to be discretized.
    stepSize : float
        The size of the steps for discretization.  # 离散化的步长 0.2，0.3这样的步长

    Returns:
    np.ndarray
        The discretized signal capped between -1 and 1.
    """
    
    # Discretize the signal by rounding to the nearest stepSize
    signal1 = (signal0 / stepSize).round() * stepSize  # Discretize
    signal1[signal1 > 1] = 1  # Cap values above 1
    signal1[signal1 < -1] = -1  # Floor values below -1
    
    return signal1


'''
10.1 Using the formulation in Section 10.3, plot the bet size (m) as a function of the
maximum predicted probability (̃p) when ‖X‖ = 2,3,…,10.
'''
import matplotlib.pyplot as plt
#画出getSignal 参数numClasses= 2,3,…,10.时的# 1 单事件(bet size)

#构建数据
n_samples = 10000
min_prob = 1e-3 #by right we should used [-1,0], but to avoid -inf and error msg we use something else
max_prob = 1.
class_labels = np.arange(2,11)
steps = [0.01, 0.05, 0.1]

def make_randomt1_data(n_samples: int =10000, max_days: float = 5., Bdate: bool = True):
    # generate a random dataset for a classification problem
    if Bdate:
        _freq = pd.tseries.offsets.BDay()
    else:
        _freq = 'D'
    _today = dt.datetime.today()
    df0 = pd.date_range(periods=n_samples, freq=_freq, end=_today)
    rand_days = np.random.uniform(1, max_days, n_samples)
    rand_days = pd.Series([dt.timedelta(days = d) for d in rand_days], index = df0)
    df1 = df0 + pd.to_timedelta(rand_days, unit='d')
    df1.sort_values(inplace=True)
    X = pd.Series(df1, index = df0, name='t1').to_frame()
    return X

X =  make_randomt1_data(n_samples=n_samples,
                           max_days = 25.,
                           Bdate = False) # True = business days only


X["prob"] = np.linspace(start = min_prob, 
                            stop = max_prob,
                            num = n_samples,
                            endpoint = False)

plt.figure(figsize=(12,8))
for cls in class_labels:
    
    X["Z_score"] = X["prob"].apply(lambda prob: (prob - 1/cls) / (prob * (1 - prob))**0.5)
    X["bet_size_prob"] = X.apply(
    lambda z: (2 * norm.cdf(z["Z_score"]) - 1) , # 转换为[-1,1]的信号
    axis=1
)
#     X["bet_size_prob2"] = X.apply(
#     lambda z: (2 * norm.cdf(z["Z_score"]) - 1) * (1 if (z['Z_score'] > 0) else -1),
#     axis=1
# )
    plt.plot(X["prob"],X["bet_size_prob"], label=f"||X||={cls}", linewidth=2, alpha=1)
    
plt.ylim(-1, 1)
plt.xlim(0, 1) 
plt.axhline(y=0, c='r',ls='--')
plt.axvline(x=0.1, c='r',ls='--') # predict prob = 0.1
plt.axvline(x=0.33, c='r',ls='--') 
plt.axvline(x=0.5, c='r',ls='--') # predict prob = 0.5
plt.ylabel("Bet Size $m=2Z[z]-1$")
plt.xlabel(r"Maximum Predicted Probability $\tilde{p}=max_i${$p_i$}")
plt.title("Bet Size vs. Maximum Predicted Probability")
plt.legend(title="Number of bet size labels")
plt.show()


# signal0 = (prob - 1. / numClasses) / (prob * (1. - prob)) ** 0.5  # t-value of OvR
# signal0 = pred * (2 * norm.cdf(signal0) - 1)  # Signal = side * size



'''
10.2 Draw 10,000 random numbers from a uniform distribution with bounds
U[.5,1.].
(a) Compute the bet sizes m for ‖X‖ = 2.
(b) Assign 10,000 consecutive calendar days to the bet sizes.
(c) Draw 10,000 random numbers from a uniform distribution with bounds
U[1,25].
(d) Form a pandas series indexed by the dates in 2.b, and with values equal
to the index shifted forward the number of days in 2.c. This is a t1 object
similar to the ones we used in Chapter 3.
(e) Compute the resulting average active bets, following Section 10.4.
'''
#a
lower_bound = 0
upper_bound = 1.
num_samples = 10000
prob = np.random.uniform(lower_bound, upper_bound, num_samples)
pred= np.where(prob > 0.5, 1, -1)
Z_scores = (prob - 1/2) / np.sqrt(prob * (1 - prob))
bet_size = (2 * norm.cdf(Z_scores) - 1)
#b
start_date = dt.datetime.today()
end_date = start_date + dt.timedelta(days=10000-1)
dates = pd.date_range(start=start_date, end=end_date, freq='D')
X = pd.Series(bet_size, index=dates).to_frame(name='signal')
X['prob']=prob
X['pred']=pred
#c#d
rand_days = np.random.uniform(1, 25, num_samples)
rand_days = pd.Series([dt.timedelta(days = d) for d in rand_days], index = dates)
t1 = dates + pd.to_timedelta(rand_days, unit='d')

#e
X['t1'] = t1

avg_bet_size =avgActiveSignals(X, 5)
avg_bet_size=avg_bet_size.to_frame(name='avg_bet_size')

avg_bet_size=avg_bet_size.merge(X, left_index=True, right_index=True,how='left')


# avg_bet_size.to_excel(r'D:\Git\book\avg_bet_size.xlsx')




'''
10.3 Using the t1 object from exercise 2.d:
(a) Determine the maximum number of concurrent long bets, ̄cl.
(b) Determine the maximum number of concurrent short bets, ̄cs.
(c) Derive the bet size as mt = ct,l*1̄/cl − ct,s ̄*1/cs, where ct,l is the number of con
current long bets at time t, and ct,s is the number of concurrent short bets at
time t.
'''
#a 确定最大的并发多头
longs = X[X['pred'] == 1].copy()	
long_events = []
for idx, row in longs.iterrows():
    t0 = idx
    t1 = row['t1']
    if t0 <= t1:  # 防止无效区间
        long_events.append((t0, +1))
        long_events.append((t1, -1))

# 排序事件，按时间顺序
long_events.sort(key=lambda x: (x[0], x[1]))

# 初始化并发多头和短头计数
ct_l = 0
max_ct_l = 0

for t, delta in long_events:
    ct_l += delta
    max_ct_l = max(max_ct_l, ct_l)

print(f"最大并发多头数: {max_ct_l}")

#b 确定最大的并发空头
shorts = X[X['pred'] == -1].copy()	
short_events = []
for idx, row in shorts.iterrows():
    t0 = idx
    t1 = row['t1']
    if t0 <= t1:  # 防止无效区间
        short_events.append((t0, +1))
        short_events.append((t1, -1))

# 排序事件，按时间顺序
short_events.sort(key=lambda x: (x[0], x[1]))

ct_s = 0
max_ct_s = 0

for t, delta in short_events:
    ct_s += delta
    max_ct_s = max(max_ct_s, ct_s)

print(f"最大并发空头数: {max_ct_s}")

#c 修改下注大小等于 ct,l*1̄/max_ct_l − ct,s ̄*1/max_ct_s
#将每个时间点的多空头计算写到X的列
X['ct_l'] = 0
for t, delta in long_events:
    X.loc[t:, 'ct_l'] += delta


X['ct_s'] = 0
for t, delta in short_events:
    X.loc[t:, 'ct_s'] += delta



#计算ct,l*1̄/max_ct_l − ct,s ̄*1/max_ct_s 作为下注大小   
# ##  这里的下注大小是根据并发多头和空头数量计算得到的，不是根据模型预测置信度计算得到的。
X['bet_size'] = X['ct_l']*1/max_ct_l - X['ct_s']*1/max_ct_s







'''
10.4 Using the t1 object from exercise 2.d:
(a) Compute the series ct = ct,l − ct,s, where ct,l is the number of concurrent
long bets at time t, and ct,s is the number of concurrent short bets at time t.
(b) Fit a mixture of two Gaussians on {ct}. You may want to use the method
described in L´opez de Prado and Foreman [2014].

(c) Derive the bet size as mt = if ct ≥ 0, then
(F[ct]−F[0])
/(1−F[0])
, else
(F[ct]−F[0])/F[0]
, where F[x] is the CDF of the fitted mixture of two Gaussians for a value x.
(d) Explain how this series {mt} differ from the bet size series computed in
exercise 3.

'''
#a 计算并发多头和空头数量的差值
X['ct'] = X['ct_l'] - X['ct_s']

#b 拟合双高斯混合模型
#| 核心思想 | ct 值并非来自单一随机过程，而是由两个隐藏状态（latent regimes）交替生成，如 “震荡盘整、均值回归”态， “上涨趋势、动量强化”态
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
ct = X['ct'].dropna().values.reshape(-1, 1)  # shape: (n_samples, 1)
scaler = StandardScaler()
ct_scaled = scaler.fit_transform(ct)

gmm = GaussianMixture(
    n_components=2,
    covariance_type='full',   # 允许每个高斯有独立方差（推荐）
    random_state=42,         # 可复现
    max_iter=100,
    tol=1e-4
)
gmm.fit(ct_scaled)

# --- Step 5: 计算后验概率 P(state=k | ct) ---
posterior = gmm.predict_proba(ct_scaled)  # shape: (n, 2)

# --- Step 6: 还原高斯参数到原始 ct 尺度（关键！）---
# 均值还原：μ_orig = μ_scaled * σ_ct + μ_ct
mu_ct = X['ct'].mean()
std_ct = X['ct'].std()
means_orig = scaler.inverse_transform(gmm.means_.reshape(-1, 1)).flatten()
# 方差还原：Var_orig = Var_scaled * σ_ct²
covs_orig = gmm.covariances_.flatten() * (std_ct ** 2)
stds_orig = np.sqrt(covs_orig)

# --- Step 7: 将结果写回 X（严格对齐索引）---
valid_idx = X['ct'].dropna().index
X.loc[valid_idx, 'ct_gmm_prob_state1'] = posterior[:, 0]  # P(state 1 | ct)  状态1的后验概率
X.loc[valid_idx, 'ct_gmm_prob_state2'] = posterior[:, 1]  # P(state 2 | ct)  状态2的后验概率
X.loc[valid_idx, 'ct_gmm_state'] = np.argmax(posterior, axis=1) + 1  # 1 or 2 #模型预测处于状态

# --- Step 8: 打印解读性结果 ---
weights = gmm.weights_
means = means_orig
stds = stds_orig

print("🔍 GMM Fitted on Net Concurrent Exposure ct = ct_l - ct_s:")
print(f"   State 1 weight: {weights[0]:.3f}  → likely 'neutral/low-exposure' regime")
print(f"   State 2 weight: {weights[1]:.3f}  → likely 'high-exposure' regime (trendy)")
print(f"   State 1 mean:   {means[0]:+.4f}  (std = {stds[0]:.4f})")
print(f"   State 2 mean:   {means[1]:+.4f}  (std = {stds[1]:.4f})")

#结果及解释
'''
GMM Fitted on Net Concurrent Exposure ct = ct_l - ct_s:
   State 1 weight: 0.697  → likely 'neutral/low-exposure' regime
   State 2 weight: 0.303  → likely 'high-exposure' regime (trendy)
   State 1 mean:   +1.1495  (std = 3.0457)
   State 2 mean:   -2.9973  (std = 2.9976)

State 1（70%，μ₁ ≈ +1.15）→ “温和多头主导的平衡市” 市场缓慢爬升，多头信号零星触发、空头极少
std=3.0 很大 → State 1 内部变化剧烈：可能从 ct=−3（强空）到 ct=+5（强多）都属于它；
不能简单认为 State 1 = “安全区” —— 它是主流量，但包含大量噪声和假突破。

State 2（30%，μ₂ ≈ −3.00）→ “强空头主导的趋势市” 市场快速下降，空头信号频繁触发、多头极少
虽只占 30% 时间，但这是 高信息量、高确定性的趋势段

#然后就可以对不同状态的市场采取不同的下注策略

此外，还可以对市场切分为均值回归和趋势两种市场状态等等，或者两两交叉为4种状态，高斯混合模型还是有点东西的。

'''

# X.to_excel(r'D:\Git\book\X_with_gmm_ct.xlsx')


#c 计算下注大小
def F(x):
    """
    Cumulative Distribution Function (CDF) of the fitted 2-Gaussian Mixture.
    
    Parameters:
    -----------
    x : float or array-like
        Point(s) at which to evaluate the CDF.
    
    Returns:
    --------
    float or np.ndarray : F(x) = P(ct <= x)
    """
    x = np.asarray(x)
    # 对每个高斯成分计算 Φ((x - μ)/σ)，再加权求和
    cdf1 = norm.cdf(x, loc=means[0], scale=stds[0])
    cdf2 = norm.cdf(x, loc=means[1], scale=stds[1])
    return weights[0] * cdf1 + weights[1] * cdf2



X['mt'] = np.where(
    X['ct'] >= 0,
     (F(X['ct']) - F(0)) / (1 - F(0)),
    (F(X['ct']) - F(0)) / F(0)
)


#d
#评价：10.3的bet_size 大多时间都是在0附近，只有极端值时才会有较大的下注大小。
#10.4的mt下注更加贴近ct的状态，即市场的两种状态。下注更为灵敏，而且不会只局限在0附近，更好。
#但是都是依赖于市场状态判断，而不是对于模型预测置信度，直接的依赖于市场判断的正确性pred。更适合对指数进行判断，而不是对个股进行判断。从这个角度看，课本的betsize计算方式更好



'''
10.5 Repeat exercise 1, where you discretize m with a stepSize=.01,
stepSize=.05, and stepSize=.1.
'''

#构建数据
n_samples = 10000
min_prob = 1e-3 #by right we should used [-1,0], but to avoid -inf and error msg we use something else
max_prob = 1.
class_labels = np.arange(2,11)
steps = [0.01, 0.05, 0.1]

X =  make_randomt1_data(n_samples=n_samples,
                           max_days = 25.,
                           Bdate = False) # True = business days only


X["prob"] = np.linspace(start = min_prob, 
                            stop = max_prob,
                            num = n_samples,
                            endpoint = False)

for step in steps:
    plt.figure(figsize=(12,8))
    for cls in class_labels:
        
        X["Z_score"] = X["prob"].apply(lambda prob: (prob - 1/cls) / (prob * (1 - prob))**0.5)
        X["bet_size_prob"] = X.apply(
        lambda z: (2 * norm.cdf(z["Z_score"]) - 1) , # 转换为[-1,1]的信号
        axis=1
    )
        X['bet_size_step'] = discreteSignal(X["bet_size_prob"], step)
        plt.plot(X["prob"],X["bet_size_step"], label=f"||X||={cls}", linewidth=2, alpha=0.5)

        
    plt.ylim(-1, 1)
    plt.xlim(0, 1) 
    plt.axhline(y=0, c='r',ls='--')
    plt.axvline(x=0.1, c='r',ls='--') # predict prob = 0.1
    plt.axvline(x=0.33, c='r',ls='--') 
    plt.axvline(x=0.5, c='r',ls='--') # predict prob = 0.5
    plt.ylabel("Bet Size $m=2Z[z]-1$")
    plt.xlabel(r"Maximum Predicted Probability $\tilde{p}=max_i${$p_i$}")
    plt.title(f"Bet Size vs. Maximum Predicted Probability，stepSize={step}")
    plt.legend(title="Number of bet size labels")
    plt.show()

#随着stepSize的增加，bet_size的变化更加缓慢，在一定范围内变化时不执行交易，只在变化超过stepSize时才执行交易。减少手续费的风险,降低交易频率。



'''
10.6 Rewrite the equations in Section 10.6, so that the bet size is determined by a
power function rather than a sigmoid function.
'''



'''
10.7 Modify Snippet 10.4 so that it implements the equations you derived in exercise 6.

'''





'''
第十章总结：
1.根据模型预测置信度调整下注大小。
2.（深入）已经下注的订单根据模型预测与市场价格动态调整下注额和确定下单限价。.
3. 注 使用了norm.cdf(signal0) 但是咩有要求数据是正态分布，这里只是映射到【-1,1】范围使用。只是作为归一化工具
4.可以使用高斯混合模型来对市场状态进行切分，从而更好的判断市场状态。比如切分为震荡市和趋势市，再使用高斯混合模型判断市场。
5. 10.6和10.7内容是动态调整下注大小的，即不要固定仓位，而是根据模型预测置信度动态调整仓位。改为幂函数效果见FIGURE10.3 。可是模型怎么对同一事件进行持续预测呢？而且预测是的价格才好动态调整。这部分属于进阶内容了，暂时不展开。先使用与事件相关的betsize调整吧。（伪动态调整）
'''


#%%

#第十一章 回测的危险

'''
11.1 An analyst fits an RF classifier where some of the features include seasonally
adjusted employment data. He aligns with January data the seasonally adjusted
value of January, etc. What “sin” has he committed?
'''
#使用了未来函数，1月份季度调整值在1月份是没有公布的


'''
11.2 Ananalyst develops an MLalgorithm where he generates a signal using closing
prices, and executed at close. What’s the sin?
'''
#无法成交，而且有使用未来函数的嫌疑


'''
11.3 There is a 98.51% correlation between total revenue generated by arcades and computer science doctorates awarded in the United States. As the number of doctorates is expected to grow, should we invest in arcades companies? If not, what’s the sin?
'''
#相关性不是因果性。这样投资会吃大亏。可以通过经济学逻辑，样本外数据验证，构建事件驱动标签来验证，将相关性转为可投资信号。
#经济逻辑：“博士增多 → 影响谁？谁的行为会变？如何影响街机？” → 若找不到合理链条（如“博士创业投AR公司→AR街机设备升级→客单价↑”），则放弃。
#样本外数据验证：找其他国家，其他时间段的样本看看。
#构建事件：定义：“某地新增 1 所 CS 强校 + 该市 6 个月内新开 ≥2 家 VR 街机馆” → 是否预示未来 12 个月本地街机上市公司营收超预期？ 并以此类事件来决策自己的投资。

'''
11.4 The Wall Street Journal has reported that September is the only month of the
year that has negative average stock returns, looking back 20, 50, and 100 years.
Should we sell stocks at the end of August? If not, what’s the sin?
'''
#日历效应？。犯了一个叫未预注册假设（No Pre-Registered Hypothesis）的错误。任何可交易信号必须源于事前（ex-ante）经济/行为逻辑，而非事后（ex-post）数据挖掘。

# 现在微盘股股里面也有这样类似的说法，1月，4月。甚至有对应的解释：年底回收钱，4月发财报。这是又微观机制支持，而且只存在小盘股的，即先预设小盘股存在这样的日历效应。当然，如果将其按周重采样或许就没有了，这个措施还没有推广到周维度，说明对特征重要性的挖掘还不到位。
#但是本质上这仍旧是相关性而不是因果性————————为啥用于避险就觉得合理，但是用于投资就觉得风险很大呢？看来还是倾向于规避风险。

#可以对任何市场进行类似的日历效应测试。


'''
11.5 Wedownload P/E ratios from Bloomberg, rank stocks every month, sell the top
quartile, and buy the long quartile. Performance is amazing. What’s the sin
'''

#市盈率是用最新财报回填的，即存在未来函数.




'''
第十一章总结：
1.回测是一种假设，而不是实验，不能反映未来的真实情况。回测效果非常好，也可能是过拟合了。
2.对一种策略采取多次回测，然后选择其中一种，就很容易陷入选择性偏差的情况。（web3吃过亏了）有一种情况，如聚宽，大多数人分享的回测结果，仅是那些看似获胜的投资策略。这里面是有选择性偏差的。
3.在所有处理完成前不要进行回测。（1-10章）
4.还可以生成模拟数据进行测试，而不单单使用历史数据，历史数据只是一种可能走势中的一种。
5.Apply bagging (Chapter 6) as a means 减少过拟合的产生。
6.对整个资产类别进行模型开发，而不仅仅是单个资产，这样有助于抓住共性分散投资。（保守的开发）
7.提出使用 probability of backtest overfitting (PBO) 方法对选择性偏差进行评估。提高样本外回测效果可信度。

'''

#%%

#第十二章 cpcv回测的正确姿势

'''
12.1 Suppose that you develop a momentum strategy on a futures contract, where the forecast is based on an AR(1) process.  You backtest this strategy using the WF method, and the Sharpe ratio is 1.5.  You then repeat the backtest on the reversed series and achieve a Sharpe ratio of –1.5.  What would be the mathe matical grounds for disregarding the second result, if any?
'''
#wf 方法的反向序列验证的是策略是否稳健，是否只是正好的搭上了市场的便车，依赖于市场的历史趋势。如果在反向序列表现良好，说明策略捕捉的可能是与市场方向无关的行为模式。
#这里虽然反向也是1。5的夏普，说明捕捉到了一些与市场方向无关的行为模型，但是由于还是单一历史路径，鲁棒性仍旧不够高。

#补充：
#对于平稳 AR(1)，反向序列也是 AR(1)，且参数相同，只是残差序列的顺序反了。如果正向夏普 = 1.5，反向夏普 ≈ -1.5，这恰好说明策略完全依赖该序列的特定方向性趋势，策略可能只是做多了样本内的整体趋势。所以如果实盘则风险极高，不稳健。

'''
12.2 You develop a mean-reverting strategy on a futures contract. Your WF backtest achieves a Sharpe ratio of 1.5. You increase the length of the warm-up period, and the Sharpe ratio drops to 0.7. You go ahead and present only the result with the higher Sharpe ratio, arguing that a strategy with a shorter warm-up is more realistic. Is this selection bias?
'''

#肯定是选择性偏差，只展示回测较好的结果。而且预热期增加，导致夏普急剧下降，说明稳定性很差。
#热身期不是越长越好。还得看策略是什么类型的，如果是高频套利的，5分钟-两三个小时就足够了，不需要几天的数据。并不是热身期越长越好，对wf来说。
# 高频统计套利 → 热身 5–15 分钟（tick 级）；
# 日线均值回归 → 热身 10–30 日（捕捉短期均值，避开季度趋势）；
# 宏观因子择时 → 热身 6–12 个月（需覆盖完整经济周期片段）
#但是就稳定性来说，貌似是越长越好，对各种情况都能够处理，但是相应的，下注就会变小，模型越发的谨慎。

'''
12.3 Your strategy achieves a Sharpe ratio of 1.5 on a WF backtest, but a Sharpe ratio of 0.7 on a CV backtest. You go ahead and present only the result with the higher Sharpe ratio, arguing that the WF backtest is historically accurate, while the CV backtest is a scenario simulation, or an inferential exercise. Is this selection bias?
'''
#这是选择性偏差，由于wf的结果方差更大，所以更容易出现貌似好的结果。cv一般来说模型更稳健。而只报告高夏普回测则肯定陷入了选择性偏差，甚至说过拟合。



'''
12.4 Your strategy produces 100,000 forecasts over time. You would like to derive the CPCVdistribution of Sharpe ratios by generating 1,000 paths. What are the possible combinations of parameters (N,k) that will allow you to achieve that?
'''
#数学公式：path[N,k]= k/N *（ “N 选 N−k”），(C  N N-K)
#求解数学公式的话可以有很多，但是常用的K=2，这时能够充分的使用训练集数据。此时，path=N-1
#所以当path=1000时，选择N=1001,K=2即可.

#注：AI分析path[N,k]= k/N *（ “N 选 N−k”）正整数解只有两组。（1001，2） 和（1001,1000） 有点巧合？不过反正k=2都固定死，只需要选择path数量即可。


'''
12.5 你发现了一个策略，在WF回测中实现了1.5的夏普比率。你写一篇论文来解释证明这个结果的理论，并把它提交给学术期刊。编辑回复说，一个裁判要求你使用N = 100和k = 2的CPCV方法重复你的回测，包括你的代码和完整的数据集。按照这些说明，夏普比率的平均值为- 1，标准差为0.5。愤怒的你没有回复，而是撤回了你的投稿，并在另一个影响因子更高的期刊上重新投稿。6个月后，你的论文被接受。你安抚自己的良心，认为如果发现是假的，那是期刊没有要求进行CPCV测试的错。你会想：“这不可能是不道德的，因为这是允许的，而且每个人都在这么做。”你的行为有什么科学或伦理上的理由？
'''

#毫无疑问的选择偏差。所以对于很高的回测结果，要求对方进行cpcv回测测试，这才是真实的样本外夏普比率。



'''
第十二章总结：
1.提到了比聚宽直接用历史数据回测一遍更好，更适应机器学习的回测方法。WF回测，通过训练窗口，验证窗口，滚动步长来动态调参，然后近可能的获取OOS结果。这样回测在数据利用上已经比聚宽模式好了，但是也提出了这样回测方式是具有缺陷的，于是进一步提出了CV回测方式。
2.CV回测，即将数据切分为N个训练窗口，选择N-1个为训练集，1个为测试集，进行k次循环计算。优点是充分考虑牛熊等多个市场状态，甚至是极端值，但是不是明确的历史路径解释，即重走一回不一定是这个收益。最终样本外决策可信度是比wf回测更高。但是仍旧只是基于历史数据单一路径进行的回测。
3.CPCV回测。通过对数据的切分和二次抽取，构建出多条随机缺失数据（拿去当测试集了）的回测路径，这样的样本外鲁棒性更强。注意使用清除和禁止来解决数据可能的未来函数问题。
4.在12.5通过数学上论证了cpcv方法比cv和wf方法具有更接近真实夏普效果。也提出了根据cpcv方法减少别人回测结果的偏差。所以要将别人的回测结果修正为正确的夏普结果，就得用cpcv方法。
5.截止202512，市面上貌似只有mlfinlab 这个库跟AFML是符合的，但是这个库是收费的。
6.wf可以用于初步的验证，计算量较少，cpcv用于最终的确定，统计特征更稳定。目标不是找出最好的回测结果，而是估计真实的样本外表现。类似于cpcv是统计特性，wf是历史中多种可能实际落地的一种。
7.还没有写cpcv的代码。待完成开发时进行。

'''


#%%

#第十三章 在模拟数据回测
from random import gauss
from itertools import product

# main()遍历了市场状态,然后batch（）是止盈止损，甚至是三重障碍的设置。因为不管是什么策略，都是用类似的止盈止损策略，比如2.5个标准差止盈止损之类的。而市场状态都是遍历了的，所以也不需要额外的设置了。
#选择好这些预定的参数后，再通过百万次路径的模拟，展示出相应的夏普比率，然后计算出对应较优的止盈止损。
def batch(coeffs, nIter=1e5, maxHP=100, rPT=np.linspace(0.5, 10, 20), rSLm=np.linspace(0.5, 10, 20), seed=0):
    """
    在合成的均值回归过程中模拟交易。

    参数:
    -----------
    coeffs : dict, 包含 'forecast' (长期趋势), 'hl' (半衰期), 'sigma' (连续时间的噪声标准差σ，而不仅仅是残差标准差) 
    nIter : int, 每个 (PT, SL) 参数对的蒙特卡洛模拟次数
    maxHP : int, 最大持仓周期（步数），超过则强制平仓
    rPT : array, 止盈阈值列表 (正值)  实际为1*噪声标准差，1.5*噪声标准差，2*噪声标准差等这样的假设
    rSLm : array, 止损幅度列表 (正值，实际应用为 -rSLm) 实际为-1*噪声标准差，-1.5*噪声标准差，-2*噪声标准差等这样的假设
    seed : float, 初始价格水平

    返回:
    --------
    列表，每个元素为元组 (PT, SL, 平均盈亏, 盈亏标准差, 夏普比率)
    """
    # 计算 AR(1) 过程的衰减因子 phi
    # 公式：phi = 2^(-1/hl)，hl 是半衰期
    # 这决定了价格向长期趋势增长的速度
    phi = 2 ** (-1.0 / coeffs['hl'])
    
    # 存储最终结果的列表
    output1 = []

    # 遍历所有止盈 (PT) 和止损 (SL) 的组合
    for comb_ in product(rPT, rSLm):
        # 存储当前 (PT, SL) 组合下，所有模拟的盈亏结果
        output2 = []
        
        # 进行 nIter 次蒙特卡洛模拟
        for iter_ in range(int(nIter)):
            # 初始化当前价格 p 为 seed (即入场价)
            p = seed
            # 持仓周期计数器
            hp = 0
            
            # 进入一个交易循环，直到满足退出条件
            while True:
                # 更新价格 p，使用 AR(1) 均值回归模型
                # p_{t} = (1-phi) * forecast + phi * p_{t-1} + sigma * 高斯噪声
                # 这个模型模拟了价格围绕 'forecast' 值长期趋势增长的特性
                p = (1 - phi) * coeffs['forecast'] + phi * p + coeffs['sigma'] * gauss(0, 1)
                
                # 计算当前盈亏 (cP)：当前价格 - 入场价 (seed)
                cP = p - seed
                # 持仓周期加一
                hp += 1

                # 检查退出条件：
                # 1. 达到止盈：cP >= PT
                # 2. 触发止损：cP <= -SL
                # 3. 超过最大持仓周期：hp > maxHP
                if cP >= comb_[0] or cP <= -comb_[1] or hp > maxHP:
                    # 将本次模拟的最终盈亏结果存入 output2
                    output2.append(cP)
                    # 跳出本次交易循环，开始下一次模拟
                    break

        # 计算当前 (PT, SL) 组合下，nIter 次模拟的统计结果
        mean_pnl = np.mean(output2)  # 平均盈亏
        std_pnl = np.std(output2)    # 盈亏标准差 (风险)
        # 夏普比率：单位风险的收益 (注意处理 std 为 0 的情况)
        sharpe_ratio = mean_pnl / std_pnl if std_pnl != 0 else 0.0

        # 打印当前参数组合的统计结果 (Python 3 需要 print() 函数)
        print(f"PT={comb_[0]:.2f}, SL={comb_[1]:.2f} | Mean={mean_pnl:.4f}, Std={std_pnl:.4f}, Sharpe={sharpe_ratio:.4f}")
        
        # 将当前 (PT, SL) 组合的结果元组添加到总结果列表中
        output1.append((comb_[0], comb_[1], mean_pnl, std_pnl, sharpe_ratio))

    # 返回所有 (PT, SL) 组合的统计结果列表
    return output1


def main():
    """
    主函数：遍历不同的市场参数 (forecast, hl)，对每个参数组合运行 batch 模拟

    """
    # 定义止盈和止损的搜索范围 (0 到 10，共 21 个点)
    rPT = rSLm = np.linspace(0, 10, 21)
    
    # 计数器，用于跟踪当前运行的是第几组参数
    count = 0
    all_outputs=[]
    
    # 遍历预测值 (forecast) 和半衰期 (hl) 的组合
    for forecast in [10, 5, 0, -5, -10]:
        for hl in [5, 10, 25, 50, 100]:
            count += 1
            # 构建当前参数字典
            coeffs = {'forecast': forecast, 'hl': hl, 'sigma': 1}
            
            print(f"\n--- 运行第 {count} 次: forecast={forecast}, hl={hl} ---")
            
            # 调用 batch 函数进行模拟
            # 注意：原代码中返回的 output 在循环内被覆盖，最后只返回最后一次的结果
            # 如果想保留所有结果，应将 output 添加到一个列表中
            output = batch(
                coeffs=coeffs,
                nIter=1e5,  # 模拟次数
                maxHP=100,  # 最大持仓周期
                rPT=rPT,    # 止盈范围
                rSLm=rSLm,  # 止损范围
                seed=0      # 初始价格
            )
            # 如果需要保留所有结果，可以这样修改：
            all_outputs.append((forecast, hl, output))
    
    # 原代码返回最后一次 batch 的结果
    return all_outputs

#根据价格序列估计 OU 过程的参数，即计算 'forecast' (长期趋势), 'hl' (半衰期), 'sigma'
#还有额外的方法：'yule_walker' : Yule-Walker 方法（对噪声更稳健），'mle' : 最大似然估计
#注意有个前提是价格序列是平稳的，否则估计的参数可能不准确，需要经过adf检验，判断是否平稳，对于股票价格，通常使用对数价格，因为价格p很难展示出长期趋势
def estimate_ou_parameters(price_series, dt=1/252, return_details=False):
    """
    估计离散型 Ornstein-Uhlenbeck 过程参数
    
    参数:
    -----------
    price_series : array-like
        价格序列，可以是 list, numpy array 或 pandas Series
    dt : float
        时间间隔（以年为单位）。默认 1/252 表示交易日
    return_details : bool
        是否返回详细统计结果
    
    返回:
    -----------
    dict: 包含以下键值:
        'forecast' : 长期均衡价格 (mu)
        'hl' : 半衰期（年）
        'sigma' : 连续 O-U 过程的波动率参数（年化）
        'beta_0', 'beta_1' : AR(1) 回归系数
        'theta' : 均值回归速度
        'residual_std' : 残差标准差
    """
    
    # 转换为 numpy 数组
    prices = np.array(price_series).flatten()
    
    if len(prices) < 10:
        raise ValueError("数据点太少，无法进行可靠估计")
    
    # 创建滞后序列
    P_t = prices[1:]      # 当前期
    P_t_1 = prices[:-1]   # 滞后一期
    
    # 执行 AR(1) 回归: P_t = beta_0 + beta_1 * P_{t-1} + epsilon
    # 添加常数项
    X = sm.add_constant(P_t_1)
    model = sm.OLS(P_t, X)
    results = model.fit()
    
    beta_0 = results.params[0]
    beta_1 = results.params[1]
    residual_std = np.std(results.resid)  # 残差标准差
    
    # 检查平稳性条件
    if beta_1 >= 1:
        print(f"警告: beta_1={beta_1:.4f} >= 1，过程可能非平稳")
        # 可以限制 beta_1 在合理范围
        beta_1 = min(0.999, beta_1)
    
    # 1. 计算长期均衡价格 (forecast)
    mu = beta_0 / (1 - beta_1)
    
    # 2. 计算半衰期
    # 连续时间均值回归速度: theta = -ln(beta_1)/dt
    theta = -np.log(beta_1) / dt
    hl = np.log(2) / theta  # 以年为单位
    
    # 3. 计算连续 O-U 过程的 sigma（年化）
    # 公式: sigma = sigma_epsilon * sqrt(2*theta/(1-exp(-2*theta*dt)))
    # 这里 sigma_epsilon 是残差的标准差
    sigma_epsilon = residual_std
    exponent = -2 * theta * dt
    sigma = sigma_epsilon * np.sqrt(2 * theta / (1 - np.exp(exponent)))
    
    # 整理结果
    result = {
        'forecast': mu,
        'hl': hl,
        'sigma': sigma,
        'beta_0': beta_0,
        'beta_1': beta_1,
        'theta': theta,
        'residual_std': residual_std,
        'r_squared': results.rsquared,
        't_stat_beta_1': results.tvalues[1]
    }
    
    if return_details:
        # 添加更多统计信息
        result['regression_results'] = results
        result['residuals'] = results.resid
        result['fitted_values'] = results.fittedvalues
    
    return result


'''
13.1 Suppose you are an execution trader. A client calls you with an order to cover a short position she entered at a price of 100. She gives you two exit conditions: profit-taking at 90 and stop-loss at 105.
(a) Assuming the client believes the price follows an O-U process, are these levels reasonable? For what parameters?
(b) Can you think of an alternative stochastic process under which these levels make sense?
'''

#a 需要对价差 log(P1) - log(P2) 或 P1 - beta*P2 进行平稳性检验（adf test） ，如果检验通过了，才能对数据进行 OU 过程的参数估计，获取长期均衡价格、半衰期、波动率等参数。
#假设已经是平稳的，这个退出条件不是最优的，因为这是高止盈低止损，大多数市场下都是高止损，中低止盈。不管长期均衡价格是正数，负数还是零。

#客户认为价格是均值回归，而且长期均值价格要低于90，这样止盈才有意义。
#均值回归速率 ，这决定了价格从 100 回归到 μ 的速度有多快————————这个是提高单位时间交易频率的好东西，揭示了等待的时间有多久。
#波动率 σ ，形容价格的震荡幅度。在这里比较影响止损。所以回归的速度和这个随机震荡还不太一样。


#b 其他适用于这样止盈止损的统计模型：
#1. 几何布朗运动（GBM）+ 趋势 (Drift)：价格呈指数增长或下降趋势，但同时伴有随机波动。其动态由漂移率 μ （趋势）和波动率 决定。这样的统计模型就适用于单边上涨/下跌的行情，也可以按照这样的统计模型对止盈止损进行开发
#2. 布朗运动+价格跳跃：跳跃扩散允许非连续的、突发的大幅价格变动
#3. 带漂移的随机游走 ：GBM 的离散版本。价格在每个时间步长都以一定的概率向上或向下移动一个固定步长，但总体上有一个偏向（漂移）。

'''
13.2 Fit the time series of dollar bars of E-mini S&P 500 futures to an O-U process.
Given those parameters:
(a) Produce a heat-map of Sharpe ratios for various profit-taking and stop-loss levels.
(b) What is the OTR?  即最优止盈止损条件。
'''

#这里拿中证2000 作为例子进行计算。 但是找不到，找某个股票的5min级别数据来进行。赣锋锂业 (002460)
import pandas as pd
import baostock as bs

lg = bs.login()

stock_code = "sz.002460"  # 股票代码，格式为 "市场.代码"，例如 sh.600000 (浦发银行) 或 sz.000001 (平安银行)
start_date = "2025-01-06" # 开始日期，格式 YYYY-MM-DD
end_date = "2025-01-06"   # 结束日期，格式 YYYY-MM-DD (可以是同一天获取当天数据)
frequency = "5"           # 数据频率：'d' for day, 'w' for week, 'm' for month, '5' for 5min, '15' for 15min, '30' for 30min, '60' for 60min
adjustflag = "3"          # 复权标志：'3' for 不复权, '2' for 后复权, '1' for 前复权

# 2. 调用查询函数
rs = bs.query_history_k_data_plus(stock_code,
                                  "date,time,code,open,high,low,close,volume,amount,adjustflag", # 指定要查询的字段
                                  start_date=start_date,
                                  end_date=end_date,
                                  frequency=frequency,
                                  adjustflag=adjustflag)

if rs.error_code != '0':
    print(f"Query failed. Error code: {rs.error_code}, Error message: {rs.error_msg}")
else:
    print("Query succeeded. Fetching data...")

 # 4. 循环读取数据并存入列表
data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())

# 5. 将列表转换为 pandas DataFrame
result = pd.DataFrame(data_list, columns=rs.fields)

#time 的格式转换。20250106133500000转为 2025-01-06 13：35：00 000 年月日 时分秒格式
result['time']=pd.to_datetime(result['time'], format='%Y%m%d%H%M%S%f')

bs.logout()




'''
13.3 Repeat exercise 2, this time on a time series of dollar bars of
(a) 10-year U.S. Treasure Notes futures
(b) WTI Crude Oil futures
(c) Are the results significantly different? Does this justify having execution
traders specialized by product?
'''

'''
13.4 Repeat exercise 2 after splitting the time series into two parts:
(a) The first time series ends on 3/15/2009.
(b) The second time series starts on 3/16/2009.
(c) Are the OTRs significantly different?
'''

'''
13.5 How long do you estimate it would take to derive OTRs on the 100 most liquid
futures contracts worldwide? Considering the results from exercise 4, how often
do youthink you mayhavetore-calibrate the OTRs? Does it make sense to pre
compute this data?
'''


'''
13.6 Parallelize Snippets 13.1 and 13.2 using the mpEngine module described in
Chapter 20.

'''


'''
第十三章总结：
1.提出了如何根据原数据构建止盈止损条件。能够在数学上证明的，不是基于回测的。（回测不是研究手段，用回测实现止盈止损有很大的过拟合嫌疑）。也就是能够使用在三重障碍法的上下障碍的止盈止损条件，类似。
2.通过实验得到了对应市场状态（forecast（趋势值）, hl（半衰期））下的最优止盈止损条件。状态，只需要判断市场是什么样的市场和半衰期就可以根据本章的batch函数，得到最优的止盈止损条件。或者按照课本去对应出来已有的结果，也能出个大概。
3.通过将价格平稳化后可以计算OU过程的参数，即长期均衡价格 (forecast)，半衰期 (hl)，连续 O-U 过程的波动率参数（年化） (sigma)。而且这就与前面数据处理的过程联系起来，要使得数据平稳，正态，才富有统计意义。———————— 一般来说，需要使用对数价格，对数收益率来进行adf检验，因为一般来说，价格是很少能一阶平稳的！！！！
4.基于不同的统计模型，可以开发不同的止盈止损条件。比如，价格呈指数增长或下降趋势时，适合使用几何布朗运动（GBM）+ 趋势 (Drift) 模型。



'''
