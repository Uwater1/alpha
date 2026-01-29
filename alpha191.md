# alpha191因子 - JoinQuant

[](/ "joinQuant")

-   [短周期价量特征 191 Alphas 因子函数使用说明](#短周期价量特征-191-alphas-因子函数使用说明 "短周期价量特征 191 Alphas 因子函数使用说明")
-   [因子来源](#因子来源 "因子来源")
-   [因子使用](#因子使用 "因子使用")
-   [公用函数说明](#公用函数说明 "公用函数说明")
-   [因子说明](#因子说明 "因子说明")
-   [alpha(获取全部因子值)](#alpha获取全部因子值 "alpha(获取全部因子值)")
-   [alpha\_001](#alpha001 "alpha_001")
-   ...
-   [alpha\_191](#alpha191 "alpha_191")

## 短周期价量特征 191 Alphas 因子函数使用说明

### 因子来源

通过交易型阿尔法策略的研究，发现在 A 股市场，与传统多因子模型所获取的股票价值阿尔法收益相比，交易型阿尔法收益的空间更大、收益稳定性也更强。

短周期交易型阿尔法体系既是对传统多因子体系的补充，也可以说是全新思路、独立设计的交易体系。在这其中，量化模型不再仅仅是低风险低收益的投资策略，同样也可获得高额的收益回报。

JoinQuant聚宽（专业的量化投研平台）旨在为大家提供更多的投资思路及可使用的数据，因此我们根据国泰君安数量化专题研究报告 - [_基于短周期价量特征的多因子选股体系_](见末尾附录)给出了 191 个短周期交易型阿尔法因子，方便大家使用。

其中因子数据则均来自于个股日频率的价格与成交量数据，并且在编写短周期交易型 Alpha191因子时，有对缺失部分和不合理部分的因子公式进行调整。

我们初衷是想为大家提供更多的投资思路及可使用的数据。至于这些因子如何使用能达到策略最佳收益，或者说这些因子是否适用于A股市场等问题，还需要大家自己去研究与钻研。

**注：在对编写 alpha191因子时，有对缺失部分和不合理部分的因子公式进行调整**

### 因子使用

```python
# 导入 Alpha191 库
>>> from jqlib.alpha191 import *

# 获取沪深300成分股的 alpha_001 因子值
>>>end_date = '2017-03-10'
>>>code= list(get_index_stocks('000300.XSHG'))
>>> a = alpha_001(code,end_date)

# 查看前5行的因子值
>>> a.head()
000001.XSHE   -0.496667
000002.XSHE    0.226667
000008.XSHE   -0.043333
000009.XSHE   -0.093333
000027.XSHE   -0.030000
Name: rank_value_boolean, dtype: float64

# 查看平安银行的因子值
>>> a['000001.XSHE']
-0.067912591286056825

# 获取所有股票 alpha_007 的因子值
>>>end_date = '2017-04-04'
>>>code = list(get_all_securities(['stock'],date=end_date).index)
>>> a = alpha_007(code,end_date)
# 查看欣旺达(300207)的因子值
>>> a['300207.XSHE']
1.2494895018526142

# 查询函数说明
>>> alpha_001?
Signature: alpha_001(code, end_date=None)
Docstring:
公式：
    (-1 * CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
Inputs:
    code: 股票池
    end_date: 查询日期
Outputs:
    因子的值
File:      ~/alpha191.py
Type:      function
```

### 公用函数说明

**股票的默认获取时间长度为截止日期的前350个交易日**

OPEN  
_开盘价_

HIGH  
_最高价_

LOW  
_最低价_

CLOSE  
_收盘价_

VWAP  
_均价_

VOLUME  
_成交量_

AMOUNT  
_成交额_

BANCHMARKINDEXCLOSE  
_基准指数的开盘价_

BANCHMARKINDEXOPEN  
_基准指数的收盘价_

RET  
_每日收益率（收盘/前收盘-1）_

DTM  
_(OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))_

DBM  
_(OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))_

TR  
_MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))_

HD  
_HIGH-DELAY(HIGH,1)_

LD  
_DELAY(LOW,1)-LOW_

HML SMB MKE  
_Fama French 三因子_

SELF  
_特殊变量，出现在 Alpha143，表示 t-1 日的 Alpha143 因子计算结果_

RANK(A)  
_向量 A 升序排序_

MAX(A, B)  
_在 A,B 中选择最大的数_

MIN(A, B)  
_在 A,B 中选择最小的数_

STD(A, n)  
_序列 A 过去 n 天标准差_

CORR(A, B, n)  
_序列 A、 B 过去 n 天相关系数_

DELTA(A, n)  

LOG(A)  
_自然对数函数_

SUM(A, n)  
_序列 A 过去 n 天求和_

ABS(A)  
_绝对值函数_

MEAN(A, n)  
_序列 A 过去 n 天均值_

TSRANK(A, n)  
_序列 A 的末位值在过去 n 天的顺序排位_

SIGN(A)  
_符号函数：1 if A>0; o if A=0 ; -1 if A \\<0;_

COVIANCE (A, B, n)  
_序列 A、 B 过去 n 天协方差_

DELAY(A, n)  

TSMIN(A, n)  
_序列 A 过去 n 天的最小值_

TSMAX(A, n)  
_序列 A 过去 n 天的最大值_

PROD(A, n)  
_序列 A 过去 n 天累乘_

COUNT(condition, n)  
_计算前 n 期满足条件 condition 的样本个数_

REGBETA(A, B, n)  
_前 n 期样本 A 对 B 做回归所得回归系数_

REGRESI(A, B, n)  
_前 n 期样本 A 对 B 做回归所得的残差_

SMA(A, n, m)

，其中 Yˆ 表示最终结果

SUMIF(A, n, condition)  
_对 A 前 n 项条件求和，其中 condition 表示选择条件_

WMA(A, n)  
_计算 A前 n期样本加权平均值权重为 0.9i，(i 表示样本距离当前时点的间隔)_

DECAYLINEAR(A, d)  
_对 A 序列计算移动平均加权，其中权重对应 d,d-1,…,1（权重和为 1）_

FILTER(A, condition)  
_对 A 筛选出符合选择条件 condition 的样本_

HIGHDAY(A, n)  
_计算 A 前 n 期时间序列中最大值距离当前时点的间隔_

LOWDAY(A, n)  
_计算 A 前 n 期时间序列中最小值距离当前时点的间隔_

SEQUENCE(n)  
_生成 1~n 的等差序列_

SUMAC(A, n)  
_计算 A 的前 n 项的累加_

&  
_逻辑运算与_

||  
_逻辑运算或_

A?B:C  
_若 A 成立，则为 B，否则为 C_

### 因子说明

**在对编写 alpha191因子时，有对缺失部分和不合理部分的因子公式进行调整**

#### alpha(获取全部因子值)

获取标的191个全部因子值，因计算量较大，运行会有少许缓慢

```python
alpha(code,benchmark='000300.XSHG',end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 DataFrame，包括股票列表中每一只股票的 alpha001-alpha191 的值

#### alpha\_001

```python
alpha_001(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1 \* CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)

(对应WQ alpha002)

#### alpha\_002

```python
alpha_002(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    \-1 \* delta((((close-low)-(high-close))/((high-low)),1))

#### alpha\_003

```python
alpha_003(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)

#### alpha\_004

```python
alpha_004(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?(-1\*1):(((SUM(CLOSE,2)/2)<((SUM(CLOSE,8)/8)-STD(CLOSE,8)))?1:(((1<(VOLUME/MEAN(VOLUME,20)))||((VOLUME/MEAN(VOLUME,20))==1))?1:(-1\*1))))

#### alpha\_005

```python
alpha_005(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*TSMAX(CORR(TSRANK(VOLUME,5),YSRANK(HIGH,5),5),3))

#### alpha\_006

```python
alpha_006(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(SIGN(DELTA((((OPEN\*0.85)+(HIGH\*0.15))),4)))\*-1)

#### alpha\_007

```python
alpha_007(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((RANK(MAX((VWAP-CLOSE),3))+RANK(MIN((VWAP-CLOSE),3)))\*RANK(DELTA(VOLUME,3)))

#### alpha\_008

```python
alpha_008(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   RANK(DELTA(((((HIGH+LOW)/2)_0.2)+(VWAP\*0.8)),4)_\-1

#### alpha\_009

```python
alpha_009(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/\*(HIGH-LOW)/VOLUME,7，2)

#### alpha\_010

```python
alpha_010(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(MAX(((RET\\<0)?STD(RET,20):CLOSE)^2),5))

#### alpha\_011

```python
alpha_011(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).\*VOLUME,6)

#### alpha\_012

```python
alpha_012(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK((OPEN-(SUM(VWAP,10)/10))))_(-1_(RANK(ABS((CLOSE-VWAP)))))

#### alpha\_013

```python
alpha_013(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (((HIGH\*LOW)^0.5)-VWAP)

#### alpha\_014

```python
alpha_014(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   CLOSE-DELAY(CLOSE,5)

#### alpha\_015

```python
alpha_015(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   OPEN/DELAY(CLOSE,1)-1

#### alpha\_016

```python
alpha_016(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5))

#### alpha\_017

```python
alpha_017(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   RANK((VWAP-MAX(VWAP,15)))^DELTA(CLOSE,5)

#### alpha\_018

```python
alpha_018(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   CLOSE/DELAY(CLOSE,5)

#### alpha\_019

```python
alpha_019(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))

#### alpha\_020

```python
alpha_020(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)\*100

#### alpha\_021

```python
alpha_021(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   REGBETA(MEAN(CLOSE,6),SEQUENCE(6))

#### alpha\_022

```python
alpha_022(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)

#### alpha\_023

```python
alpha_023(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))\*100

#### alpha\_024

```python
alpha_024(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(CLOSE-DELAY(CLOSE,5),5,1)

#### alpha\_025

```python
alpha_025(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((-1\*RANK((DELTA(CLOSE,7)_(1-RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME,20)),9))))))_(1+RANK(SUM(RET,250))))

#### alpha\_026

```python
alpha_026(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((((SUM(CLOSE,7)/7)-CLOSE))+((CORR(VWAP,DELAY(CLOSE,5),230))))

#### alpha\_027

```python
alpha_027(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)\*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)\*100,12)

#### alpha\_028

```python
alpha_028(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   3\*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))\*100,3,1)-2\*SMA(SMA((CLOSE-TSMIN(LOW,9))/( MAX(HIGH,9)-TSMAX(LOW,9))\*100,3,1),3,1)

#### alpha\_029

```python
alpha_029(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)\*VOLUME

#### alpha\_030（尚未实现）

```python
alpha_030(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML，60))^2,20)

#### alpha\_031

```python
alpha_031(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   LOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)\*100

#### alpha\_032

```python
alpha_032(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*SUM(RANK(CORR(RANK(HIGH),RANK(VOLUME),3)),3))

#### alpha\_033

```python
alpha_033(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((((-1\*TSMIN(LOW,5))+DELAY(TSMIN(LOW,5),5))\*RANK(((SUM(RET,240)-SUM(RET,20))/220)))\*TSRANK(VOLUME,5))

#### alpha\_034

```python
alpha_034(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   MEAN(CLOSE,12)/CLOSE

#### alpha\_035

```python
alpha_035(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)),RANK(DECAYLINEAR(CORR((VOLUME),((OPEN\*0.65)+(OPEN\*0.35)),17),7)))\*-1)

#### alpha\_036

```python
alpha_036(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   RANK(SUM(CORR(RANK(VOLUME),RANK(VWAP)),6),2)

#### alpha\_037

```python
alpha_037(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*RANK(((SUM(OPEN,5)\*SUM(RET,5))-DELAY((SUM(OPEN,5)\*SUM(RET,5)),10))))

#### alpha\_038

```python
alpha_038(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (((SUM(HIGH,20)/20)<HIGH)?(-1\*DELTA(HIGH,2)):0)

#### alpha\_039

```python
alpha_039(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((RANK(DECAYLINEAR(DELTA((CLOSE),2),8))-RANK(DECAYLINEAR(CORR(((VWAP\*0.3)+(OPEN\*0.7)),SUM(MEAN(VOLUME,180),37),14),12)))\*-1

#### alpha\_040

```python
alpha_040(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)\*100

#### alpha\_041

```python
alpha_041(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(MAX(DELTA((VWAP),3),5))\*-1)

#### alpha\_042

```python
alpha_042(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*RANK(STD(HIGH,10)))\*CORR(HIGH,VOLUME,10))

#### alpha\_043

```python
alpha_043(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)

#### alpha\_044

```python
alpha_044(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (TSRANK(DECAYLINEAR(CORR(((LOW)),MEAN(VOLUME,10),7),6),4)+TSRANK(DECAYLINEAR(DELTA((VWAP),3),10),15))

#### alpha\_045

```python
alpha_045(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(DELTA((((CLOSE\*0.6)+(OPEN\*0.4))),1))\*RANK(CORR(VWAP,MEAN(VOLUME,150),15)))

#### alpha\_046

```python
alpha_046(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4\*CLOSE)

#### alpha\_047

```python
alpha_047(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))\*100,9,1)

#### alpha\_048

```python
alpha_048(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*((RANK(((SIGN((CLOSE-DELAY(CLOSE,1)))+SIGN((DELAY(CLOSE,1)-DELAY(CLOSE,2))))+SIGN((DELAY(CLOSE,2)-DELAY(CLOSE,3))))))\*SUM(VOLUME,5))/SUM(VOLUME,20))

#### alpha\_049

```python
alpha_049(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))

#### alpha\_050

```python
alpha_050(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0: MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELA Y(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))

#### alpha\_051

```python
alpha_051(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：
    
    -   SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    
    #### alpha\_052
    

```python
alpha_052(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)\*100

#### alpha\_053

```python
alpha_053(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   COUNT(CLOSE>DELAY(CLOSE,1),12)/12\*100

#### alpha\_054

```python
alpha_054(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*RANK((STD(ABS(CLOSE-OPEN))+(CLOSE-OPEN))+CORR(CLOSE,OPEN,10)))

#### alpha\_055

```python
alpha_055(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM(16\*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CL OSE,1))>ABS(LOW-DELAY(CLOSE,1))&ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))&ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))\*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)

#### alpha\_056

```python
alpha_056(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK((OPEN-TSMIN(OPEN,12)))<RANK((RANK(CORR(SUM(((HIGH +LOW)/2),19),SUM(MEAN(VOLUME,40),19),13))^5)))

#### alpha\_057

```python
alpha_057(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))\*100,3,1)

#### alpha\_058

```python
alpha_058(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   COUNT(CLOSE>DELAY(CLOSE,1),20)/20\*100

#### alpha\_059

```python
alpha_059(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)

#### alpha\_060

```python
alpha_060(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).\*VOLUME,20)

#### alpha\_061

```python
alpha_061(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80),8)),17)))\*-1)

#### alpha\_062

```python
alpha_062(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*CORR(HIGH,RANK(VOLUME),5))

#### alpha\_063

```python
alpha_063(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)\*100

#### alpha\_064

```python
alpha_064(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE),RANK(MEAN(VOLUME,60)),4),13),14)))\*-1)

#### alpha\_065

```python
alpha_065(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   MEAN(CLOSE,6)/CLOSE

#### alpha\_066

```python
alpha_066(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)\*100

#### alpha\_067

```python
alpha_067(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)\*100

#### alpha\_068

```python
alpha_068(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)\*(HIGH-LOW)/VOLUME,15,2)

#### alpha\_069

```python
alpha_069(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (SUM(DTM,20)>SUM(DBM,20)?(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):(SUM(DTM,20)=SUM(DBM,20)？0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))

#### alpha\_070

```python
alpha_070(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   STD(AMOUNT,6)

#### alpha\_071

```python
alpha_071(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)\*100

#### alpha\_072

```python
alpha_072(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))\*100,15,1)

#### alpha\_073

```python
alpha_073(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE),VOLUME,10),16),4),5)-RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3)))\*-1)

#### alpha\_074

```python
alpha_074(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(CORR(SUM(((LOW\*0.35)+(VWAP\*0.65)),20),SUM(MEAN(VOLUME,40),20),7))+RANK(CORR(RANK(VWAP),RANK(VOLUME),6)))

#### alpha\_075

```python
alpha_075(code,benchmark='000300.XSHG',end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
    -   benchmark:基准指数，默认为沪深300
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKIN DEXOPEN,50)

#### alpha\_076

```python
alpha_076(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)

#### alpha\_077

```python
alpha_077(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   MIN(RANK(DECAYLINEAR(((((HIGH+LOW)/2)+HIGH)-(VWAP+HIGH)),20)),RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),MEAN(VOLUME,40),3),6)))

#### alpha\_078

```python
alpha_078(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015\*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))

#### alpha\_079

```python
alpha_079(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)\*100

#### alpha\_080

```python
alpha_080(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)\*100

#### alpha\_081

```python
alpha_081(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(VOLUME,21,2)

#### alpha\_082

```python
alpha_082(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))\*100,20,1)

#### alpha\_083

```python
alpha_083(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*RANK(COVIANCE(RANK(HIGH),RANK(VOLUME),5)))

#### alpha\_084

```python
alpha_084(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)

#### alpha\_085

```python
alpha_085(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (TSRANK((VOLUME/MEAN(VOLUME,20)),20)\*TSRANK((-1\*DELTA(CLOSE,7)),8))

#### alpha\_086

```python
alpha_086(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((0.25<(((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10)-((DELAY(CLOSE,10)-CLOSE)/10)))?(-1\*1):(((((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10)-((DELAY(CLOSE,10)-CLOSE)/10))\\<0)?1:((-1\*1)\*(CLOSE-DELAY(CLOSE,1)))))

#### alpha\_087

```python
alpha_087(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((RANK(DECAYLINEAR(DELTA(VWAP,4),7))+TSRANK(DECAYLINEAR(((((LOW\*0.9)+(LOW\*0.1))-VWAP)/(OPEN-((HIGH+LOW)/2))),11),7))\*-1)

#### alpha\_088

```python
alpha_088(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)\*100

#### alpha\_089

```python
alpha_089(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   2\*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))

#### alpha\_090

```python
alpha_090(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(CORR(RANK(VWAP),RANK(VOLUME),5))\*-1)

#### alpha\_091

```python
alpha_091(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((RANK((CLOSE-MAX(CLOSE,5)))_RANK(CORR((MEAN(VOLUME,40)),LOW,5)))_\-1)

#### alpha\_092

```python
alpha_092(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE\*0.35)+(VWAP\*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))\*-1)

#### alpha\_093

```python
alpha_093(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)

#### alpha\_094

```python
alpha_094(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)

#### alpha\_095

```python
alpha_095(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   STD(AMOUNT,20)

#### alpha\_096

```python
alpha_096(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))\*100,3,1),3,1)

#### alpha\_097

```python
alpha_097(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   STD(VOLUME,10)

#### alpha\_098

```python
alpha_098(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))\\<0.05)||((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))==0.05))?(-1\*(CLOSE-TSMIN(CLOSE,100))):(-1\*DELTA(CLOSE,3)))

#### alpha\_099

```python
alpha_099(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*RANK(COVIANCE(RANK(CLOSE),RANK(VOLUME),5)))

#### alpha\_100

```python
alpha_100(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   STD(VOLUME,20)

#### alpha\_101

```python
alpha_101(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))\\

#### alpha\_102

```python
alpha_102(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)\*100

#### alpha\_103

```python
alpha_103(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((20-LOWDAY(LOW,20))/20)\*100

#### alpha\_104

```python
alpha_104(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*(DELTA(CORR(HIGH,VOLUME,5),5)\*RANK(STD(CLOSE,20))))

#### alpha\_105

```python
alpha_105(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*CORR(RANK(OPEN),RANK(VOLUME),10))

#### alpha\_106

```python
alpha_106(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   CLOSE-DELAY(CLOSE,20)

#### alpha\_107

```python
alpha_107(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (((-1\*RANK((OPEN-DELAY(HIGH,1))))\*RANK((OPEN-DELAY(CLOSE,1))))\*RANK((OPEN-DELAY(LOW,1))))

#### alpha\_108

```python
alpha_108(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((RANK((HIGH-MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))\*-1)

#### alpha\_109

```python
alpha_109(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)

#### alpha\_110

```python
alpha_110(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)\*100

#### alpha\_111

```python
alpha_111(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(VOL\*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL\*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)

#### alpha\_112

```python
alpha_112(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)\\<0?ABS(CLOS E-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DE LAY(CLOSE,1)\\<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))\*100

#### alpha\_113

```python
alpha_113(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*((RANK((SUM(DELAY(CLOSE,5),20)/20))\*CORR(CLOSE,VOLUME,2))\*RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))))

#### alpha\_114

```python
alpha_114(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((RANK(DELAY(((HIGH-LOW)/(SUM(CLOSE,5)/5)),2))\*RANK(RANK(VOLUME)))/(((HIGH-LOW)/(SUM(CLOSE,5)/5))/(VWAP-CLOSE)))

#### alpha\_115

```python
alpha_115(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(CORR(((HIGH\*0.9)+(CLOSE\*0.1)),MEAN(VOLUME,30),10))^RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7)))

#### alpha\_116

```python
alpha_116(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   REGBETA(CLOSE,SEQUENCE,20)

#### alpha\_117

```python
alpha_117(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((TSRANK(VOLUME,32)_(1-TSRANK(((CLOSE+HIGH)-LOW),16)))_(1-TSRANK(RET,32)))

#### alpha\_118

```python
alpha_118(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)\*100

#### alpha\_119

```python
alpha_119(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))

#### alpha\_120

```python
alpha_120(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK((VWAP-CLOSE))/RANK((VWAP+CLOSE)))

#### alpha\_121

```python
alpha_121(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((RANK((VWAP-MIN(VWAP,12)))^TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3))\*-1)

#### alpha\_122

```python
alpha_122(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)

#### alpha\_123

```python
alpha_123(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((RANK(CORR(SUM(((HIGH+LOW)/2),20),SUM(MEAN(VOLUME,60),20),9))\\

#### alpha\_124

```python
alpha_124(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (CLOSE-VWAP)/DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)

#### alpha\_125

```python
alpha_125(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(DECAYLINEAR(CORR((VWAP),MEAN(VOLUME,80),17),20))/RANK(DECAYLINEAR(DELTA(((CLOSE\*0.5)+(VWAP\*0.5)),3),16)))

#### alpha\_126

```python
alpha_126(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (CLOSE+HIGH+LOW)/3

#### alpha\_127

```python
alpha_127(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (MEAN((100\*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)

#### alpha\_128

```python
alpha_128(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3\*VOLUM E:0),14)/SUM(((HIGH+LOW+CLOSE)/3\\

#### alpha\_129

```python
alpha_129(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM((CLOSE-DELAY(CLOSE,1)\\<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)

#### alpha\_130

```python
alpha_130(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),MEAN(VOLUME,40),9),10))/RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3)))

#### alpha\_131

```python
alpha_131(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(DELAT(VWAP,1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18))

#### alpha\_132

```python
alpha_132(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   MEAN(AMOUNT,20)

#### alpha\_133

```python
alpha_133(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((20-HIGHDAY(HIGH,20))/20)\*100-((20-LOWDAY(LOW,20))/20)\*100

#### alpha\_134

```python
alpha_134(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)\*VOLUME

#### alpha\_135

```python
alpha_135(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    \-SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)

#### alpha\_136

```python
alpha_136(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((-1\*RANK(DELTA(RET,3)))\*CORR(OPEN,VOLUME,10))

#### alpha\_137

```python
alpha_137(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   16\*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) &ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))\*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))

#### alpha\_138

```python
alpha_138(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((RANK(DECAYLINEAR(DELTA((((LOW\*0.7)+(VWAP\*0.3))),3),20))-TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW,8),TSRANK(MEAN(VOLUME,60),17),5),19),16),7))\* -1)

#### alpha\_139

```python
alpha_139(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*CORR(OPEN,VOLUME,10))

#### alpha\_140

```python
alpha_140(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：
-   MIN(RANK(DECAYLINEAR(((RANK(OPEN)+RANK(LOW))-(RANK(HIGH)+RANK(CLOSE))),8)),TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20),8),7),3))

#### alpha\_141

```python
alpha_141(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9))\*-1)

#### alpha\_142

```python
alpha_142(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (((-1\*RANK(TSRANK(CLOSE,10)))\*RANK(DELTA(DELTA(CLOSE,1),1)))\*RANK(TSRANK((VOLUME/MEAN(VOLUME,20)),5)))

#### alpha\_143（尚未实现）

```python
alpha_143(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)\*SELF:SELF

#### alpha\_144

```python
alpha_144(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)

#### alpha\_145

```python
alpha_145(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)\*100

#### alpha\_146

```python
alpha_146(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)\*(( CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOS E-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,60);

#### alpha\_147

```python
alpha_147(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   REGBETA(MEAN(CLOSE,12),SEQUENCE(12))

#### alpha\_148

```python
alpha_148(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((RANK(CORR((OPEN),SUM(MEAN(VOLUME,60),9),6))<RANK((OPEN-TSMIN(OPEN,14))))\*-1)

#### alpha\_149

```python
alpha_149(code,benchmark='000300.XSHG',end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
    -   benchmark:基准指数，默认为沪深300
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)

#### alpha\_150

```python
alpha_150(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (CLOSE+HIGH+LOW)/3\*VOLUME

#### alpha\_151

```python
alpha_151(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(CLOSE-DELAY(CLOSE,20),20,1)

#### alpha\_152

```python
alpha_152(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY (CLOSE,9),1),9,1),1),26),9,1)

#### alpha\_153

```python
alpha_153(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4

#### alpha\_154

```python
alpha_154(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18)))

#### alpha\_155

```python
alpha_155(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)

#### alpha\_156

```python
alpha_156(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN\*0.15)+(LOW\*0.85)),2)/((OPEN\*0.15)+(LOW\*0.85)))_\-1),3)))_\-1)

#### alpha\_157

```python
alpha_157(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1\*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5) +TSRANK(DELAY((-1\*RET),6),5))

#### alpha\_158

```python
alpha_158(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE

#### alpha\_159

```python
alpha_159(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)\*12\*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CL OSE,1)),12)\*6\*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,D ELAY(CLOSE,1)),24)\*6\*24)\*100/(6\*12+6\*24+12\*24)

#### alpha\_160

```python
alpha_160(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)

#### alpha\_161

```python
alpha_161(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)

#### alpha\_162

```python
alpha_162(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)\*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)\*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)\*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12, 1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)\*100,12))

#### alpha\_163

```python
alpha_163(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   RANK(((((-1\*RET)_MEAN(VOLUME,20))\*VWAP)_(HIGH-CLOSE)))

#### alpha\_164

```python
alpha_164(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-D ELAY(CLOSE,1)):1),12))/(HIGH-LOW)\*100,13,2)

#### alpha\_165（尚未实现）

```python
alpha_165(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)

#### alpha\_166

```python
alpha_166(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   \-20\*(20-1)^1.5\*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)\*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)

#### alpha\_167

```python
alpha_167(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12

#### alpha\_168

```python
alpha_168(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (-1\*VOLUME/MEAN(VOLUME,20))

#### alpha\_169

```python
alpha_169(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1), 26),10,1)

#### alpha\_170

```python
alpha_170(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((((RANK((1/CLOSE))_VOLUME)/MEAN(VOLUME,20))_((HIGH\*RANK((HIGH-CLOSE)))/(SUM(HIGH,5)/5)))-RANK((VWAP-DELAY(VWAP,5))))

#### alpha\_171

```python
alpha_171(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((-1\*((LOW-CLOSE)_(OPEN^5)))/((CLOSE-HIGH)_(CLOSE^5)))

#### alpha\_172

```python
alpha_172(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)\*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)\*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)\*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)\*100/SUM(TR,14))\*100,6)

#### alpha\_173

```python
alpha_173(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   3\*SMA(CLOSE,13,2)-2\*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2);

#### alpha\_174

```python
alpha_174(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)

#### alpha\_175

```python
alpha_175(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)

#### alpha\_176

```python
alpha_176(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   CORR(RANK(((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12)))),RANK(VOLUME),6)

#### alpha\_177

```python
alpha_177(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((20-HIGHDAY(HIGH,20))/20)\*100

#### alpha\_178

```python
alpha_178(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)\*VOLUME

#### alpha\_179

```python
alpha_179(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(CORR(VWAP,VOLUME,4))\*RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12)))

#### alpha\_180

```python
alpha_180(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((MEAN(VOLUME,20)<VOLUME)?((-1\*TSRANK(ABS(DELTA(CLOSE,7)),60))\*SIGN(DELTA(CLOSE,7)):(-1\*VOLUME)))

#### alpha\_181

```python
alpha_181(code,benchmark='000300.XSHG',end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
    -   benchmark:基准指数，默认为沪深300
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)

#### alpha\_182

```python
alpha_182(code,benchmark='000300.XSHG',end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
    -   benchmark:基准指数，默认为沪深300
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   COUNT((CLOSE>OPEN&BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN&BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20

#### alpha\_183（尚未实现）

```python
alpha_183(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)

#### alpha\_184

```python
alpha_184(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (RANK(CORR(DELAY((OPEN-CLOSE),1),CLOSE,200))+RANK((OPEN-CLOSE)))

#### alpha\_185

```python
alpha_185(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   RANK((-1\*((1-(OPEN/CLOSE))^2)))

#### alpha\_186

```python
alpha_186(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   (MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)\*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)\*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)\*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)\*100/SUM(TR,14))\*100,6)+DELAY(MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)\*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)\*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)\*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)\*100/SUM(TR,14))\*100,6),6))/2

#### alpha\_187

```python
alpha_187(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)

#### alpha\_188

```python
alpha_188(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))\*100

#### alpha\_189

```python
alpha_189(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)

#### alpha\_190

```python
alpha_190(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：（公式有部分缺失 有调整）
    
    -   原公式:
        
        -   LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)_(SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))_(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))
    -   修改后公式:
        
        -   LOG((COUNT(CLOSE/DELAY(CLOSE,1)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)_(SUMIF(((CLOSE/DELAY(CLOSE,1)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE,1)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))_(SUMIF((CLOSE/DELAY(CLOSE,1)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))

#### alpha\_191

```python
alpha_191(code,end_date=None)
```

-   输入：  
    -   code: 股票代码列表
    -   end\_date: 计算哪一天的因子 ,默认为None也就是最近一交易日的日期
-   输出：  
    -   一个 Series：index 为成分股代码，values 为对应的因子值
-   因子公式：  
    -   ((CORR(MEAN(VOLUME,20),LOW,5)+((HIGH+LOW)/2))-CLOSE)

 

