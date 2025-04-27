#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : nickdecodes
@Email   : nickdecodes@163.com
@Usage   :
@FileName: model_util.py
@DateTime: 2024/7/22 10:06
@SoftWare: PyCharm
模型工具模块 model_util.py
提供多种回归、预测及特征变换方法，包括 EMA、线性回归、随机森林、RSI、SARIMA 等。
"""

import os
from tempfile import template

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Dict, Any
from .calculate_util import CalculateUtil


class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    自定义特征转换器，用于随机森林管道中：
      - 对输入 X 做标准化；
      - 使用指定 RandomForestRegressor 进行拟合与预测；
      - 计算每行的标准差与 RSI；
      - 输出 [最后元素, 预测值, 标准差, RSI] 组合特征。
    """
    def __init__(self, model: RandomForestRegressor):
        self.calculate_util = CalculateUtil()
        self.model_util = ModelUtil()
        self.model = model
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CustomTransformer':
        """
        拟合转换器：
          - 对 X 标准化后，训练 RandomForestRegressor。
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换数据：
          - 对 X 标准化；
          - 生成预测值 predictions；
          - 计算每行的标准差 sd_per_row 和 RSI rsi_per_row；
          - 提取每行最后一个值 last_elements；
          - 拼合为新特征阵 transformed_data。
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        sd_per_row = np.array([self.calculate_util.calculate_standard_deviation_welford(row) for row in X])
        rsi_per_row = np.array([self.model_util.relative_strength_index(row, period=len(row)//2) for row in X])
        last_elements = X[:, -1]
        transformed_data = np.c_[last_elements, predictions, sd_per_row, rsi_per_row]
        return transformed_data


class ModelUtil:
    """
    多种预测策略工具类：
      - 指数移动平均 EMA；
      - 线性回归；
      - 多项式回归；
      - 谐波回归；
      - 随机森林预测管道；
      - RSI 计算；
      - SARIMA 预测。
    """

    @staticmethod
    def exponential_moving_average_next_value(
        numeric_sequence: List[int],
        span: int = 5,
        enable_rolling_difference: bool = False
    ) -> int:
        """
        基于 EMA（指数移动平均）与可选差分预测下一个值，并结合 RSI 调整向上/向下取整。
        :param numeric_sequence: 输入序列
        :param span: EMA 平滑窗口
        :param enable_rolling_difference: 是否结合差分项
        """
        import math
        import pandas as pd
        if not numeric_sequence:
            raise ValueError("序列不可为空")
        if not all(isinstance(x, (int, float)) for x in numeric_sequence):
            raise ValueError("序列元素必须为数值")

        series = pd.Series(numeric_sequence)
        ema = series.ewm(span=span, min_periods=min(span, len(series)), adjust=False).mean()
        diff = series.diff()

        if enable_rolling_difference and len(diff) > 1:
            pred = ema.iloc[-1] + diff.iloc[-1]
        else:
            pred = ema.iloc[-1]

        # 根据 RSI 决定向上或向下取整
        rsi = ModelUtil.relative_strength_index(numeric_sequence, period=5)
        return math.ceil(pred) if rsi <= 50 else math.floor(pred)

    @staticmethod
    def linear_regression_next_value(numeric_sequence: List[int], degree: int = 1) -> int:
        """
        基于线性（或多项式）回归预测下一个值。
        :param numeric_sequence: 输入序列
        :param degree: 回归多项式次数，degree=1 即线性
        """
        import numpy as np
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        if len(numeric_sequence) < 2:
            raise ValueError("序列长度至少为2")

        X = np.arange(len(numeric_sequence)).reshape(-1, 1)
        y = np.array(numeric_sequence)
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        next_X = poly.transform([[len(y)]])
        pred = model.predict(next_X)[0]
        return CalculateUtil.real_round(pred)

    @staticmethod
    def multivariate_polynomial_regression_next_value(
        numeric_sequence: List[int],
        rolling_size: int = 3,
        degrees: int = 3
    ) -> float:
        """
        多变量多项式回归：滚动窗口生成训练集，预测下一个值。
        """
        import numpy as np
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        if rolling_size > len(numeric_sequence):
            raise ValueError("滚动窗口大小不能超过序列长度")

        train_x, train_y = CalculateUtil.generate_datasets_with_rolling_size(
            data=numeric_sequence, rolling_size=rolling_size
        )
        X = np.array(train_x)
        y = np.array(train_y)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = make_pipeline(PolynomialFeatures(degrees), LinearRegression())
        model.fit(Xs, y)

        test = scaler.transform([numeric_sequence[-rolling_size:]])
        return model.predict(test)[0]

    @staticmethod
    def harmonic_regression_next_value(numeric_sequence: List[int], frequency: float = 1.0) -> int:
        """
        谐波回归：拟合正弦、余弦分量进行预测。
        """
        import numpy as np
        from sklearn.linear_model import LinearRegression
        X = np.array(numeric_sequence[:-1]).reshape(-1, 1)
        y = np.array(numeric_sequence[1:])
        sine = np.sin(2*np.pi*frequency*X)
        cosine = np.cos(2*np.pi*frequency*X)
        features = np.hstack((X, sine, cosine))
        model = LinearRegression().fit(features, y)

        last = np.array([[numeric_sequence[-1]]])
        next_feat = np.hstack((
            last,
            np.sin(2*np.pi*frequency*last),
            np.cos(2*np.pi*frequency*last)
        ))
        pred = model.predict(next_feat)[0]
        return CalculateUtil.real_round(pred)

    @staticmethod
    def random_forest_regressor_transformer(
        numeric_sequence: List[int],
        rolling_size: int,
        warm_start: bool = False,
        random_state: int = 12,
        param_distributions: Optional[Dict] = None,
        param_overrides: Optional[Dict] = None
    ) -> Any:
        """
        随机森林回归管道预测：
          - 生成滚动数据集；
          - 可选随机搜索调参；
          - 构建包含 CustomTransformer 的 Pipeline；
          - 输出预测结果数组。
        """
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.pipeline import Pipeline
        train_x, train_y = CalculateUtil.generate_datasets_with_rolling_size(
            data=numeric_sequence, rolling_size=rolling_size
        )
        X = np.array(train_x)
        y = np.array(train_y)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # 基础随机森林
        base_model = RandomForestRegressor(warm_start=warm_start, random_state=random_state)
        if param_distributions:
            search = RandomizedSearchCV(base_model, param_distributions, **(param_overrides or {}))
            search.fit(Xs, y)
            best = search.best_params_
            base_model = RandomForestRegressor(warm_start=warm_start, random_state=random_state, **best)

        pipeline = Pipeline([
            ('custom', CustomTransformer(base_model)),
            ('final_rf', RandomForestRegressor(warm_start=warm_start, random_state=random_state))
        ])
        pipeline.fit(Xs, y)

        test = scaler.transform([numeric_sequence[-rolling_size:]])
        return pipeline.predict(test)

    @staticmethod
    def random_forest_regressor_next_value(numeric_sequence: List[int], degree: int = 2) -> int:
        """
        直接使用随机森林回归预测下一个值。
        """
        import numpy as np
        from sklearn.preprocessing import PolynomialFeatures
        data = np.array(numeric_sequence).reshape(-1, 1)
        X = np.arange(len(data)).reshape(-1, 1)
        poly = PolynomialFeatures(degree)
        Xp = poly.fit_transform(X)
        model = RandomForestRegressor().fit(Xp, data.ravel())
        next_Xp = poly.transform([[len(data)]])
        pred = model.predict(next_Xp)[0]
        return CalculateUtil.real_round(pred)

    @staticmethod
    def relative_strength_index(numeric_sequence: List[int], period: int = 14) -> float:
        """
        计算 RSI（相对强弱指数）。
        :param numeric_sequence: 数值序列
        :param period: 计算周期，默认为14
        :return: RSI 值（0-100）
        """
        import numpy as np
        if len(numeric_sequence) < period:
            raise ValueError("数据点不足以计算 RSI")

        deltas = np.diff(numeric_sequence)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()
        factor = 2 / (period + 1)
        for i in range(period, len(deltas)):
            avg_gain = gains[i]*factor + avg_gain*(1-factor)
            avg_loss = losses[i]*factor + avg_loss*(1-factor)

        rs = avg_gain/avg_loss if avg_loss != 0 else float('inf')
        return 100 - 100/(1+rs) if avg_loss != 0 else 100

    @staticmethod
    def seasonal_autoregressive_integrated_moving_average_next_value(numeric_sequence: List[int]) -> int:
        """
        SARIMA 模型预测下一个值：
          - 自动搜索最优 (p,d,q)(P,D,Q,m) 参数；
          - 拟合后预测 n_periods=1。
        """
        import numpy as np
        from pmdarima import auto_arima
        ts = np.array(numeric_sequence)
        model = auto_arima(
            ts, start_p=2, start_q=2, max_p=3, max_q=3, m=12,
            start_P=1, start_Q=1, max_P=3, max_Q=3,
            seasonal=True, d=1, D=1, trace=False,
            error_action='ignore', suppress_warnings=True, stepwise=True
        )
        forecast = model.predict(n_periods=1)
        return CalculateUtil.real_round(forecast[0])


class PromptUtil:
    @staticmethod
    def generate_ds_prompt(data, prefix, out_dir):
        template = """
【系统初始化指令】
※ 当前会话启用专家模式，启用以下技术栈：
1. 时序分析引擎：Prophet+ARIMA
2. 统计学习模块：XGBoost+SHAP
3. 深度学习框架：Transformer+LSTM
4. 组合优化器：NSGA-II算法
5. 异常检测器：Isolation Forest

【概念定义】
1. 前区从1-35取5个数字
2. 后区从1-12取2个数字
3. 和值对应区内选取数字之和
4. 跨度对应区内最大最小值之差
5. 区间比指把所有号码按等比分成几个范围，所选号码落在不同区间个数的比值
    5.1 前区区间(1, 12), (13, 24), (25, 35)
    5.2 后区区间(1, 6), (7, 12)
6. 奇偶比指对应区间内奇数和偶数的比值

【数据输入规范】
※ 数据编码方案（压缩比1:8）：
期号|星期|前区|后区|前区和值|前区跨度|前区区间比|前区奇偶比|后区和值|后区跨度|后区区间比|后区奇偶比
{input_data}

【核心分析矩阵】
├─ 时序分析组件
│  ├─ ARIMA参数：(p=3,d=1,q=2)
│  ├─ LSTM结构：BiLSTM(128)+Attention
│  └─ 周期性检测：傅里叶级数分解
│
├─ 统计模型集群
│  ├─ 蒙特卡洛模拟：10000次迭代
│  ├─ 马尔可夫链：5阶状态转移
│  └─ 贝叶斯网络：先验概率动态更新
│
├─ 集成学习模块
│  ├─ 特征工程：
│  │   ├─ 衍生特征：星期效应值
│  │   └─ 交互特征：跨期质数对
│  │
│  ├─ 模型堆叠：
│  │   ├─ 基模型：XGBoost/LightGBM/CatBoost
│  │   └─ 元模型：逻辑回归
│  │
│  └─ SHAP解释器：可视化特征贡献
│
└─ 组合优化层
   ├─ 多目标优化：
   │   ├─ 目标1：最大概率覆盖
   │   └─ 目标2：最小历史重复
   │
   └─ 约束条件：
       ├─ 前区极差：≥20
       └─ 后区熵值：≤1.8

【残差诊断协议】
1. 残差类型：标准化预测残差
2. 检测标准：
   - 3σ原则筛选异常点
   - DW检验自相关性（阈值0.05）
3. 残差利用：
   ├─ 生成对抗样本
   └─ 修正预测偏倚

【对抗性验证框架】
for 迭代 in 1..10:
    生成对抗样本：
    - 前区扰动：±2数值偏移
    - 后区置换：质数↔合数转换
    验证模型鲁棒性：
    - 输出稳定性系数≥0.85
    - 特征重要性排序一致性

【量子启发模块】 
量子比特映射方案：
前区数字 → 5个量子位的叠加态
后区数字 → 2个纠缠量子位
观测协议：
1. 执行Grover算法（迭代次数=15）
2. 测量获得概率幅前3的组合

【输出控制体系】
※ 结构化输出要求：
预测候选集（TOP2）：
1. 主方案：
   前区：[6,11,19,24,33] 
   后区：[5,9]
   置信度：0.87⭐
   风险标记：后区和值超历史90%分位
   
2. 备选方案1：
   前区：[8,14,17,22,29]
   后区：[3,11]
   置信度：0.82⚠️
   风险标记：前区含2期重复号

【动态反馈机制】
※ 结果验证循环：
预测发布 → 实际开奖 → 误差分析 → 模型微调
微调参数：
- 学习率：余弦退火调整
- 特征权重：动态衰减
- 集成系数：梯度更新

【技术保障措施】
1. 数据消毒流程：
   - Winsorize处理（α=0.05）
   - 星期效应正则化
2. 计算资源分配：
   - 时序分析：40% CPU
   - 深度学习：80% GPU
3. 容错机制：
   - 当连续3次预测偏离2σ时
   - 自动切换备用模型集群
"""
        with open(os.path.join(out_dir, f'{prefix}_ds_prompt.txt'), 'w') as f:
            f.write(template.format(input_data=data))

    @staticmethod
    def generate_gpt_prompt(data, prefix, out_dir):
        template = """
请充分分析以下历史开奖大数据（数据后附），使用**尽可能多的概率、统计、机器学习、时间序列、符号推理与自定义规则**，多层次剖析，并生成未来一期的“前区5+后区2”号码预测。

#### 历史数据部分（示例，每期一行，用符号分隔）

#### 概念定义
1. 前区从1-35取5个数字
2. 后区从1-12取2个数字
3. 和值对应区内选取数字之和
4. 跨度对应区内最大最小值之差
5. 区间比指把所有号码按等比分成几个范围，所选号码落在不同区间个数的比值
5.1 前区区间(1, 12), (13, 24), (25, 35)
5.2 后区区间(1, 6), (7, 12)
6. 奇偶比指对应区间内奇数和偶数的比值

#### 数据输入规范
※ 数据编码方案（压缩比1:8）：
期号|星期|前区|后区|前区和值|前区跨度|前区区间比|前区奇偶比|后区和值|后区跨度|后区区间比|后区奇偶比
{input_data}


#### 分析思路与方法要求

1. **统计分布与热力分析**
- 统计各号码出现频次，按热码/冷码/温码分类
- 分析奇偶、区间、和值等大类分布趋势
- 用多期滑动窗口分析近期异动（如哪个号突然转热）
- 计算各号码遗漏期数及最大遗漏

2. **时间序列与自回归分析**
- 运用移动平均、加权平滑分析号码和值等指标的趋势线
- 用ARIMA或LSTM思想提取相关模式，分析高波动期和低波动周期
- 探索周期性（如每逢周三出小号较多）

3. **组合与特征工程**
- 识别连号、斜连号、首尾号、对称结构、复合质数等复杂组合
- 分析区间比、跨度变化、和值撬动、冷门区爆发
- 计算前区/后区交互特征（如同一期质数个数）

4. **概率与蒙特卡洛抽样**
- 构造全号池进行万次合规采样，统计高概率组合出现概率
- 蒙特卡洛方法用于可能区间分布模拟

5. **机器学习假想建模**
- 提示LLM模仿神经网络、决策树等方法表达特征权重
- 让模型假设特征组合对结果概率的影响，并进行置信度评估
- 给出不同算法下的若干候选组合及概率分层

6. **符号与规则推理**
- 动态生成预测公式、差分模式
- 自动归纳本期与历史异同，尽量发现创新性模式
- 强制检查是否有禁止热号连开、冷号回补等现象

7. **假设检验与交叉验证**
- 对结果用历史数据拟合仿真，过滤掉反常组合
- 提供每组预测的历史表现评分

#### 输出格式要求

```
预测号码（主推）：
前区：[a, b, c, d, e]
后区：[x, y]

推荐号码组（最多3组，按置信度排序）：
1. 前区：[__] 后区：[__] 置信度：__
2. 前区：[__] 后区：[__] 置信度：__
...

分析概述：
- 主要热码、冷码分布
- 和值、跨度、区间趋势
- 本期重点关注组合/风险提示
- 预测模型主要依赖的方法
- 最终模型自评分（0-100）

【如遇历史数据中有异常，请指出具体期号及数据项】
```
"""
        with open(os.path.join(out_dir, f'{prefix}_gpt_prompt.txt'), 'w') as f:
            f.write(template.format(input_data=data))
