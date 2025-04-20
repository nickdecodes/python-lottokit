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

import math
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
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
        ts = np.array(numeric_sequence)
        model = auto_arima(
            ts, start_p=2, start_q=2, max_p=3, max_q=3, m=12,
            start_P=1, start_Q=1, max_P=3, max_Q=3,
            seasonal=True, d=1, D=1, trace=False,
            error_action='ignore', suppress_warnings=True, stepwise=True
        )
        forecast = model.predict(n_periods=1)
        return CalculateUtil.real_round(forecast[0])
