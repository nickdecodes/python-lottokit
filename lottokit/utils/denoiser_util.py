#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : nickdecodes
@Email   : nickdecodes@163.com
@Usage   :
@FileName: denoiser_util.py
@DateTime: 2024/7/22 10:25
@SoftWare: PyCharm
"""

import pywt
import numpy as np
from typing import List, Union


class DenoiserUtil:
    """
    去噪工具类，提供多种信号平滑与去噪方法：
      - 移动平均
      - 中值滤波
      - 异常值剔除
      - 小波变换去噪
    """

    @staticmethod
    def moving_average(
        sequence: List[Union[int, float]],
        window_size: int = 3
    ) -> np.ndarray:
        """
        计算输入序列的滑动平均值（Moving Average）。

        :param sequence: 原始数值序列（列表形式）。
        :param window_size: 滑动窗口大小，缺省为 3。
        :return: 经过滑动平均处理后的 numpy 数组，长度 = len(sequence) - window_size + 1。
        """
        # 使用 numpy.convolve 进行加权求和并取 valid 模式
        return np.convolve(sequence, np.ones(window_size) / window_size, mode='valid')

    @staticmethod
    def median_filter(
        sequence: List[Union[int, float]],
        kernel_size: int = 3
    ) -> List[float]:
        """
        对序列应用中值滤波（Median Filter），降低噪声影响。

        :param sequence: 原始数值序列（列表形式）。
        :param kernel_size: 滤波窗口大小，缺省为 3。
        :return: 经过中值滤波后的新序列（列表形式），长度与原序列相同。
        """
        filtered = []
        half = kernel_size // 2
        # 遍历每个位置，取窗口范围内的中值
        for i in range(len(sequence)):
            start = max(0, i - half)
            end = min(len(sequence), i + half + 1)
            median_val = np.median(sequence[start:end])
            filtered.append(median_val)
        return filtered

    @staticmethod
    def remove_outliers(
        sequence: List[Union[int, float]],
        threshold: float = 2.0
    ) -> List[Union[int, float]]:
        """
        剔除序列中的异常值（Outliers），基于均值 ± threshold * 标准差。

        :param sequence: 原始数值序列（列表形式）。
        :param threshold: 判定异常的阈值倍数，缺省为 2.0。
        :return: 去掉超过阈值范围后剩余的序列（列表形式）。
        """
        mean = np.mean(sequence)
        std = np.std(sequence)
        # 保留绝对偏差不超过 threshold * std 的样本
        return [x for x in sequence if abs(x - mean) <= threshold * std]

    @staticmethod
    def denoise_wavelet(
        sequence: List[Union[int, float]],
        wavelet: str = 'db1',
        level: int = 1
    ) -> np.ndarray:
        """
        使用小波变换（Wavelet）对序列进行去噪。

        :param sequence: 原始数值序列（列表形式）。
        :param wavelet: 小波基名称，缺省为 'db1'（Daubechies 1）。
        :param level: 分解层数，缺省为 1。
        :return: 去噪后重构的序列（numpy 数组）。
        """
        # 小波分解
        coeffs = pywt.wavedec(sequence, wavelet, level=level)
        # 估算噪声标准差 sigma
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        # 阈值设定
        uthresh = sigma * np.sqrt(2 * np.log(len(sequence)))
        # 对细节系数应用软阈值
        coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
        # 小波重构
        return pywt.waverec(coeffs, wavelet)
