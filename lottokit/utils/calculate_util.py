#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : nickdecodes
@Email   : nickdecodes@163.com
@Usage   :
@FileName: calculate_util.py
@DateTime: 2024/7/22 10:09
@SoftWare: PyCharm
"""

import math
import datetime
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple, Any, Optional, Union, Set, Dict, Generator


class CalculateUtil(ABC):
    """
    计算工具抽象基类，提供各种数值统计、特征工程及序列分析方法。
    """

    @staticmethod
    def real_round(n: float) -> int:
        """
        四舍五入：对 n + 0.5 向下取整。
        :param n: 待四舍五入的浮点数
        :return: 四舍五入后整数
        """
        return math.floor(n + 0.5)

    @staticmethod
    def invert_round(n: float) -> int:
        """
        非对称取整：小数部分>=0.5 向下取整，否则向上取整。
        :param n: 待取整数
        :return: 取整后整数
        """
        if n % 1 >= 0.5:
            return math.floor(n)
        else:
            return math.ceil(n)

    @classmethod
    def list_to_int(cls, numbers: List[int], zero_replacement: str = '') -> int:
        """
        将整型列表拼接为一个整数，可用零替换字符串替代 0。
        :param numbers: 整数列表
        :param zero_replacement: 0 的替换字符，默认空串
        :return: 拼接后整数
        """
        if not isinstance(numbers, List):
            raise ValueError("输入必须为列表")
        if not all(isinstance(x, int) for x in numbers):
            raise ValueError("列表元素必须为整数")
        parts = [zero_replacement if x == 0 else str(x) for x in numbers]
        return int(''.join(parts))

    @classmethod
    def int_to_list(cls, number: int, modulus: int = 10) -> List[int]:
        """
        将整数拆分为各位数字列表，并对每位取模。
        :param number: 输入整数
        :param modulus: 取模值，默认 10
        :return: 每位数字组成的列表
        """
        if not isinstance(number, int):
            raise ValueError("输入必须为整数")
        if not isinstance(modulus, int) or modulus <= 0:
            raise ValueError("modulus 必须为正整数")
        return [int(d) % modulus for d in str(abs(number))]

    @staticmethod
    def encode_combination(combination: List[int], max_n: int = 35) -> float:
        """
        将组合编码为唯一索引后取对数：log(索引+1)。
        :param combination: 已排序组合
        :param max_n: 最大值，默认 35
        :return: 对数索引
        """
        comb = sorted(combination)
        index = 1
        k = len(comb)
        for i, num in enumerate(comb):
            for j in range(num - 1):
                if max_n - (j + 1) >= k - (i + 1):
                    index += math.comb(max_n - (j + 1), k - (i + 1))
        return np.log(index + 1)

    @staticmethod
    def decode_combination(index: float, k: int = 5, max_n: int = 35) -> List[int]:
        """
        对数索引解码为组合。
        :param index: 编码后的对数索引
        :param k: 组合长度
        :param max_n: 最大值
        :return: 解码后组合
        """
        idx = np.exp(index) - 1
        comb: List[int] = []
        rem = k
        n = max_n
        while rem > 0 and rem <= n:
            c = math.comb(n - 1, rem - 1)
            if idx >= c:
                idx -= c
            else:
                comb.append(max_n - n + 1)
                rem -= 1
            n -= 1
        return comb

    @staticmethod
    def longest_increasing_subsequence(numeric_sequence: List[int]) -> List[int]:
        """
        求最长递增子序列 (LIS)。
        :param numeric_sequence: 输入整数序列
        :return: LIS 列表
        """
        n = len(numeric_sequence)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if numeric_sequence[i] > numeric_sequence[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        length = max(dp, default=0)
        lis: List[int] = []
        if length == 0:
            return lis
        idx = dp.index(length)
        lis.append(numeric_sequence[idx])
        cur = length - 1
        for i in range(idx - 1, -1, -1):
            if dp[i] == cur and numeric_sequence[i] < lis[-1]:
                lis.append(numeric_sequence[i])
                cur -= 1
        return list(reversed(lis))

    @staticmethod
    def longest_decreasing_subsequence(numeric_sequence: List[int]) -> List[int]:
        """
        求最长递减子序列 (LDS)。
        :param numeric_sequence: 输入整数序列
        :return: LDS 列表
        """
        n = len(numeric_sequence)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if numeric_sequence[i] < numeric_sequence[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        length = max(dp, default=0)
        lds: List[int] = []
        if length == 0:
            return lds
        idx = dp.index(length)
        lds.append(numeric_sequence[idx])
        cur = length - 1
        for i in range(idx - 1, -1, -1):
            if dp[i] == cur and numeric_sequence[i] > lds[-1]:
                lds.append(numeric_sequence[i])
                cur -= 1
        return list(reversed(lds))

    @staticmethod
    def generate_chunks_with_next(data: List[Any], chunk_size: int = 10
                                 ) -> Generator[Tuple[List[Any], Optional[Any]], None, None]:
        """
        按固定大小分块并返回每块及其下一个元素。
        :param data: 原始列表
        :param chunk_size: 分块大小
        :yield: (当前块, 下一个元素 or None)
        """
        rem = len(data) % chunk_size
        if rem:
            data = data[rem:]
        for i in range(0, len(data)):
            chunk = data[i:i + chunk_size]
            nxt = data[i + chunk_size] if i + chunk_size < len(data) else None
            if nxt is None:
                break
            yield chunk, nxt

    @staticmethod
    def generate_datasets_with_rolling_size(data: List[Any], rolling_size: int = 5,
                                            adjust: bool = False
                                            ) -> Tuple[List[List[int]], List[int]]:
        """
        生成滚动窗口训练集和对应标签集，可选剔除极值。
        :param data: 原始数列
        :param rolling_size: 窗口大小
        :param adjust: 是否去除窗口最大/最小值
        :return: (训练集列表, 预测标签列表)
        """
        x_sets: List[List[int]] = []
        y_sets: List[int] = []
        for i in range(len(data) - rolling_size):
            window = data[i:i + rolling_size]
            label = data[i + rolling_size]
            if adjust and len(window) > 2:
                max_v, min_v = max(window), min(window)
                filtered = [v for v in window if v not in (max_v, min_v)]
                x_sets.append(filtered)
            else:
                x_sets.append(window)
            y_sets.append(label)
        return x_sets, y_sets

    @staticmethod
    def calculate_euclidean_distance(point1: Tuple[Any, ...],
                                     point2: Tuple[Any, ...]) -> int:
        """
        计算两点欧氏距离并四舍五入。
        :param point1: 坐标点1
        :param point2: 坐标点2
        :return: 距离整数
        """
        if point1 is None or point2 is None:
            raise ValueError("点不能为空")
        if len(point1) != len(point2):
            raise ValueError("维度不一致")
        p1 = [float(p or 0.0) for p in point1]
        p2 = [float(p or 0.0) for p in point2]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
        return CalculateUtil.real_round(dist)

    @staticmethod
    def calculate_same_number_index(target_data: List[int],
                                    input_data: List[int]) -> List[int]:
        """
        找出 target_data 中出现在 input_data 的元素索引（0-based）。
        :param target_data: 目标列表
        :param input_data: 源列表
        :return: 索引列表
        """
        return [i for i, v in enumerate(target_data) if v in input_data]

    @staticmethod
    def calculate_edge_number_index(target_data: List[int],
                                    input_data: List[int]) -> List[int]:
        """
        找出 target_data 中与 input_data 相邻（±1）的元素索引。
        :param target_data: 目标列表
        :param input_data: 源列表
        :return: 索引列表
        """
        idxs: List[int] = []
        for i, v in enumerate(target_data):
            if any(abs(v - x) == 1 for x in input_data):
                idxs.append(i)
        return idxs

    @classmethod
    def calculate_zone_ratio(cls,
                             number_combination: Iterable[int],
                             zone_ranges: List[Tuple[int, int]]
                             ) -> Tuple[int, ...]:
        """
        统计组合中落在各个区间的数量。
        :param number_combination: 数组合
        :param zone_ranges: 区间列表 [(start,end),...]
        :return: 区间计数元组
        """
        if not zone_ranges:
            raise ValueError("必须指定区间")
        counts = [0] * len(zone_ranges)
        for num in map(int, number_combination):
            for i, (s, e) in enumerate(zone_ranges):
                if s <= num <= e:
                    counts[i] += 1
                    break
        return tuple(counts)

    @classmethod
    def calculate_big_small_ratio(cls,
                                  number_combinations: Iterable[int],
                                  big_small_ranges: List[Tuple[int, int]]
                                  ) -> Tuple[int, ...]:
        """
        统计组合中各“大小”区间的数量。
        :param number_combinations: 数组合
        :param big_small_ranges: 大小区间列表
        :return: 各区间计数
        """
        if not big_small_ranges:
            raise ValueError("必须指定大小区间")
        counts = [0] * len(big_small_ranges)
        for num in map(int, number_combinations):
            for i, (s, e) in enumerate(big_small_ranges):
                if s <= num <= e:
                    counts[i] += 1
                    break
        return tuple(counts)

    @classmethod
    def calculate_road_012_ratio(cls,
                                 number_combination: Iterable[int],
                                 road_012_ranges: List[tuple[int, ...]]
                                 ) -> Tuple[int, ...]:
        """
        统计组合中落在各“路”（自定义值集合）的数量。
        :param number_combination: 数组合
        :param road_012_ranges: 路集合列表
        :return: 各路计数
        """
        if not road_012_ranges:
            raise ValueError("必须指定路范围")
        counts = [0] * len(road_012_ranges)
        for num in map(int, number_combination):
            for i, vals in enumerate(road_012_ranges):
                if num in vals:
                    counts[i] += 1
                    break
        return tuple(counts)

    @classmethod
    def calculate_odd_even_ratio(cls,
                                 number_combination: Iterable[int]
                                 ) -> Tuple[int, int]:
        """
        统计组合中奇数和偶数的数量。
        :param number_combination: 数组合
        :return: (奇数, 偶数)
        """
        odd = sum(1 for v in number_combination if v % 2)
        total = len(list(number_combination))
        return odd, total - odd

    @classmethod
    def calculate_prime_composite_ratio(cls,
                                        number_combination: Iterable[int]
                                        ) -> Tuple[int, int]:
        """
        统计组合中素数（1 视为素数）和合数的数量。
        :param number_combination: 数组合
        :return: (素数, 合数)
        """
        def is_prime(x: int) -> bool:
            if x == 1:
                return True
            if x < 2 or (x % 2 == 0):
                return x == 2
            for i in range(3, int(x**0.5) + 1, 2):
                if x % i == 0:
                    return False
            return True

        nums = list(number_combination)
        prime_cnt = sum(1 for v in nums if is_prime(v))
        return prime_cnt, len(nums) - prime_cnt

    @classmethod
    def calculate_span(cls,
                       number_combination: Iterable[int]
                       ) -> int:
        """
        计算数列跨度（最大值-最小值）。
        :param number_combination: 数组合
        :return: 跨度
        """
        vals = list(number_combination)
        return int(max(vals) - min(vals))

    @classmethod
    def calculate_sum_total(cls,
                            number_combination: Iterable[Union[int, float]]
                            ) -> Union[int, float]:
        """
        计算数列元素总和。
        :param number_combination: 数组合
        :return: 总和
        """
        return sum(number_combination)

    @classmethod
    def calculate_sum_tail(cls,
                           number_combination: Iterable[int]
                           ) -> int:
        """
        计算数列总和的个位数。
        :param number_combination: 数组合
        :return: 个位数
        """
        return cls.calculate_sum_total(number_combination) % 10

    @classmethod
    def calculate_weekday(cls,
                          date: str,
                          date_format: str = '%Y-%m-%d'
                          ) -> int:
        """
        计算日期对应的星期（1-周一，7-周日）。
        :param date: 日期字符串或数字
        :param date_format: 格式字符串
        :return: 星期编号
        """
        if date.isdigit():
            return int(date)
        d = datetime.datetime.strptime(date, date_format).date()
        return d.weekday() + 1

    @classmethod
    def calculate_ac(cls,
                     number_combination: Iterable[int]
                     ) -> int:
        """
        计算组合复杂度 AC: 不同差值数 - (元素数-1)。
        :param number_combination: 数组合
        :return: 复杂度
        """
        nums = list(number_combination)
        diffs: Set[int] = set()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(abs(nums[j] - nums[i]))
        return len(diffs) - (n - 1)

    @classmethod
    def calculate_avg(cls,
                      number_combination: Iterable[int]
                      ) -> int:
        """
        计算数列平均值向下取整。
        :param number_combination: 数组合
        :return: 平均值
        """
        vals = list(number_combination)
        return math.floor(sum(vals) / len(vals))

    @classmethod
    def calculate_consecutive_numbers(cls,
                                      number_combination: Iterable[int]
                                      ) -> List[List[int]]:
        """
        提取数列中所有连续递增子序列。
        :param number_combination: 数组合
        :return: 连续子序列列表
        """
        nums = list(number_combination)
        seqs: List[List[int]] = []
        cur = [nums[0]] if nums else []
        for v in nums[1:]:
            if v == cur[-1] + 1:
                cur.append(v)
            else:
                if len(cur) > 1:
                    seqs.append(cur)
                cur = [v]
        if len(cur) > 1:
            seqs.append(cur)
        return seqs

    @classmethod
    def calculate_repeated_numbers(cls,
                                   number_combinations: Iterable[Iterable[int]],
                                   window: int = 2
                                   ) -> List[int]:
        """
        计算最近 window 期所有组合的交集。
        :param number_combinations: 多期数组
        :param window: 窗口期数
        :return: 重复号码列表
        """
        combos = list(number_combinations)[-window:]
        if not combos:
            return []
        inter = set(combos[0])
        for c in combos[1:]:
            inter &= set(c)
        return list(inter)

    @classmethod
    def calculate_edge_numbers(cls,
                               number_combinations: Iterable[Iterable[int]],
                               window: int = 2
                               ) -> List[int]:
        """
        计算相邻期 ±1 范围内的“边缘”号码。
        :param number_combinations: 多期数组
        :param window: 窗口期数
        :return: 边缘号码列表
        """
        combos = list(number_combinations)[-window:]
        edges: Set[int] = set()
        if len(combos) < 2:
            return []
        for prev, curr in zip(combos, combos[1:]):
            prev_set = {x + d for x in prev for d in (-1, 1) if x + d > 0}
            edges |= set(curr) & prev_set
        return sorted(edges)

    @classmethod
    def calculate_cold_hot_numbers(cls,
                                   number_combinations: Iterable[Iterable[int]],
                                   all_numbers: Iterable[int],
                                   window: int = 5
                                   ) -> Tuple[List[int], List[int]]:
        """
        计算窗口期内冷热号：
          - 冷号：未出现
          - 热号：出现过
        :param number_combinations: 多期数组
        :param all_numbers: 全数列表
        :param window: 窗口期数
        :return: (冷号列表, 热号列表)
        """
        combos = list(number_combinations)[-window:]
        appeared: Set[int] = set(x for period in combos for x in period)
        all_set = set(all_numbers)
        cold = sorted(all_set - appeared)
        hot = sorted(appeared)
        return cold, hot

    @classmethod
    def calculate_omitted_numbers(cls,
                                  number_combinations: Iterable[Iterable[int]],
                                  all_numbers: Iterable[int],
                                  window: int = 10
                                  ) -> Dict[int, int]:
        """
        计算遗漏值：各号码自上次出现以来的期数。
        :param number_combinations: 多期数组
        :param all_numbers: 全数列表
        :param window: 窗口期数
        :return: {号码: 遗漏期数}
        """
        combos = list(number_combinations)[-window:]
        draws = 0
        omit: Dict[int, int] = {num: -1 for num in all_numbers}
        for period in reversed(combos):
            for num in all_numbers:
                if omit[num] == -1 and num in period:
                    omit[num] = draws
            draws += 1
        # 未出现的设置为 draws
        for num in all_numbers:
            if omit[num] == -1:
                omit[num] = draws
        return omit

    @staticmethod
    def calculate_standard_deviation_welford(numeric_sequence: List[Union[int, float]],
                                             decay_factor: float = 0.95
                                             ) -> float:
        """
        Welford 在线算法加权标准差，近期数据权重更高。
        :param numeric_sequence: 数值序列
        :param decay_factor: 衰减因子
        :return: 标准差
        """
        if not numeric_sequence:
            raise ValueError("序列不能为空")
        mean = 0.0
        M2 = 0.0
        weighted_n = 0.0
        n = len(numeric_sequence)
        for i, x in enumerate(numeric_sequence, 1):
            weight = decay_factor ** (n - i)
            delta = x - mean
            weighted_n += weight
            mean += delta * weight / weighted_n
            delta2 = x - mean
            M2 += delta * delta2 * weight
        return math.sqrt(M2 / weighted_n) if weighted_n >= 2 else 0.0

    @staticmethod
    def calculate_standard_deviation(numeric_sequence: List[Union[int, float]]
                                     ) -> float:
        """
        计算普通标准差（总体标准差）。
        :param numeric_sequence: 数值序列
        :return: 标准差
        """
        if not numeric_sequence:
            raise ValueError("序列不能为空")
        mean = sum(numeric_sequence) / len(numeric_sequence)
        var = sum((x - mean) ** 2 for x in numeric_sequence) / len(numeric_sequence)
        return math.sqrt(var)

    @abstractmethod
    def calculate_winning_amount(
        self,
        winning_number_combination: List[int],
        purchase_number_combinations: List[List[int]],
        **kwargs: Any
    ) -> Tuple[float, int]:
        """
        抽象方法：根据中奖号码和投注号码计算中奖金额和中注数，子类实现。
        :param winning_number_combination: 中奖号码列表
        :param purchase_number_combinations: 投注号码列表集
        :return: (总奖金, 中奖注数)
        """
        pass
