#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : nickdecodes
@Email   : nickdecodes@163.com
@Usage   :
@FileName: daletou.py
@DateTime: 2023/12/7 19:58
@SoftWare:
"""

import os
from itertools import combinations
from datetime import datetime, timedelta
from collections import Counter, namedtuple
from typing import List, Tuple, Any, Optional, Union, Dict, NamedTuple, Callable, Iterable, Set
from .utils import IOUtil, ModelUtil, SpiderUtil, CalculateUtil, AnalyzeUtil, PromptUtil


class Daletou(IOUtil, ModelUtil, SpiderUtil, CalculateUtil, AnalyzeUtil):
    # Default values for t prediction configuration
    AWARD_URL = 'https://www.lottery.gov.cn/kj/kjlb.html?dlt'
    PREDICT_NUM = 5  # Default number of t tickets to predict
    NORMAL_SIZE = 7  # Length of normal winning numbers e.g. 12,17,27,29,34,06,09
    ORIGIN_SIZE = 9  # Length of historical record data e.g. 07013,2007-06-27,12,17,27,29,34,06,09
    SINA_SIZE = 13  # Length of historical record data e.g. 2025042,6,06,08,11,18,20,05,11,63,14,3:2:0,1:4
    # Super Lotto prize rules (pre-tax, without considering floating conditions)
    AWARD_RULES = {
        (5, 2): 10000000,
        (5, 1): 800691,
        (5, 0): 10000,
        (4, 2): 3000,
        (4, 1): 300,
        (3, 2): 200,
        (4, 0): 100,
        (3, 1): 15,
        (2, 2): 15,
        (3, 0): 5,
        (2, 1): 5,
        (1, 2): 5,
        (0, 2): 5
    }

    # Front area configuration
    FRONT_SIZE = 5  # Number of numbers to choose in the front area
    FRONT_VOCAB_SIZE = 35  # Types of numbers in the front area
    FRONT_ZONE_RANGES = [(1, 12), (13, 24), (25, 35)]  # Front area zone distribution
    FRONT_BIG_SMALL_RANGES = [(18, 35), (1, 17)]  # Numbers 01–17 are small, 18–35 are big in the front area.
    FRONT_ROAD_012_RANGES = [
        (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33),
        (1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34),
        (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    ]  # Front area 012 road division

    # Back area configuration
    BACK_SIZE = 2  # Number of numbers to choose in the back area
    BACK_VOCAB_SIZE = 12  # Types of numbers in the back area
    BACK_ZONE_RANGES = [(1, 6), (7, 12)]  # Back area zone distribution
    BACK_BIG_SMALL_RANGES = [(7, 12), (1, 6)]  # Numbers 01–06 are small, 07–12 are big in the back area.
    BACK_ROAD_012_RANGES = [
        (3, 6, 9, 12),
        (1, 4, 7, 10),
        (2, 5, 8, 11)
    ]  # Back area 012 road division

    # NamedTuple for t data
    Lottery = namedtuple('Lottery', ['period', 'weekday', 'front', 'back', 'sum_total', 'span', 'zone_ratio', 'odd_even_ratio'])

    def __init__(self, **kwargs):
        """
        Initialize the Daletou object with configuration parameters.

        :param kwargs: A dictionary of keyword arguments where:
            - 'url': str - The URL to fetch the data from. Defaults to the class's AWARD_URL if not provided.
            - 'log_file': Optional[str] - The path to the log file. Defaults to None.
            - 'dataset_dir': str - The directory path for the dataset. Defaults to './dataset'.
            - 'predict_num': int - The number of predictions to make. Defaults to the class's PREDICT_NUM.
            - 'normal_size': int - The size of the normal number set. Defaults to the class's NORMAL_SIZE.
            - 'origin_size': int - The original size of the dataset. Defaults to the class's ORIGIN_SIZE.
            - 'award_rules': Dict - The rules for the awards. Defaults to the class's AWARD_RULES.
            - 'front_size': int - The size of the front number set. Defaults to the class's FRONT_SIZE.
            - 'front_vocab_size': int - The vocabulary size of the front number set.
                                  Defaults to the class's FRONT_VOCAB_SIZE.
            - 'back_size': int - The size of the back number set. Defaults to the class's BACK_SIZE.
            - 'back_vocab_size': int - The vocabulary size of the back number set.
                                 Defaults to the class's BACK_VOCAB_SIZE.
            - 'front_zone_ranges': List[Tuple[int, int]] - The range of front zones.
                                   Defaults to the class's FRONT_ZONE_RANGES.
            - 'front_big_small_ranges': List[Tuple[int, int]] - The range of front big/small numbers.
                                        Defaults to the class's FRONT_BIG_SMALL_RANGES.
            - 'front_road_012_ranges': List[Tuple[int, ...]] - The range of front road 012 numbers.
                                       Defaults to the class's FRONT_ROAD_012_RANGES.
            - 'back_zone_ranges': List[Tuple[int, int]] - The range of back zones.
                                  Defaults to the class's BACK_ZONE_RANGES.
            - 'back_big_small_ranges': List[Tuple[int, int]] - The range of back big/small numbers.
                                       Defaults to the class's BACK_BIG_SMALL_RANGES.
            - 'back_road_012_ranges': List[Tuple[int, ...]] - The range of back road 012 numbers.
                                      Defaults to the class's BACK_ROAD_012_RANGES.
        """
        super().__init__(**kwargs)
        self.url: str = kwargs.get('url', self.AWARD_URL)
        self.log_file: Optional[str] = kwargs.get('log_file', None)
        self.dataset_dir: str = kwargs.get('dataset_dir', os.path.join('dataset'))
        self.predict_num: int = kwargs.get('predict_num', self.PREDICT_NUM)
        self.normal_size: int = kwargs.get('normal_size', self.NORMAL_SIZE)
        self.origin_size: int = kwargs.get('origin_size', self.ORIGIN_SIZE)
        self.sina_size: int = kwargs.get('sina_size', self.SINA_SIZE)
        self.award_rules: Dict = kwargs.get('award_rules', self.AWARD_RULES)
        self.front_size: int = kwargs.get('front_size', self.FRONT_SIZE)
        self.front_vocab_size: int = kwargs.get('front_vocab_size', self.FRONT_VOCAB_SIZE)
        self.back_size: int = kwargs.get('back_size', self.BACK_SIZE)
        self.back_vocab_size: int = kwargs.get('back_vocab_size', self.BACK_VOCAB_SIZE)
        self.front_zone_ranges: List[Tuple[int, int]] = kwargs.get('front_zone_ranges', self.FRONT_ZONE_RANGES)
        self.front_big_small_ranges: List[Tuple[int, int]] = kwargs.get('front_big_small_ranges', self.FRONT_BIG_SMALL_RANGES)
        self.front_road_012_ranges: List[Tuple[int, ...]] = kwargs.get('front_road_012_ranges', self.FRONT_ROAD_012_RANGES)
        self.back_zone_ranges: List[Tuple[int, int]] = kwargs.get('back_zone_ranges', self.BACK_ZONE_RANGES)
        self.back_big_small_ranges: List[Tuple[int, int]] = kwargs.get('back_big_small_ranges', self.BACK_BIG_SMALL_RANGES)
        self.back_road_012_ranges: List[Tuple[int, ...]] = kwargs.get('back_road_012_ranges', self.BACK_ROAD_012_RANGES)

        self.app_log = self.get_logger(self.log_file)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        self.history_record_path = os.path.join(self.dataset_dir, 'daletou_history.csv')
        self.predict_record_path = os.path.join(self.dataset_dir, 'daletou_predict.csv')
        self.period_record_path = os.path.join(self.dataset_dir, 'daletou_period.json')
        self.weekday_record_path = os.path.join(self.dataset_dir, 'daletou_weekday.json')
        self.consecutive_record_path = os.path.join(self.dataset_dir, 'consecutive.json')
        self.repeated_record_path = os.path.join(self.dataset_dir, 'repeated.json')
        self.edge_record_path = os.path.join(self.dataset_dir, 'edge.json')
        self.cold_hot_record_path = os.path.join(self.dataset_dir, 'cold_hot.json')
        self.omitted_record_path = os.path.join(self.dataset_dir, 'omitted.json')
        self.all_fronts = list(combinations(range(1, self.front_vocab_size + 1), self.front_size))
        self.all_backs = list(combinations(range(1, self.back_vocab_size + 1), self.back_size))
        self.feature_keys = ['zone_ratio', 'big_small_ratio', 'road_012_ratio', 'odd_even_ratio',
                             'prime_composite_ratio', 'span', 'sum_total', 'sum_tail', 'ac',
                             'consecutive_numbers']
        self.append_feature_mapping = {
            'repeated_numbers': 1,
            'edge_numbers': 1,
            # 'cold_hot_numbers': 4,
            # 'omitted_numbers': 9
        }

    """
    A pure virtual method inherited from util.AnalyzeUtil
    """

    def analyze_same_period_numbers(self, data=None, **kwargs: Any) -> None:
        """
        分析同一期号历史数据，并将新记录追加到 period_record_path 对应分组。
        """
        if data is None:
            return

        # 读取已有 JSON 记录
        if os.path.exists(self.period_record_path):
            record_data = self.read_json_data_from_file(self.period_record_path, app_log=self.app_log)
        else:
            record_data = {}

        for d in data:
            # 构造通用记录: [期号, 原始星期, 计算星期, 前区列表, 后区列表]
            lottery_data = self.convert_lottery_data(d)
            # 如果已有记录的 last_period >= 本次期号，则跳过
            if int(record_data.get('last_period', 0)) >= int(lottery_data.period):
                continue

            # 日志输出
            self.app_log.info(f"Update {lottery_data} to {self.period_record_path}")
            # 更新 last_period 并按期号后三位分组追加
            record_data['last_period'] = lottery_data.period
            period_key = lottery_data.period[-3:]
            record_data.setdefault(period_key, []).append(lottery_data)

        # 写回文件
        self.write_json_data_to_file(self.period_record_path, record_data, app_log=self.app_log)

    def analyze_same_weekday_numbers(self, data=None, **kwargs: Any) -> None:
        """
        分析同一星期的历史数据，并将新记录追加到 weekday_record_path 对应分组。
        """
        if data is None:
            return

        # 读取已有 JSON 记录
        if os.path.exists(self.weekday_record_path):
            record_data = self.read_json_data_from_file(self.weekday_record_path, app_log=self.app_log)
        else:
            record_data = {}

        for d in data:
            lottery_data = self.convert_lottery_data(d)
            if int(record_data.get('last_period', 0)) >= int(lottery_data.period):
                continue

            self.app_log.info(f"Update {lottery_data} to {self.weekday_record_path}")
            record_data['last_period'] = d[0]
            weekday_key = str(lottery_data.weekday)
            record_data.setdefault(weekday_key, []).append(lottery_data)

        self.write_json_data_to_file(self.weekday_record_path, record_data, app_log=self.app_log)

    def analyze_repeated_numbers(self, data=None, **kwargs: Any) -> None:
        """
        分析最近两期内重复出现的号码，并追加到 repeated_record_path。
        """
        if data is None:
            return

        if os.path.exists(self.repeated_record_path):
            record_data = self.read_json_data_from_file(self.repeated_record_path, app_log=self.app_log)
        else:
            record_data = {}

        for i, d in enumerate(data):
            # 需要前一期数据，i 从 1 开始
            if i == 0:
                continue
            sum_total     = int(data[-4]) if len(d) > self.origin_size else 0
            span          = int(data[-3]) if len(d) > self.origin_size else 0
            zone_ratio    = tuple(int(x) for x in str(data[-2]).split(':')) if len(d) > self.origin_size else ()
            odd_even_ratio= tuple(int(x) for x in str(data[-1]).split(':')) if len(d) > self.origin_size else ()
            common = [
                d[0],
                d[1],
                # self.calculate_weekday(d[1]),
                self.calculate_front(d),
                self.calculate_back(d),
                sum_total,
                span,
                zone_ratio,
                odd_even_ratio,
            ]
            if int(record_data.get('last_period', 0)) >= int(d[0]):
                continue

            # 计算前区和后区的重复号码
            prev = data[i - 1]
            repeated_front = self.calculate_repeated_numbers([
                self.calculate_front(prev),
                self.calculate_front(d)
            ])
            repeated_back = self.calculate_repeated_numbers([
                self.calculate_back(prev),
                self.calculate_back(d)
            ])
            common.extend([repeated_front, repeated_back])

            self.app_log.info(f"Update {common} to {self.repeated_record_path}")
            record_data['last_period'] = d[0]
            record_data.setdefault(d[0], []).extend(common)

        self.write_json_data_to_file(self.repeated_record_path, record_data, app_log=self.app_log)

    def analyze_edge_numbers(self, data=None, **kwargs: Any) -> None:
        """
        分析最近两期前后相邻(+1/-1)号码，并追加到 edge_record_path。
        """
        if data is None:
            return

        if os.path.exists(self.edge_record_path):
            record_data = self.read_json_data_from_file(self.edge_record_path, app_log=self.app_log)
        else:
            record_data = {}

        for i, d in enumerate(data):
            # 需要前一期数据，i 从 1 开始
            if i == 0:
                continue
            sum_total     = int(data[-4]) if len(d) > self.origin_size else 0
            span          = int(data[-3]) if len(d) > self.origin_size else 0
            zone_ratio    = tuple(int(x) for x in str(data[-2]).split(':')) if len(d) > self.origin_size else ()
            odd_even_ratio= tuple(int(x) for x in str(data[-1]).split(':')) if len(d) > self.origin_size else ()
            common = [
                d[0],
                d[1],
                # self.calculate_weekday(d[1]),
                self.calculate_front(d),
                self.calculate_back(d),
                sum_total,
                span,
                zone_ratio,
                odd_even_ratio,
            ]
            if int(record_data.get('last_period', 0)) >= int(d[0]):
                continue

            # 计算前区和后区的相邻号码集合
            prev = data[i - 1]
            edge_front = self.calculate_edge_numbers([
                self.calculate_front(prev),
                self.calculate_front(d)
            ])
            edge_back = self.calculate_edge_numbers([
                self.calculate_back(prev),
                self.calculate_back(d)
            ])
            common.extend([edge_front, edge_back])

            self.app_log.info(f"Update {common} to {self.edge_record_path}")
            record_data['last_period'] = d[0]
            record_data.setdefault(d[0], []).extend(common)

        self.write_json_data_to_file(self.edge_record_path, record_data, app_log=self.app_log)

    def analyze_cold_hot_numbers(self, data=None, **kwargs: Any) -> None:
        """
        分析冷热号码（最近4期出现与未出现），并追加到 cold_hot_record_path。
        """
        if data is None:
            return

        if os.path.exists(self.cold_hot_record_path):
            record_data = self.read_json_data_from_file(self.cold_hot_record_path, app_log=self.app_log)
        else:
            record_data = {}

        for i, d in enumerate(data):
            # 保证至少有4期历史
            if i < 4:
                continue
            sum_total     = int(data[-4]) if len(d) > self.origin_size else 0
            span          = int(data[-3]) if len(d) > self.origin_size else 0
            zone_ratio    = tuple(int(x) for x in str(data[-2]).split(':')) if len(d) > self.origin_size else ()
            odd_even_ratio= tuple(int(x) for x in str(data[-1]).split(':')) if len(d) > self.origin_size else ()
            common = [
                d[0],
                d[1],
                # self.calculate_weekday(d[1]),
                self.calculate_front(d),
                self.calculate_back(d),
                sum_total,
                span,
                zone_ratio,
                odd_even_ratio,
            ]
            if int(record_data.get('last_period', 0)) >= int(d[0]):
                continue

            # 取最近4期作为统计窗
            window = data[i-4:i]
            front_cold, front_hot = self.calculate_cold_hot_numbers(
                [self.calculate_front(x) for x in window],
                all_numbers=range(1, self.front_vocab_size + 1)
            )
            back_cold, back_hot = self.calculate_cold_hot_numbers(
                [self.calculate_back(x) for x in window],
                all_numbers=range(1, self.back_vocab_size + 1)
            )
            common.extend([(front_cold, front_hot), (back_cold, back_hot)])

            self.app_log.info(f"Update {common} to {self.cold_hot_record_path}")
            record_data['last_period'] = d[0]
            record_data.setdefault(d[0], []).extend(common)

        self.write_json_data_to_file(self.cold_hot_record_path, record_data, app_log=self.app_log)

    def analyze_omitted_numbers(self, data=None, **kwargs: Any) -> None:
        """
        分析遗漏号码统计（遗漏值、平均遗漏、最大遗漏、现期遗漏等），并追加到 omitted_record_path。
        """
        if data is None:
            return

        if os.path.exists(self.omitted_record_path):
            record_data = self.read_json_data_from_file(self.omitted_record_path, app_log=self.app_log)
        else:
            record_data = {}

        for i, d in enumerate(data):
            # 保证至少有10期历史
            if i < 9:
                continue
            sum_total     = int(data[-4]) if len(d) > self.origin_size else 0
            span          = int(data[-3]) if len(d) > self.origin_size else 0
            zone_ratio    = tuple(int(x) for x in str(data[-2]).split(':')) if len(d) > self.origin_size else ()
            odd_even_ratio= tuple(int(x) for x in str(data[-1]).split(':')) if len(d) > self.origin_size else ()
            common = [
                d[0],
                d[1],
                # self.calculate_weekday(d[1]),
                self.calculate_front(d),
                self.calculate_back(d),
                sum_total,
                span,
                zone_ratio,
                odd_even_ratio,
            ]
            if int(record_data.get('last_period', 0)) >= int(d[0]):
                continue

            # 取最近10期作为统计窗
            window = data[i-9:i+1]
            front_omitted = self.calculate_omitted_numbers(
                [self.calculate_front(x) for x in window],
                all_numbers=range(1, self.front_vocab_size + 1)
            )
            back_omitted = self.calculate_omitted_numbers(
                [self.calculate_back(x) for x in window],
                all_numbers=range(1, self.back_vocab_size + 1)
            )
            common.extend([front_omitted, back_omitted])

            self.app_log.info(f"Update {common} to {self.omitted_record_path}")
            record_data['last_period'] = d[0]
            record_data.setdefault(d[0], []).extend(common)

        self.write_json_data_to_file(self.omitted_record_path, record_data, app_log=self.app_log)

    """
    A pure virtual method inherited from util.CalculateUtil
    """

    def calculate_winning_amount(
        self,
        winning_number_combination: Union[NamedTuple, List[int], None],
        purchase_number_combinations: Union[List[NamedTuple], List[List[int]], None],
        **kwargs: Any
    ) -> Tuple[int, int]:
        """
        根据中奖号码和投注号码组合计算总中奖金额及中奖注数。

        参数:
          winning_number_combination (NamedTuple 或 List[int] 或 None):
            中奖号码组合，支持包含 front/back 属性的命名元组或纯号码列表。
          purchase_number_combinations (List[NamedTuple] 或 List[List[int]] 或 None):
            投注号码组合列表，格式同上。
          **kwargs: 其他可选参数（当前未使用）。

        返回:
          Tuple[int, int]:
            (总中奖金额, 中奖注数)
        """

        def single_winning_amount(
            _winning: Union[NamedTuple, List[int], None],
            _purchase: Union[NamedTuple, List[int], None]
        ) -> int:
            """
            计算单注的中奖奖金。

            规则:
              1. 提取中奖与投注号码的前区(front)和后区(back)列表；
              2. 计算前区和后区的交集大小(match_front, match_back)；
              3. 根据 (match_front, match_back) 在 award_rules 中查找奖金，缺省为 0。
            """
            # 提取中奖号码前后区
            if hasattr(_winning, 'front') and hasattr(_winning, 'back'):
                award_front, award_back = _winning.front, _winning.back
            else:
                award_front = self.calculate_front(_winning)
                award_back  = self.calculate_back(_winning)

            # 提取投注号码前后区
            if hasattr(_purchase, 'front') and hasattr(_purchase, 'back'):
                purchase_front, purchase_back = _purchase.front, _purchase.back
            else:
                purchase_front = self.calculate_front(_purchase)
                purchase_back  = self.calculate_back(_purchase)

            # 计算匹配数量
            match_front = len(set(award_front) & set(purchase_front))
            match_back  = len(set(award_back)  & set(purchase_back))

            # 返回对应奖金，默认为 0
            return self.award_rules.get((match_front, match_back), 0)

        total_amount = 0  # 累计中奖金额
        win_count    = 0  # 中奖注数

        # 遍历每注投注，累加奖金并统计中奖注数
        for combo in purchase_number_combinations or []:
            amt = single_winning_amount(winning_number_combination, combo)
            total_amount += amt
            if amt > 0:
                win_count += 1

        return total_amount, win_count

    """
    Static methods with no side effects
    """

    """
    other instance method
    """

    def download_data(self, force: bool = False) -> None:
        """
        spider full data and save data to history record file

        Parameters:
        - force: If True, files will write even if they already exist. Defaults to False.
        """

        if not os.path.exists(self.history_record_path) or force:
            history_data = SpiderUtil.dlt_sina_full_data()
            self.app_log.info(f'is saving history record to {self.history_record_path}')
            self.write_csv_data_to_file(self.history_record_path, data=history_data, app_log=self.app_log)
            self.analyze_same_period_numbers(history_data)
            self.analyze_same_weekday_numbers(history_data)
        else:
            print(f"{self.history_record_path} already exists in {self.dataset_dir}. Use force=True to overwrite.")

    def fetch_data(self, data=None):
        """
        fetch data. then update dataset

        :param data: New data to be added. If None, data will be fetched using Spider.get_history_data().
        :return: None
        """

        def load_recent_data(_data):
            _old_data = []
            if _data is None:
                if not os.path.exists(self.history_record_path):
                    return SpiderUtil.dlt_sina_full_data()

                try:
                    recent_data = SpiderUtil.dlt_sina_recent_data()
                except Exception as ex:
                    self.app_log.exception("Failed to spider recent data: {}".format(ex))
                    return None

                try:
                    _old_data = self.read_csv_data_from_file(self.history_record_path, app_log=self.app_log)
                    previous_data = _old_data[-1]
                except IndexError:
                    self.app_log.warning("History file is empty or does not exist.")
                    return None

                last_index = next((i for i, value in enumerate(recent_data) if value[0] == previous_data[0]), None)

                if last_index is None:
                    self.app_log.info("No matching data found in history.")
                    return None

                # Slice the history_data list to get the new relevant content
                _new_data = sorted(recent_data[last_index + 1:], key=lambda x: int(x[0]))
            elif isinstance(_data, str):
                _new_data = [_data.split()]
            else:
                _new_data = _data
            return _old_data, _new_data

        old_data, new_data = load_recent_data(data)
        if not new_data:
            self.app_log.info('Not found new data: {}'.format(new_data))
            return
        self.app_log.info('Updating history data: {}'.format(new_data))

        # update history record
        self.write_csv_data_to_file(self.history_record_path, data=new_data, app_log=self.app_log)

        # update analyze
        self.analyze_same_period_numbers(new_data)
        self.analyze_same_weekday_numbers(new_data)
        # self.analyze_repeated_numbers([*old_data[-1:], *new_data])
        # self.analyze_edge_numbers([*old_data[-1:], *new_data])
        # self.analyze_cold_hot_numbers([*old_data[-4:], *new_data])
        # self.analyze_omitted_numbers([*old_data[-9:], *new_data])

    @staticmethod
    def get_next_weekday(date_string: str = None) -> int:
        """
        获取下次开奖的星期几。
        开奖时间：
          - 每日 21:30 开奖
          - 若当前时间在 21:30 之前：
              - 星期三或星期四（weekday 2、3）返回 3
              - 星期五、星期六或星期日（weekday 4、5、6）返回 6
              - 其他（星期一、星期二）返回 1
          - 若当前时间在 21:30 之后：
              - 当天为周一～周三（weekday 0～2）返回 3
              - 当天为周四～周六（weekday 3～5）返回 6
              - 当天为周日（weekday 6）返回 1
        参数:
          date_string (str, optional): 日期字符串 "YYYY-MM-DD"，用于测试或回溯计算。默认 None 使用当前系统时间。
        返回:
          int: 下次开奖的星期标识 (1、3、6)。
        """
        # 解析时间: 测试模式用传入日期，否则用当前时间
        now = datetime.now() if date_string is None else datetime.strptime(date_string, "%Y-%m-%d")
        weekday = now.weekday()   # Python weekday: 周一=0, …, 周日=6
        hour, minute = now.hour, now.minute

        # 若在当日开奖前
        if (hour, minute) < (21, 30):
            if weekday in (2, 3):       # 周三、周四
                return 3
            elif weekday in (4, 5, 6):  # 周五、周六、周日
                return 6
            else:                       # 周一、周二
                return 1
        # 若已过当日开奖时间，计算下一个开奖日
        else:
            if weekday <= 2:            # 周一～周三
                return 3
            elif weekday <= 5:          # 周四～周六
                return 6
            else:                       # 周日
                return 1

    @staticmethod
    def get_next_period(date_string: str = None) -> int:
        """
        计算下次开奖在当年中的期号序号。
        - 每年的第一期开奖日为当年首个星期一/星期三/星期六。
        - 当日若在 21:30 之前，则当天不算开奖，开奖日向后顺推。
        - 统计从年初首期开奖到下次开奖之间经过的开奖日次数。
        参数:
          date_string (str, optional): 日期字符串 "YYYY-MM-DD"，用于测试或回溯计算。默认 None 使用当前系统时间。
        返回:
          int: 当年中下次开奖的序号（第一期为 1）。
        """
        # 解析时间
        now = datetime.now() if date_string is None else datetime.strptime(date_string, "%Y-%m-%d")
        current_year = now.year

        # 找到当年第一期的开奖日期（首个周一/周三/周六）
        first = datetime(current_year, 1, 1)
        while first.weekday() not in (0, 2, 5):
            first += timedelta(days=1)

        # 确定下次开奖日 datetime 对象
        if now.hour < 21 or (now.hour == 21 and now.minute < 30):
            # 若在开奖前，当天 21:30 为开奖时刻
            draw = now.replace(hour=21, minute=30, second=0, microsecond=0)
        else:
            # 已过开奖时刻，计算翌日或更后可开奖日
            draw = now.replace(hour=21, minute=30, second=0, microsecond=0) + timedelta(days=1)
            while draw.weekday() not in (0, 2, 5):
                draw += timedelta(days=1)

        # 计算从年初首期开奖到下次开奖的天数差
        delta_days = (draw - first).days

        # 每周三次开奖，先按整周累加
        periods = (delta_days // 7) * 3
        # 再统计余下天数内的开奖日
        for i in range((delta_days % 7) + 1):
            if (first + timedelta(days=i)).weekday() in (0, 2, 5):
                periods += 1

        return periods

    def get_kill_numbers(self, next_period: int = None, next_weekday: int = None, show_details: str = None) -> Tuple[Set[int], Set[int]]:
        """
        计算并返回前区与后区的杀号集合。

        参数：
          next_period  (int, optional): 下期期号；若为 None，则自动获取。
          next_weekday (int, optional): 下期星期几；若为 None，则自动获取。
          show_details (str, optional): 日志显示语言 'en' 或 'zh'。

        返回：
          Tuple[Set[int], Set[int]]: (前区杀码集合, 后区杀号集合)
        """
        # 1. 确定 next_weekday 与 next_period
        next_weekday = next_weekday or self.get_next_weekday()
        next_period  = next_period  or self.get_next_period()

        # 2. 读取并转换历史数据
        history_data = self.get_previous_history_data(next_period=next_period)
        period_data  = [self.convert_lottery_data(d) for d in self.get_previous_period_data(next_period=next_period)]
        weekday_data = [self.convert_lottery_data(d) for d in self.get_previous_weekday_data(next_period=next_period, next_weekday=next_weekday)]

        # 3. 计算前区杀码
        self.detail_log(self.app_log, show_details, en="front", zh="前区")
        front_kill_numbers = set()
        for data in (history_data, period_data):
            # 提取多条前区号码
            front_sequences = [self.calculate_front(d) for d in data]
            # 调用辅助方法计算杀号
            tmp = self.calculate_front_kills(front_sequences, next_period, next_weekday, show_details)
            # 过滤到合法范围后加入结果集
            front_kill_numbers.update({n for n in tmp if 1 <= n <= self.front_vocab_size})
        self.detail_log(
            self.app_log, show_details,
            en=f"front kill numbers: {sorted(front_kill_numbers)}, size: {len(front_kill_numbers)}",
            zh=f"前区杀码: {sorted(front_kill_numbers)}, 数量: {len(front_kill_numbers)}"
        )

        # 4. 计算后区杀码
        self.detail_log(self.app_log, show_details, en="back", zh="后区")
        back_kill_numbers = set()
        for data in (history_data, period_data, weekday_data):
            back_sequences = [self.calculate_back(d) for d in data]
            tmp = self.calculate_back_kills(back_sequences, next_weekday, show_details)
            back_kill_numbers.update({n for n in tmp if 1 <= n <= self.back_vocab_size})
        self.detail_log(
            self.app_log, show_details,
            en=f"back kill numbers: {sorted(back_kill_numbers)}, size: {len(back_kill_numbers)}",
            zh=f"后区杀码: {sorted(back_kill_numbers)}, 数量: {len(back_kill_numbers)}"
        )

        return front_kill_numbers, back_kill_numbers

    def calculate_front_kills(self, sequences: List[List[int]], next_period: int, next_weekday: int, show_details: str = None) -> Set[int]:
        """
        辅助函数：根据最后一条前区序列计算杀号。

        参数：
          sequences   (List[List[int]]): 多条前区号码序列。
          next_period (int): 下期期号，用于训练预测。
          next_weekday(int): 下期星期几，用于计算。
          show_details(str, optional): 日志显示语言标志。

        返回：
          Set[int]: 本次计算得到的杀号集合。
        """
        kills = set()
        last_sequence = sequences[-1]

        def train_predict(chunk, func, chunk_size=1):
            """
            训练式预测：对 chunk 分块并调用 func 生成预测结果集合。
            """
            train_data = [
                result
                for c, n in self.generate_chunks_with_next(chunk, chunk_size)
                for result in func(c[-1], n)
            ]
            if len(train_data) > 2:
                return {
                    self.exponential_moving_average_next_value(train_data),
                    self.linear_regression_next_value(train_data),
                    self.harmonic_regression_next_value(train_data)
                }
            return set()

        def update_zone_ratio_index_num_set_to_kills():
            """
            根据区间比 ratio 与奇偶比差值生成杀号。
            """
            oe_ratio   = self.calculate_odd_even_ratio(last_sequence)
            z_ratio    = self.calculate_zone_ratio(last_sequence, self.front_zone_ranges)
            idx_min    = min(n-1 if n else n for n in z_ratio)
            idx_max    = max(n-1 if n else n for n in z_ratio)
            dist       = self.calculate_euclidean_distance((z_ratio[0], z_ratio[-1]),(oe_ratio[0], oe_ratio[-1]))
            set_nums = {
                self.real_round(abs(last_sequence[idx_min] - last_sequence[idx_max])),
                self.real_round((abs(last_sequence[-idx_min-1] + last_sequence[-idx_max-1]) + dist) / next_weekday) % self.front_vocab_size
            }
            kills.update(set_nums)
            self.detail_log(
                self.app_log, show_details,
                en=f"zone ratio kills: {sorted(set_nums)}, size: {len(set_nums)}",
                zh=f"过滤区间比相关杀号: {sorted(set_nums)}, 数量: {len(set_nums)}"
            )

        def update_zone_average_set_to_kills():
            """
            根据各区段内号码平均值生成杀码。
            """
            avg_set = set()
            for zone in self.front_zone_ranges:
                tmp = [n for n in last_sequence if zone[0] <= n <= zone[1]]
                if len(tmp) == 1:
                    i = last_sequence.index(tmp[0])
                    tmp = [last_sequence[i-1], last_sequence[(i+1)%len(last_sequence)]]
                if tmp:
                    avg_set.add(self.real_round(sum(tmp)/next_weekday + len(tmp)) % self.front_vocab_size)
            kills.update(avg_set)
            self.detail_log(
                self.app_log, show_details,
                en=f"zone average kills: {sorted(avg_set)}, size: {len(avg_set)}",
                zh=f"过滤区间平均杀号: {sorted(avg_set)}, 数量: {len(avg_set)}"
            )

        def update_sum_total_average_set_to_kills():
            """
            根据和值与极值计算杀码。
            """
            total = self.calculate_sum_total(last_sequence)
            st_set = {
                self.real_round(total/next_weekday) % self.front_vocab_size,
                self.real_round((min(last_sequence)+max(last_sequence))/next_weekday)
                % self.front_vocab_size
            }
            kills.update(st_set)
            self.detail_log(
                self.app_log, show_details,
                en=f"sum total kills: {sorted(st_set)}, size: {len(st_set)}",
                zh=f"过滤和值杀号: {sorted(st_set)}, 数量: {len(st_set)}"
            )

        def update_span_set_to_kills():
            """
            根据跨度值计算杀码。
            """
            span = self.calculate_span(last_sequence)
            span_set = {
                abs(last_sequence[(span % self.front_size) % next_weekday] - span),
                (last_sequence[next_weekday % self.front_size] + span)
                % self.front_vocab_size
            }
            kills.update(span_set)
            self.detail_log(
                self.app_log, show_details,
                en=f"span kills: {sorted(span_set)}, size: {len(span_set)}",
                zh=f"过滤跨度杀号: {sorted(span_set)}, 数量: {len(span_set)}"
            )

        # 执行各类杀码更新函数
        update_zone_ratio_index_num_set_to_kills()
        update_zone_average_set_to_kills()
        update_sum_total_average_set_to_kills()
        update_span_set_to_kills()

        return kills

    def calculate_back_kills(self, sequences: List[List[int]], next_weekday: int, show_details: str = None) -> Set[int]:
        """
        辅助函数：根据最后一条后区序列计算杀号。

        参数：
          sequences   (List[List[int]]): 多条后区号码序列。
          next_weekday(int): 下期星期几，用于计算。
          show_details(str, optional): 日志显示语言标志。

        返回：
          Set[int]: 本次计算得到的后区杀号集合。
        """
        kills = set()
        last_sequence = sequences[-1]

        def update_zone_ratio_index_num_set_to_kills():
            """
            后区区间比率差值杀号。
            """
            oe_ratio = self.calculate_odd_even_ratio(last_sequence)
            z_ratio  = self.calculate_zone_ratio(last_sequence, self.back_zone_ranges)
            z_ratio_index = [min((n - 1 if n != 0 else n) for n in z_ratio),
                             max((n - 1 if n != 0 else n) for n in z_ratio)]
            print(oe_ratio, z_ratio)
            dist     = self.calculate_euclidean_distance((z_ratio[0], z_ratio[-1]),(oe_ratio[0], oe_ratio[-1]))
            print(last_sequence)
            num = self.real_round(abs(last_sequence[-(z_ratio_index[0]+1)] + last_sequence[-(z_ratio_index[-1]+1)])) % self.back_vocab_size
            kills.add(num)
            self.detail_log(
                self.app_log, show_details,
                en=f"back zone ratio kills: {num}",
                zh=f"后区区间比杀号: {num}"
            )

        def update_zone_average_set_to_kills():
            """
            后区区段平均杀号。
            """
            avg_set = set()
            for zone in self.back_zone_ranges:
                tmp = [n for n in last_sequence if zone[0] <= n <= zone[1]]
                if len(tmp) == 1:
                    i = last_sequence.index(tmp[0])
                    tmp = [last_sequence[i-1], last_sequence[(i+1)%len(last_sequence)]]
                if tmp:
                    avg_set.add(self.real_round(sum(tmp)/next_weekday) % self.back_vocab_size)
            kills.update(avg_set)
            self.detail_log(
                self.app_log, show_details,
                en=f"back zone average kills: {sorted(avg_set)}",
                zh=f"后区区段平均杀号: {sorted(avg_set)}"
            )

        update_zone_ratio_index_num_set_to_kills()
        update_zone_average_set_to_kills()

        return kills

    def get_banker_numbers(self, next_period: int = None, next_weekday: int = None, show_details: str = None) -> Tuple[Set[int], Set[int]]:
        """
        计算并返回前区与后区的胆码集合。

        参数与 get_kill_numbers 类似。
        返回：
          Tuple[Set[int], Set[int]]: (前区胆码集合, 后区胆码集合)
        """
        next_weekday = next_weekday or self.get_next_weekday()
        next_period  = next_period  or self.get_next_period()

        history_data = self.get_previous_history_data(next_period)
        period_data  = [self.convert_lottery_data(d) for d in self.get_previous_period_data(next_period)]
        weekday_data = [self.convert_lottery_data(d) for d in self.get_previous_weekday_data(next_period, next_weekday)]
        print(3, weekday_data)

        # 计算前区胆码
        front_banker_numbers = set()
        for data in (history_data, period_data, weekday_data):
            seqs = [self.calculate_front(d) for d in data]
            tmp = self.calculate_front_bankers(seqs, next_period, next_weekday, show_details)
            front_banker_numbers.update({n for n in tmp if 1 <= n <= self.front_vocab_size})
        self.detail_log(
            self.app_log, show_details,
            en=f"front banker numbers: {sorted(front_banker_numbers)}, size: {len(front_banker_numbers)}",
            zh=f"前区胆码: {sorted(front_banker_numbers)}, 数量: {len(front_banker_numbers)}"
        )

        # 计算后区胆码
        back_banker_numbers = set()
        for data in (period_data, weekday_data):
            seqs = [self.calculate_back(d) for d in data]
            tmp = self.calculate_back_bankers(seqs, next_weekday, show_details)
            back_banker_numbers.update({n for n in tmp if 1 <= n <= self.back_vocab_size})
        self.detail_log(
            self.app_log, show_details,
            en=f"back banker numbers: {sorted(back_banker_numbers)}, size: {len(back_banker_numbers)}",
            zh=f"后区胆码: {sorted(back_banker_numbers)}, 数量: {len(back_banker_numbers)}"
        )

        return front_banker_numbers, back_banker_numbers

    def calculate_front_bankers(self, sequences: List[List[int]], next_period: int, next_weekday: int, show_details: str = None) -> Set[int]:
        """
        辅助函数：根据最后一条前区序列计算胆码。

        参数与 calculate_front_kills 类似，返回胆码集合。
        """
        bankers = set()
        last_sequence = sequences[-1]

        def update_zone_ratio_index_num_set_to_bankers():
            """
            前区区间比率杀号逻辑，相似但结果作为胆码。
            """
            oe_ratio = self.calculate_odd_even_ratio(last_sequence)
            z_ratio = self.calculate_zone_ratio(last_sequence, self.front_zone_ranges)
            dist = self.calculate_euclidean_distance((z_ratio[0], z_ratio[-1]), (oe_ratio[0], oe_ratio[-1]))
            nums = {
                self.real_round(abs(last_sequence[z_ratio[0]-1] - last_sequence[z_ratio[-1]-1])),
                self.real_round((abs(last_sequence[-(z_ratio[0]+1)] + last_sequence[-(z_ratio[-1]+1)]) + dist) / next_weekday) % self.front_vocab_size
            }
            bankers.update(nums)
            self.detail_log(
                self.app_log, show_details,
                en=f"zone ratio bankers: {sorted(nums)}, size: {len(nums)}",
                zh=f"前区区间比胆码: {sorted(nums)}, 数量: {len(nums)}"
            )

        def update_zone_average_set_to_bankers():
            """
            前区区段平均胆码。
            """
            avg_set = set()
            for zone in self.front_zone_ranges:
                tmp = [n for n in last_sequence if zone[0] <= n <= zone[1]]
                if len(tmp) == 1:
                    i = last_sequence.index(tmp[0])
                    tmp = [last_sequence[i-1], last_sequence[(i+1)%len(last_sequence)]]
                if tmp:
                    avg_set.add(self.real_round(sum(tmp)/next_weekday + len(tmp)) % self.front_vocab_size)
            bankers.update(avg_set)
            self.detail_log(
                self.app_log, show_details,
                en=f"zone average bankers: {sorted(avg_set)}, size: {len(avg_set)}",
                zh=f"前区区段平均胆码: {sorted(avg_set)}, 数量: {len(avg_set)}"
            )

        # 执行胆码生成逻辑
        update_zone_ratio_index_num_set_to_bankers()
        update_zone_average_set_to_bankers()
        # 也可添加 sum_total、span 等逻辑

        return bankers

    def calculate_back_bankers(self, sequences: List[List[int]], next_weekday: int, show_details: str = None) -> Set[int]:
        """
        辅助函数：根据最后一条号码序列和下期星期几，计算后区胆码集合。

        参数：
          sequences   (List[List[int]]): 多条后区号码序列列表。
          next_weekday(int)            : 下期的星期几，用于索引计算。
          show_details(str, optional)  : 日志展示语言标志 'en' / 'zh'。

        返回：
          Set[int]: 计算得到的后区胆码集合。
        """

        def update_span_set_to_backers():
            # 计算前区跨度（调用 calculate_span）
            span = self.calculate_span(last_sequence)
            # 根据跨度和下期星期几生成候选胆码
            span_set = {
                # 绝对差值：当天位置号码与跨度的差的绝对值
                abs(last_sequence[next_weekday % self.back_size] - span),
                # 求和取模：当天位置号码与跨度之和再对后区 vocab_size 取模
                (last_sequence[next_weekday % self.back_size] + span) % self.back_vocab_size,
            }
            # 将候选胆码加入总集合
            bankers.update(span_set)
            # 如需展示详细日志，输出本次生成的跨度相关候选
            self.detail_log(
                app_log=self.app_log,
                show_details=show_details,
                en=f"span about add: {sorted(span_set)}, size: {len(span_set)}",
                zh=f"过滤跨度相关值: {sorted(span_set)}, 数量: {len(span_set)}"
            )

        # 初始化胆码集合
        bankers = set()
        # 取最后一条号码序列
        last_sequence = sequences[-1]

        # 调用内部函数生成跨度相关胆码
        update_span_set_to_backers()

        return bankers

    def get_previous_history_data(self, next_period: int = None) -> List[List[Any]]:
        """
        从历史 CSV 文件中读取开奖记录，并返回指定下期期号之前的所有数据。

        参数:
          next_period (int, optional): 下期期号，如 24001。若为 None，则返回全部历史记录。

        返回:
          List[List[Any]]: 原始数据列表，每项为一期开奖结果列表；出错时返回空列表。
        """
        try:
            # 1. 从文件读取完整历史记录
            history_data = self.read_csv_data_from_file(
                self.history_record_path,
                app_log=self.app_log
            )
            # 2. 若指定 next_period，则截取至该期之前的数据
            if next_period is None:
                return history_data[:]
            index = next(
                (i for i, row in enumerate(history_data)
                 if row[0] == str(next_period)),
                -1
            )
            return history_data[:index] if index != -1 else history_data[:]
        except Exception as ex:
            # 读取或解析失败时，返回空列表
            self.app_log.error(f"get_previous_history_data 异常: {ex}")
            return []

    def get_previous_period_data(self, next_period: int = None) -> List[List[Any]]:
        """
        从按期号分组的 JSON 文件中读取数据，并返回指定下期期号之前的同组数据。

        参数:
          next_period (int, optional): 下期期号，如 24001。若为 None，则使用当前 next_period 组别。

        返回:
          List[List[Any]]: 同一期号历史数据列表；出错时返回空列表。
        """
        try:
            # 1. 从文件读取按末两位期号分组的 JSON 数据
            period_data_map = self.read_json_data_from_file(
                self.period_record_path,
                app_log=self.app_log
            )
            # 2. 确定要使用的组别键（末两位期号，左填零至3位）
            key = (str(next_period % 100).zfill(3)
                   if next_period is not None
                   else str(self.get_next_period()).zfill(3))
            period_data = period_data_map.get(key, [])
            # 3. 若指定 next_period，则截取至该期之前的数据
            if next_period is not None:
                index = next(
                    (i for i, row in enumerate(period_data)
                     if row[0] == str(next_period)),
                    -1
                )
                period_data = period_data[:index] if index != -1 else period_data[:]
            return period_data
        except Exception as ex:
            self.app_log.error(f"get_previous_period_data 异常: {ex}")
            return []

    def get_previous_weekday_data(self, next_period: int = None, next_weekday: int = None) -> List[List[Any]]:
        """
        从按星期分类的 JSON 文件中读取数据，并返回指定星期及期号之前的记录。

        参数:
          next_period  (int, optional): 下期期号，如 24001。若指定，则截取该日期之前的数据。
          next_weekday (int, optional): 下期星期，如 1～7。若为 None，则使用当前 get_next_weekday()。

        返回:
          List[List[Any]]: 同一天星期的历史记录列表；出错时返回空列表。
        """
        try:
            # 1. 从文件读取按星期分类的 JSON 数据
            weekday_data_map = self.read_json_data_from_file(
                self.weekday_record_path,
                app_log=self.app_log
            )
            # 2. 确定要使用的星期键
            key = str(next_weekday) if next_weekday is not None else str(self.get_next_weekday())
            weekday_data = weekday_data_map.get(key, [])
            # 3. 若指定 next_period，则截取至该期之前的数据
            if next_period is not None:
                index = next((i for i, row in enumerate(weekday_data) if str(row[0]).endswith(str(next_period))), -1)
                weekday_data = weekday_data[:index] if index != -1 else weekday_data[:]
            return weekday_data
        except Exception as ex:
            self.app_log.error(f"get_previous_weekday_data 异常: {ex}")
            return []

    def convert_lottery_data(self, data: Union[List[Any], Lottery]) -> Lottery:
        """
        从原始列表解析并返回 Lottery 命名元组。
        列表格式示例（长度 ≥ 10）：
          [period, weekday, f1, f2, f3, f4, f5, b1, b2,
           sum_total, span, 'z0:z1:z2', 'o0:o1']
        字段含义：
          - period: 期号（int/str）；
          - weekday: 星期（int, 1～7）；
          - front: 前区 5 个号码（索引 2–6）；
          - back: 后区 2 个号码（索引 7–8）；
          - sum_total: 前区和值（倒数第4 项）；
          - span: 前区跨度（倒数第3 项）；
          - zone_ratio: 区段比例，形如 '2:1:2'（倒数第2 项）；
          - odd_even_ratio: 奇偶比例，形如 '3:2'（最后一项）。
        解析失败时回退按原始顺序 unpack。
        """
        try:
            if isinstance(data, self.Lottery):
                return data

            period = data[0]
            weekday = int(data[1]) if data[1] else ''
            front = data[2] if len(data) == 4 else self.calculate_front(data)
            back  = data[3] if len(data) == 4 else self.calculate_back(data)
            sum_total     = int(data[-4]) if len(data) > self.origin_size else 0
            span          = int(data[-3]) if len(data) > self.origin_size else 0
            zone_ratio    = tuple(int(x) for x in str(data[-2]).split(':')) if len(data) > self.origin_size else ()
            odd_even_ratio= tuple(int(x) for x in str(data[-1]).split(':')) if len(data) > self.origin_size else ()

            return self.Lottery(
                period=period,
                weekday=weekday,
                front=front,
                back=back,
                sum_total=sum_total,
                span=span,
                zone_ratio=zone_ratio,
                odd_even_ratio=odd_even_ratio
            )
        except Exception as ex:
            # 回退：直接按原始列表顺序 unpack
            return self.Lottery(*data)

    def calculate_front(self, data: Union[Lottery, Iterable[Any]]) -> List[int]:
        """
        根据预设尺寸规则提取数据的前区部分。
        支持 Lottery 对象或原始可迭代列表。
        """
        length = len(list(data))

        # 如果已是 Lottery 对象，直接返回 front 属性
        if isinstance(data, self.Lottery):
            return data.front

        # 新浪格式（含额外字段）截取中间前区
        if length == self.sina_size:
            trimmed = data[:self.origin_size - self.sina_size]
            return list(map(int, trimmed[self.back_size:-self.back_size]))

        # 完整原始格式（前区+后区）
        if length == self.origin_size:
            return list(map(int, data[self.back_size:-self.back_size]))

        # 普通格式：前区+后区拼接
        if length == self.normal_size:
            return list(map(int, data[:self.front_size]))

        # 纯前区列表
        if length == self.front_size:
            return list(map(int, data))

        # 只有后区时返回空
        if length == self.back_size:
            return []

        raise IndexError(f"calculate_front: length {length} 不在预期尺寸列表中")

    def calculate_back(self, data: Union[Lottery, Iterable[Any]]) -> List[int]:
        """
        根据预设尺寸规则提取数据的后区部分。
        支持 Lottery 对象或原始可迭代列表。
        """
        length = len(list(data))

        # 如果已是 Lottery 对象，直接返回 back 属性
        if isinstance(data, self.Lottery):
            return data.back

        # 新浪格式（含额外字段）截取末尾后区
        if length == self.sina_size:
            trimmed = data[:self.origin_size - self.sina_size]
            return list(map(int, trimmed[-self.back_size:]))

        # 完整原始格式或普通拼接格式
        if length in (self.origin_size, self.normal_size):
            return list(map(int, data[-self.back_size:]))

        # 纯后区列表
        if length == self.back_size:
            return list(map(int, data))

        # 只有前区时返回空
        if length == self.front_size:
            return []

        raise IndexError(f"calculate_back: length {length} 不在预期尺寸列表中")

    def revised_features(self, last_window_period_datas: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备修正后的特征数据结构，供预测算法使用。
        """
        def _feature_reviser(predictions: Counter, delta_size: int, feature_key: str) -> Counter:
            """
            根据 feature_key 和 delta_size 过滤预测结果。

            如果 feature_key 包含 'ratio'，只保留预测元组之和等于 delta_size 的项；
            否则保留所有预测结果。
            """
            if 'ratio' in feature_key:
                # 只保留和等于目标大小的预测
                return Counter({k: v for k, v in predictions.items() if sum(k) == delta_size})
            else:
                # 不包含 ratio，则返回所有预测
                return predictions

        # 对每个 feature 和 region 调用 _feature_reviser，返回修正后的预测集合
        return {
            feature: {
                region: _feature_reviser(
                    predicts,
                    self.front_size if region == 'front' else self.back_size,
                    feature
                )
                for region, predicts in item.get('predictions', {}).items()
            }
            for feature, item in last_window_period_datas.items()
        }

    def calculate_features(self, lottery_datas: List[Lottery], region: int = 3) -> Dict[str, Dict[str, List]]:
        """
        计算并返回前区/后区的各种特征数据。

        支持按照 feature_keys 中定义的所有特征方法逐一计算，
        并对“跨度”、“和值”等特殊方法做列表封装。
        """
        def _calculate_parser(compute_method: Callable, data: List[int], param: Optional[List] = None) -> Any:
            # 如果传入 param，则调用 compute_method(data, param)，否则 compute_method(data)
            return compute_method(data, param) if param is not None else compute_method(data)

        features: Dict[str, Dict[str, List]] = {}

        # 定义需要附带区段范围参数的特征
        add_ranges_params: Dict[str, Tuple[List, List]] = {
            'zone_ratio':        (self.front_zone_ranges, self.back_zone_ranges),
            'big_small_ratio':   (self.front_big_small_ranges, self.back_big_small_ranges),
            'road_012_ratio':    (self.front_road_012_ranges, self.back_road_012_ranges),
        }
        # 特殊方法列表：返回单值列表
        special_method_list = ['calculate_span', 'calculate_sum_total', 'calculate_sum_tail', 'calculate_ac']

        # 1. 普通特征计算
        for lottery_data in lottery_datas:
            for feature in self.feature_keys:
                compute_method: Callable = getattr(self, f'calculate_{feature}')
                params = add_ranges_params.get(feature)
                # region_mapping 提供前区/后区输入参数
                region_mapping = {
                    1: (lottery_data.front, params[0] if params else None),
                    2: (lottery_data.back,  params[1] if params else None)
                }

                # 前区计算
                if region & 1:
                    front_val = _calculate_parser(compute_method, *region_mapping[1])
                    features.setdefault(feature, {}).setdefault('front', []).append(
                        [front_val] if compute_method.__name__ in special_method_list
                        else [len(front_val)] if compute_method.__name__ == 'calculate_consecutive_numbers'
                        else front_val
                    )

                # 后区计算
                if region & 2:
                    back_val = _calculate_parser(compute_method, *region_mapping[2])
                    features.setdefault(feature, {}).setdefault('back', []).append(
                        [back_val] if compute_method.__name__ in special_method_list
                        else [len(back_val)] if compute_method.__name__ == 'calculate_consecutive_numbers'
                        else back_val
                    )

        # 2. 追加特征计算（如冷热遗漏）
        add_append_params: Dict[str, Tuple] = {
            'cold_hot_numbers': (range(1, self.front_vocab_size + 1), range(1, self.back_vocab_size + 1)),
            'omitted_numbers':  (range(1, self.front_vocab_size + 1), range(1, self.back_vocab_size + 1)),
        }
        for feature, chunk_size in self.append_feature_mapping.items():
            compute_method: Callable = getattr(self, f'calculate_{feature}')
            params = add_append_params.get(feature)
            # 按滑窗生成历史块与下期数据
            for chunk, next_element in self.generate_chunks_with_next(lottery_datas, chunk_size):
                region_mapping = {
                    1: ([c.front for c in chunk] + [next_element.front], params[0] if params else None),
                    2: ([c.back for c in chunk]  + [next_element.back],  params[1] if params else None)
                }
                if region & 1:
                    val = _calculate_parser(compute_method, *region_mapping[1])
                    features.setdefault(feature, {}).setdefault('front', []).append([len(val)])
                if region & 2:
                    val = _calculate_parser(compute_method, *region_mapping[2])
                    features.setdefault(feature, {}).setdefault('back', []).append([len(val)])

        return features

    def calculate_predictions(self, feature_results: Dict[str, Dict[str, List[List]]]) -> Dict[str, Dict[str, Any]]:
        """
        对每个特征和区段使用多种预测算法，生成预测结果集合。

        对包含 'ratio' 的特征，按元组形式存储预测；其它特征按单值形式存储。
        """
        predictions: Dict[str, Dict[str, Counter]] = {}

        for feature, item in feature_results.items():
            for region, matrix in item.items():
                # 转置矩阵，按列生成序列列表
                matrix_flip = [[matrix[row][col] for row in range(len(matrix))]
                               for col in range(len(matrix[0]))]

                # 依次应用三种预测算法
                for predictor in [
                    self.exponential_moving_average_next_value,
                    self.linear_regression_next_value,
                    self.random_forest_regressor_next_value
                ]:
                    # 初始化嵌套 Counter 容器
                    region_preds = predictions.setdefault(feature, {}).setdefault(region, Counter())

                    if 'ratio' in feature:
                        # ratio 特征：每列预测结果为元组
                        region_preds.update([tuple(predictor(seq) for seq in matrix_flip)])
                    else:
                        # 其它特征：直接将每列预测值计入 Counter
                        region_preds.update(predictor(seq) for seq in matrix_flip)

        return predictions

    def handle_last_window_data(self, data: Optional[List[List[Any]]], window: int = 10) -> Dict[str, Any]:
        """
        处理历史开奖数据的最后 window 条记录，计算各区块特征比率并生成下一期预测。

        参数:
          data (Optional[List[List[Any]]]): 历史开奖原始数据列表，每项是前区+后区号码列表。
          window (int): 需要处理的最近期数窗口大小，默认 10；若为 -1，则动态扩展至覆盖所有号码。

        返回:
          Dict[str, Any]: 经过修正的特征和对应预测结果的字典映射。
        """
        # 1. 检查输入数据
        if not data:
            raise ValueError("No data provided and no file reading implemented.")

        # 内部函数：根据索引长度和区段计算特征比率
        def _predict(index: int, zone: str = 'front') -> Dict[str, Dict[str, Any]]:
            # 取最后 index 条数据并转换为 Lottery 对象
            recent = [self.convert_lottery_data(row) for row in data[-index:]]
            # 不同区段调用 calculate_features 获取特征
            return (self.calculate_features(recent, region=1)
                    if zone == 'front'
                    else self.calculate_features(recent, region=2))

        # 2. 确定前区和后区的窗口大小
        if window == -1:
            front_ind = back_ind = 0
            front_set, back_set = set(), set()
            for idx, row in enumerate(reversed(data)):
                ld = self.convert_lottery_data(row)
                front_set.update(ld.front)
                back_set .update(ld.back)
                if front_ind == 0 and len(front_set) >= self.front_vocab_size:
                    front_ind = idx + 1
                if back_ind == 0 and len(back_set)  >= self.back_vocab_size:
                    back_ind  = idx + 1
                if front_ind and back_ind:
                    break
        else:
            front_ind = back_ind = window

        # 3. 分别计算前区/后区特征比率
        features_front = _predict(front_ind, zone='front')
        features_back  = _predict(back_ind,  zone='back')

        # 4. 合并前区与后区特征到一个字典
        features: Dict[str, Dict[str, Any]] = {
            feat: {
                'front': features_front.get(feat, {}).get('front'),
                'back' : features_back .get(feat, {}).get('back'),
            }
            for feat in set(features_front) | set(features_back)
        }

        # 5. 基于合并特征计算预测结果
        predictions: Dict[str, Any] = self.calculate_predictions(features)

        # 6. 构建特征 + 预测映射并做最终修正
        mapping: Dict[str, Dict[str, Any]] = {
            feat: {
                'features'   : features.get(feat),
                'predictions': predictions.get(feat)
            }
            for feat in set(features) | set(predictions)
        }
        result: Dict[str, Any] = self.revised_features(mapping)

        # 7. 返回修正后的特征与预测结果
        return result

    def predict_by_last_window_data(self, data: Optional[List[List[Any]]], window: int = 10, use_index: bool = False) -> List[Lottery]:
        """
        基于最近窗口数据调用多种预测模型，生成 Lottery 对象列表。

        参数:
          data (Optional[List[List[Any]]]): 历史开奖数据，每行是包含前区+后区号码的列表。
          window (int): 滑窗大小，默认 10。若为 -1，则动态根据前后区 vocab_size 补齐窗口。
          use_index (bool): 是否先将号码编码为索引再预测，默认 False。

        返回:
          List[Lottery]: 由预测结果生成的 Lottery NamedTuple 列表。
        """

        # 内部函数：根据指定索引长度和区段（前区/后区），调用各模型生成原始预测索引
        def _predict(index: int, zone: str = 'front'):
            # 将最后 index 条记录转换为 Lottery 对象列表
            last_window = [self.convert_lottery_data(row) for row in data[-index:]]

            if use_index:
                # 若使用索引模式，先对每期号码做编码
                if zone == 'front':
                    matrix = [self.encode_combination(l.front, max_n=self.front_vocab_size) for l in last_window]
                else:
                    matrix = [self.encode_combination(l.back,  max_n=self.back_vocab_size) for l in last_window]
                # 对每个模型函数，直接传入编码后的矩阵
                predictions = [(model_fn(matrix), ) for model_fn in model_functions]
            else:
                # 普通模式：提取原始号码列表
                if zone == 'front':
                    matrix = [l.front for l in last_window]
                else:
                    matrix = [l.back  for l in last_window]
                # 转置矩阵：按列生成序列
                matrix_flip = [[matrix[r][c] for r in range(len(matrix))] for c in range(len(matrix[0]))]
                # 对每个模型函数，传入每列序列，收集预测结果
                predictions = [tuple(model_fn(col_seq) for col_seq in matrix_flip) for model_fn in model_functions]
            return predictions

        # 1. 数据为空时抛错
        if not data:
            raise ValueError("No data provided and no file reading implemented.")

        # 2. 计算前区和后区各自的窗口大小
        if window == -1:
            # 动态寻找能覆盖所有号码的最小窗口
            front_ind = back_ind = 0
            front_set = set()
            back_set  = set()
            for idx, row in enumerate(reversed(data)):
                ld = self.convert_lottery_data(row)
                front_set.update(ld.front)
                back_set .update(ld.back)
                if front_ind == 0 and len(front_set) >= self.front_vocab_size:
                    front_ind = idx + 1
                if back_ind == 0 and len(back_set)  >= self.back_vocab_size:
                    back_ind = idx + 1
                if front_ind and back_ind:
                    break
        else:
            front_ind = back_ind = window

        # 3. 定义所有要使用的模型函数列表
        model_functions = [
            self.exponential_moving_average_next_value,
            self.linear_regression_next_value,
            self.random_forest_regressor_next_value,
        ]

        # 4. 调用内部预测函数，分别获取前区和后区的原始预测索引
        predictions_front = _predict(front_ind, zone='front')
        predictions_back  = _predict(back_ind,  zone='back')

        # 5. 将索引值裁剪到合法范围，并解码为实际号码
        predictions_all: List[List[int]] = []
        for i in range(len(model_functions)):
            # 保证索引在 [1, vocab_size] 范围
            idx_front = [min(self.front_vocab_size, max(1, x)) for x in predictions_front[i]]
            idx_back  = [min(self.back_vocab_size,  max(1, x)) for x in predictions_back[i]]
            if use_index:
                # 解码回前区和后区组合
                nums_front = self.decode_combination(idx_front[0], k=self.front_size, max_n=self.front_vocab_size)
                nums_back  = self.decode_combination(idx_back[0], k=self.back_size, max_n=self.back_vocab_size)
            else:
                nums_front = idx_front
                nums_back  = idx_back
            predictions_all.append(nums_front + nums_back)

        # 6. 将每组预测数字列表重新转换为 Lottery 对象输出
        return [self.convert_lottery_data(['', '', *nums]) for nums in predictions_all]

    def predict_by_last_period(
        self,
        next_period: Optional[int] = None,
        show_details: Optional[str] = None,
        window_size: int = 15,
        use_index: bool = False,
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        基于“上一期”历史数据窗口进行预测，
        返回多组预测号码及对应特征。

        参数:
          next_period (int, optional): 下期期号，例如 24001。
          show_details (str, optional): 日志显示语言标志，可选 'en' / 'zh'。
          window_size (int): 滑窗大小，默认 15。
          use_index (bool): 是否使用索引特征，默认 False。

        返回:
          Tuple[List[List[int]], Dict[str, Any]]:
            - predict_data: 多组预测号码，每组为前5+后2数字列表；
            - handle_result: handle_last_window_data 返回的特征字典。
        """
        # 1. 获取上一期历史开奖数据
        history_data = self.get_previous_history_data(next_period=next_period)

        # 2. 在历史数据上应用滑窗预测，得到 MaybeCombination 列表
        maybe_combinations = self.predict_by_last_window_data(data=history_data, window=window_size, use_index=use_index)

        # 3. 将 NamedTuple 的 front/back 合并为纯数字列表
        predict_data: List[List[int]] = [combo.front + combo.back for combo in maybe_combinations]

        # 4. 如需显示详情，打印所有预测号码
        self.detail_log(app_log=self.app_log, show_details=show_details, en=predict_data, zh=predict_data)

        # 5. 对历史数据做滑窗特征处理，获取特征字典
        handle_result: Dict[str, Any] = self.handle_last_window_data(data=history_data, window=window_size)

        # 6. 打印处理后得到的特征结果
        self.detail_log(app_log=self.app_log, show_details=show_details, en=handle_result, zh=handle_result)

        # 7. 返回预测号码与特征
        return predict_data, handle_result

    def predict_by_same_period(
        self,
        next_period: Optional[int] = None,
        show_details: Optional[str] = None,
        window_size: int = 15,
        use_index: bool = False,
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        基于“同期期号”历史数据窗口进行预测，
        返回多组预测号码及对应特征。

        参数:
          next_period (int, optional): 下期期号，例如 24001。
          show_details (str, optional): 日志显示语言标志，可选 'en' / 'zh'。
          window_size (int): 滑窗大小，默认 15。
          use_index (bool): 是否使用索引特征，默认 False。

        返回:
          Tuple[List[List[int]], Dict[str, Any]]:
            - predict_data: 多组预测号码，每组为前5+后2数字列表；
            - handle_result: handle_last_window_data 返回的特征字典。
        """
        # 1. 获取同一期号历史开奖数据
        period_data = self.get_previous_period_data(next_period=next_period)

        # 2. 在历史数据上应用滑窗预测
        maybe_combinations = self.predict_by_last_window_data(data=period_data, window=window_size, use_index=use_index)

        # 3. 合并 NamedTuple 的 front/back 为数字列表
        predict_data: List[List[int]] = [combo.front + combo.back for combo in maybe_combinations]

        # 4. 如需显示详情，打印所有预测号码
        self.detail_log(app_log=self.app_log, show_details=show_details, en=predict_data, zh=predict_data)

        # 5. 对历史数据做滑窗特征处理
        handle_result: Dict[str, Any] = self.handle_last_window_data(data=period_data, window=window_size)

        # 6. 打印处理后得到的特征结果
        self.detail_log(app_log=self.app_log, show_details=show_details, en=handle_result, zh=handle_result)

        # 7. 返回预测号码与特征
        return predict_data, handle_result

    def predict_by_last_weekday(
        self,
        next_weekday: Optional[int] = None,
        next_period: Optional[int] = None,
        show_details: Optional[str] = None,
        window_size: int = 15,
        use_index: bool = False,
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        基于“同星期”历史数据窗口进行预测，
        返回多组预测号码及对应的特征字典。

        参数:
          next_weekday (int, optional): 下期星期几，例如 1、3 或 6。
          next_period  (int, optional): 下期期号，例如 24001。
          show_details (str, optional): 是否输出详细日志，可选 'en' / 'zh'。
          window_size  (int): 滑窗大小，默认 15。
          use_index    (bool): 是否使用索引特征，默认 False。

        返回:
          Tuple[List[List[int]], Dict[str,Any]]:
            - predict_data: [[f1…f5,b1,b2], …] 多组预测结果
            - handle_result: handle_last_window_data 生成的特征字典
        """
        # 1. 获取指定星期的历史开奖数据列表
        weekday_data = self.get_previous_weekday_data(next_weekday=next_weekday, next_period=next_period)

        # 2. 基于历史数据按滑窗预测，返回 Lottery 对象列表
        maybe_combinations = self.predict_by_last_window_data(data=weekday_data, window=window_size, use_index=use_index)

        # 3. 将 Lottery.namedtuple 拆分为纯数字列表（前区 + 后区）
        predict_data: List[List[int]] = [combo.front + combo.back for combo in maybe_combinations]

        # 4. 如果需要详细展示，打印所有预测号码
        self.detail_log(app_log=self.app_log, show_details=show_details, en=predict_data, zh=predict_data)

        # 5. 对历史数据做滑窗特征处理，返回特征字典
        handle_result: Dict[str, Any] = self.handle_last_window_data(data=weekday_data, window=window_size)

        # 6. 打印处理后的特征结果
        self.detail_log(app_log=self.app_log, show_details=show_details, en=handle_result, zh=handle_result)

        # 7. 返回预测号码和特征字典
        return predict_data, handle_result

    def model_predict(
        self,
        next_period: Optional[int] = None,
        next_weekday: Optional[int] = None,
        show_details: Optional[str] = None,
        window_size: int = 15,
        use_index: bool = False,
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        基于三种策略（上一期、同期期、同星期）生成初步预测结果并汇总特征。

        参数:
          next_period (int, optional): 下期期号，例如 24001。
          next_weekday (int, optional): 下期星期几，例如 1、3、6。
          show_details (str, optional): 是否打印详细日志，可选 'en' / 'zh'。
          window_size (int): 用于建模的滑窗大小，默认 15。
          use_index (bool): 是否使用索引特征，默认 False。

        返回:
          Tuple[List[List[int]], Dict[str,Any]]:
            - predictions: 多组预测号码列表，每行 7 个数（前 5 后 2）。
            - features: 对应的特征字典，用于分析或训练。
        """
        predictions: List[List[int]] = []
        features: Dict[str, Any] = {}

        # (1) 按“上一期”策略预测
        p_lottery, p_feature = self.predict_by_last_period(next_period=next_period, show_details=show_details,
                                                           window_size=window_size, use_index=use_index)
        predictions.extend(p_lottery)
        features.update(p_feature)

        # (2) 按“同期期”策略预测
        p_lottery, p_feature = self.predict_by_same_period(next_period=next_period, show_details=show_details,
                                                           window_size=window_size, use_index=use_index)
        predictions.extend(p_lottery)
        features.update(p_feature)

        # (3) 按“同星期”策略预测
        p_lottery, p_feature = self.predict_by_last_weekday(next_weekday=next_weekday, next_period=next_period,
                                                            show_details=show_details, window_size=window_size,
                                                            use_index=use_index)
        predictions.extend(p_lottery)
        features.update(p_feature)

        # 4. 如果需要显示详情，则统计各号码出现频次并打印
        if show_details:
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en="The following is a preliminary forecast of the data analysis :",
                zh="以下是对数据分析的初步预测："
            )
            # 拆分前区 / 后区并计数
            front_nums = [n for row in predictions for n in row[:5]]
            back_nums  = [n for row in predictions for n in row[5:]]
            freq_front = Counter(front_nums)
            freq_back  = Counter(back_nums)
            sorted_front = sorted(freq_front.items(), key=lambda x: (-x[1], x[0]))
            sorted_back  = sorted(freq_back.items(), key=lambda x: (-x[1], x[0]))

            # 打印前区频次
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en="Counts of the front zone numbers sorted by frequency:",
                zh="按频次排序的前区号码计数："
            )
            for num, cnt in sorted_front:
                self.detail_log(
                    app_log=self.app_log, show_details=show_details,
                    en=f"Number {num}: {cnt} times",
                    zh=f"数字 {num}: {cnt} 次"
                )
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=f"Total distinct front numbers: {len(sorted_front)}",
                zh=f"前区号码种类总数：{len(sorted_front)}"
            )

            # 打印后区频次
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en="Counts of the back zone numbers sorted by frequency:",
                zh="按频次排序的后区号码计数："
            )
            for num, cnt in sorted_back:
                self.detail_log(
                    app_log=self.app_log, show_details=show_details,
                    en=f"Number {num}: {cnt} times",
                    zh=f"数字 {num}: {cnt} 次"
                )
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=f"Total distinct back numbers: {len(sorted_back)}",
                zh=f"后区号码种类总数：{len(sorted_back)}"
            )

            # 最终打印每行预测
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en="\nHere are the preliminary predictions:",
                zh="\n以下是初步预测："
            )
            for row in predictions:
                self.app_log.info(', '.join(map(str, row)))

        return predictions, features

    def analyze_predict(
        self,
        next_period: Optional[int] = None,
        next_weekday: Optional[int] = None,
        show_details: Optional[str] = None,
        window_size: int = 15
    ) -> List[List[int]]:
        """
        分析型预测流程：先用模型预测，再结合杀号、胆拖等分析规则，输出最终候选号码。

        参数:
          next_period (int, optional): 下期期号，例如 24001。
          next_weekday (int, optional): 下期星期几，例如 1、3 或 6。
          show_details (str, optional): 是否打印详细日志，可选 'en' / 'zh'。
          window_size (int): 用于分析的滑窗大小，默认 15。

        返回:
          List[List[int]]: 最终候选号码列表（每行一组号码）。
        """
        # 1. 初步预测：调用 model_predict 并记录日志
        self.detail_log(
            app_log=self.app_log, show_details=show_details,
            en="Make initial predictions using model predictions",
            zh="使用模型预测进行初步预测"
        )
        predict_data, feature_data = self.model_predict(next_period=next_period, next_weekday=next_weekday,
                                                        show_details=None, window_size=window_size)

        # 2. 将多次模型结果按序分组：3 组，每组 3 条
        predict_sets = [[set(), set()] for _ in range(3)]
        pred_front_list, pred_back_list = [], []
        for idx, row in enumerate(predict_data):
            # 2.1 逐行记录预测结果
            nums_str = ', '.join(f'{n:02d}' for n in row)
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=nums_str, zh=nums_str
            )
            # 2.2 拆分前区和后区
            front_nums = self.calculate_front(row)
            back_nums  = self.calculate_back(row)
            # 2.3 按索引分配到三组
            group = idx % 3
            predict_sets[group][0].update(front_nums)
            predict_sets[group][1].update(back_nums)
            pred_front_list.append(front_nums)
            pred_back_list.append(back_nums)

        # 3. 合并三组的所有候选号码
        all_front = set().union(*(s[0] for s in predict_sets))
        all_back  = set().union(*(s[1] for s in predict_sets))
        self.detail_log(
            app_log=self.app_log, show_details=show_details,
            en=f"model predict front: {sorted(all_front)}, size: {len(all_front)}",
            zh=f"模型预测前区数字: {sorted(all_front)}, 数量: {len(all_front)}"
        )
        self.detail_log(
            app_log=self.app_log, show_details=show_details,
            en=f"model predict back: {sorted(all_back)}, size: {len(all_back)}",
            zh=f"模型预测后区数字: {sorted(all_back)}, 数量: {len(all_back)}"
        )

        # 4. 对每组分别打印详细候选
        for i, (fset, bset) in enumerate(predict_sets):
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=f"model{i+1} predict front: {sorted(fset)}, size: {len(fset)}",
                zh=f"模型{i+1}预测前区数字: {sorted(fset)}, 数量: {len(fset)}"
            )
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=f"model{i+1} predict back: {sorted(bset)}, size: {len(bset)}",
                zh=f"模型{i+1}预测后区数字: {sorted(bset)}, 数量: {len(bset)}"
            )

        # 5. 获取历史开奖数据，用于后续杀号、定胆
        history = self.get_previous_history_data(next_period=next_period)

        # 6. 杀码：获取不宜出现的号码集合
        kill_front, kill_back = set(), set()
        kills = self.get_kill_numbers(
            next_period=next_period,
            next_weekday=next_weekday,
            show_details=None
        )
        kill_front.update(kills[0])
        kill_back .update(kills[1])

        # 7. 胆码：获取强力推荐的号码集合
        banker_front, banker_back = set(), set()
        bankers = self.get_banker_numbers(
            next_period=next_period,
            next_weekday=next_weekday,
            show_details=None
        )
        banker_front.update(bankers[0])
        banker_back .update(bankers[1])

        # 8. 最终分析：统计频次并输出矩阵视图
        if show_details:
            # 8.1 日志：杀号、定胆
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=f"front kill numbers: {sorted(kill_front)}, size: {len(kill_front)}",
                zh=f"前区杀码: {sorted(kill_front)}, 数量: {len(kill_front)}"
            )
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=f"back kill numbers: {sorted(kill_back)}, size: {len(kill_back)}",
                zh=f"后区杀码: {sorted(kill_back)}, 数量: {len(kill_back)}"
            )
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=f"front banker numbers: {sorted(banker_front)}, size: {len(banker_front)}",
                zh=f"前区胆码: {sorted(banker_front)}, 数量: {len(banker_front)}"
            )
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=f"back banker numbers: {sorted(banker_back)}, size: {len(banker_back)}",
                zh=f"后区胆码: {sorted(banker_back)}, 数量: {len(banker_back)}"
            )
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=f"last combination: {history[-1]}",
                zh=f"最近一期中奖数据: {history[-1]}"
            )

            # 8.2 统计模型所有预测行中每个号码出现次数
            flat_front = [n for row in predict_data for n in row[:5]]
            flat_back  = [n for row in predict_data for n in row[5:]]
            freq_front = Counter(flat_front)
            freq_back  = Counter(flat_back)
            sorted_f = sorted(freq_front.items(), key=lambda x: (-x[1], x[0]))
            sorted_b = sorted(freq_back.items(), key=lambda x: (-x[1], x[0]))

            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en="Counts of front zone numbers sorted by freq:",
                zh="按频次排序的前区号码计数:"
            )
            for num, cnt in sorted_f:
                self.detail_log(
                    app_log=self.app_log, show_details=show_details,
                    en=f"Number {num}: {cnt} times",
                    zh=f"数字 {num}: {cnt} 次"
                )
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=f"Total distinct front numbers: {len(sorted_f)}",
                zh=f"前区号码种类总数: {len(sorted_f)}"
            )

            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en="Counts of back zone numbers sorted by freq:",
                zh="按频次排序的后区号码计数:"
            )
            for num, cnt in sorted_b:
                self.detail_log(
                    app_log=self.app_log, show_details=show_details,
                    en=f"Number {num}: {cnt} times",
                    zh=f"数字 {num}: {cnt} 次"
                )
            self.detail_log(
                app_log=self.app_log, show_details=show_details,
                en=f"Total distinct back numbers: {len(sorted_b)}",
                zh=f"后区号码种类总数: {len(sorted_b)}"
            )

            # 8.3 打印矩阵视图：最近一期 + 杀码 + 每行预测
            self.print_matrix(
                [self.calculate_front(history[-1])] + [list(kill_front)] + pred_front_list,
                self.front_vocab_size
            )
            self.print_matrix(
                [self.calculate_back(history[-1])] + [list(kill_back)] + pred_back_list,
                self.back_vocab_size
            )

        # 返回空列表或可根据策略过滤后的最终组合
        return []

    def llm_predict(
        self,
        next_period: Optional[int] = None,
        next_weekday: Optional[int] = None,
        show_details: Optional[str] = None,
        window_size: int = 15,
        output_dir: Optional[str] = None,
    ) -> List[List[int]]:
        def get_datas(_data, _size):
            datas = []
            for row in _data[-_size:]:
                ld = self.convert_lottery_data(row)
                f3 = f'{ld.zone_ratio[0]}:{ld.zone_ratio[1]}:{ld.zone_ratio[2]}'
                f4 = f'{ld.odd_even_ratio[0]}:{ld.odd_even_ratio[1]}'
                b1 = self.calculate_sum_total(ld.back)
                b2 = self.calculate_span(ld.back)
                b3 = self.calculate_zone_ratio(ld.back, zone_ranges=self.back_zone_ranges)
                b3 = f'{b3[0]}:{b3[1]}'
                b4 = self.calculate_odd_even_ratio(ld.back)
                b4 = f'{b4[0]}:{b4[1]}'
                d = (f"{ld.period}|{ld.weekday}|{','.join([str(d) for d in ld.front])}|{','.join([str(d) for d in ld.back])}|"
                     f"{ld.sum_total}|{ld.span}|{f3}|{f4}|{b1}|{b2}|{b3}|{b4}")
                datas.append(d)
            return datas

        output_dir = output_dir or "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # 按“上一期”策略预测
        history_data = self.get_previous_history_data(next_period=next_period)
        # 期号|星期|前区|后区|前区和值|前区跨度|前区区间比|前区奇偶比|后区和值|后区跨度|后区区间比|后区奇偶比
        datas = get_datas(history_data, window_size)
        PromptUtil.generate_ds_prompt('\n'.join(datas), 'history', output_dir)
        PromptUtil.generate_gpt_prompt('\n'.join(datas), 'history', output_dir)

        # 按“同期期”策略预测
        period_data = self.get_previous_period_data(next_period=next_period)
        datas = get_datas(period_data, window_size)
        PromptUtil.generate_ds_prompt('\n'.join(datas), 'period', output_dir)
        PromptUtil.generate_gpt_prompt('\n'.join(datas), 'period', output_dir)

        # 按“同星期”策略预测
        weekday_data = self.get_previous_weekday_data(next_weekday=next_weekday, next_period=next_period)
        datas = get_datas(weekday_data, window_size)
        PromptUtil.generate_ds_prompt('\n'.join(datas), 'weekday', output_dir)
        PromptUtil.generate_gpt_prompt('\n'.join(datas), 'weekday', output_dir)

        return []

    def predict(
        self,
        next_period: int = None,
        next_weekday: int = None,
        show_details: str = 'zh',
        window_size: int = 15,
        predict_type: str = 'model'
    ) -> List[List[int]]:
        """
        根据指定的预测类型，生成下一期号码的预测结果。

        参数:
          next_period (int, optional): 下期期号，缺省为 None。
          next_weekday (int, optional): 下期星期几，缺省为 None。
          show_details (str, optional): 是否展示详细信息，可选 'en' 或 'zh'，缺省不展示。
          window_size (int): 用于预测的滑窗大小，默认为 15。
          predict_type (str): 预测方式，可选 'model' 或 'analyze'，默认为 'model'。

        返回:
          List[List[int]]: 预测结果列表，每个子列表是一组预测号码。

        异常:
          KeyError: 当传入的 predict_type 不在支持列表中时抛出。
        """
        # 支持的预测类型与对应方法映射
        types: Dict[str, Callable] = {
            'model': self.model_predict,
            'analyze': self.analyze_predict,
            'llm': self.llm_predict,
        }

        # 检查用户指定的预测类型是否被支持
        if predict_type not in types:
            raise KeyError(f"不支持的预测类型：'{predict_type}'。可选值：{list(types.keys())}")

        # 根据预测类型调用对应方法，传入公共参数
        predictions = types[predict_type](next_period=next_period,next_weekday=next_weekday,show_details=show_details,window_size=window_size)

        # 返回最终预测结果
        return predictions

    def test(self):
        pass
