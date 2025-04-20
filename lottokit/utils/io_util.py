#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : nickdecodes
@Email   : nickdecodes@163.com
@Usage   :
@FileName: io_util.py
@DateTime: 2024/7/22 10:08
@SoftWare: PyCharm
"""

import sys
import os
import csv
import json
import logging
import logging.handlers
from typing import List, Any, Optional, NamedTuple, Union, Tuple


class IOUtil:
    @classmethod
    def get_logger(cls, log_file: str = None) -> logging.Logger:
        """
        获取并配置日志记录器。
        :param log_file: 日志文件路径；若为 None，则输出到终端。
        :return: 配置好的 logging.Logger 对象。
        """
        # 根据 log_file 生成唯一 logger 名称
        logger_name = log_file if log_file is not None else 'default_logger'
        logger = logging.getLogger(logger_name)

        # 若未添加过 handler，则进行初始化配置
        if not logger.handlers:
            if log_file is None:
                # 输出到终端
                fmt = '%(message)s'
                handler = logging.StreamHandler(sys.stdout)
            else:
                # 文件按午夜轮转，保留最近 7 天
                fmt = ('%(asctime)s [%(levelname)s] %(process)d %(thread)d '  
                       '%(filename)s <%(lineno)d> %(funcName)s: %(message)s')
                handler = logging.handlers.TimedRotatingFileHandler(
                    log_file, when='midnight', backupCount=7
                )

            # 设置格式与级别
            handler.setFormatter(logging.Formatter(fmt))
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    @classmethod
    def detail_log(
        cls,
        app_log: logging.Logger,
        show_details: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        根据语言标志输出详细日志。
        :param app_log: 日志对象。
        :param show_details: 语言标志 'en' 或 'zh'；若为 None，不输出。
        :param kwargs: 包含 'en' 和 'zh' 两种语言的消息。
        """
        if show_details:
            msg = kwargs.get(show_details, '')
            if msg:
                app_log.info(msg)

    @classmethod
    def print_matrix(
        cls,
        appeared_sequences: List[List[int]],
        matrix_size: int
    ) -> None:
        """
        按矩阵形式打印出现序列。
        :param appeared_sequences: 若序列中包含数字，则打印该数字；否则打印 '-'。
        :param matrix_size: 矩阵列数（1 到 matrix_size）。
        """
        for seq in appeared_sequences:
            for i in range(1, matrix_size + 1):
                print(f"{i:>4}" if i in seq else "   -", end=" ")
            print()

    @classmethod
    def write_data_to_file(
        cls,
        file_path: str,
        data: List[str],
        app_log: Optional[logging.Logger] = None,
        mode: str = 'a+',
        **kwargs: Any
    ) -> bool:
        """
        将文本数据写入文件，每行追加换行符。
        :param file_path: 目标文件路径。
        :param data: 字符串列表，每项写为一行。
        :param app_log: 可选日志对象；若 None 则新建默认 logger。
        :param mode: 打开模式，缺省 'a+'。
        :return: 写入成功返回 True，否则 False。
        """
        if app_log is None:
            app_log = cls.get_logger()
        if not data:
            app_log.warning("无数据可写入")
            return False

        try:
            with open(file_path, mode) as fp:
                app_log.debug(f"写入文件: {fp.name}")
                for line in data:
                    if line is not None:
                        fp.write(f"{line}\n")
        except Exception as ex:
            app_log.exception(f"写入文件时出错: {ex}")
            return False

        return True

    @classmethod
    def read_data_from_file(
        cls,
        file_path: str,
        app_log: Optional[logging.Logger] = None,
        mode: str = 'r'
    ) -> Optional[List[str]]:
        """
        从文本文件读取非空行，返回字符串列表。
        :param file_path: 待读取文件路径。
        :param app_log: 可选日志对象；若 None 则新建默认 logger。
        :param mode: 打开模式，缺省 'r'。
        :return: 非空行列表；发生异常时返回 None。
        """
        if app_log is None:
            app_log = cls.get_logger()

        try:
            with open(file_path, mode) as fp:
                app_log.debug(f"打开文件: {fp.name}")
                lines = [line.strip() for line in fp if line.strip()]
            app_log.info(f"读取行数: {len(lines)}")
            return lines
        except Exception as ex:
            app_log.exception(f"读取文件时出错: {ex}")
            return None

    @classmethod
    def write_csv_data_to_file(
        cls,
        file_path: str,
        data: List[List[Any]],
        app_log: Optional[logging.Logger] = None,
        mode: str = 'a+',
        newline: str = '',
        **kwargs: Any
    ) -> bool:
        """
        将二维列表写入 CSV 文件。
        :param file_path: CSV 文件路径。
        :param data: 二维列表，每项为一行。
        :param app_log: 可选日志对象；若 None 则新建默认 logger。
        :param mode: 打开模式，缺省 'a+'。
        :param newline: 控制换行，缺省 ''。
        :return: 写入成功返回 True，否则 False。
        """
        if app_log is None:
            app_log = cls.get_logger()
        if not data:
            app_log.warning("无 CSV 数据可写入")
            return False

        try:
            with open(file_path, mode=mode, newline=newline) as fp:
                app_log.debug(f"写入 CSV 文件: {fp.name}")
                writer = csv.writer(fp, **kwargs)
                writer.writerows(data)
        except Exception as ex:
            app_log.exception(f"写入 CSV 时出错: {ex}")
            return False

        return True

    @classmethod
    def read_csv_data_from_file(
        cls,
        file_path: str,
        app_log: Optional[logging.Logger] = None,
        mode: str = 'r',
        **kwargs: Any
    ) -> Optional[List[List[str]]]:
        """
        从 CSV 文件读取所有行，返回行列表。
        :param file_path: CSV 文件路径。
        :param app_log: 可选日志对象；若 None 则新建默认 logger。
        :param mode: 打开模式，缺省 'r'。
        :return: 行列表；发生异常时返回 None。
        """
        if app_log is None:
            app_log = cls.get_logger()

        try:
            with open(file_path, mode=mode, **kwargs) as fp:
                app_log.debug(f"读取 CSV 文件: {fp.name}")
                reader = csv.reader(fp, **kwargs)
                data = [row for row in reader]
            return data
        except Exception as ex:
            app_log.exception(f"读取 CSV 时出错: {ex}")
            return None

    @classmethod
    def write_json_data_to_file(
        cls,
        file_path: str,
        data: Any,
        app_log: Optional[logging.Logger] = None,
        mode: str = 'w',
        **kwargs: Any
    ) -> bool:
        """
        将数据写入 JSON 文件，格式化输出。
        :param file_path: JSON 文件路径。
        :param data: 待写入的数据（通常为 dict 或 list）。
        :param app_log: 可选日志对象；若 None 则新建默认 logger。
        :param mode: 打开模式，缺省 'w'（覆盖写入）。
        :return: 写入成功返回 True，否则 False。
        """
        if app_log is None:
            app_log = cls.get_logger()
        app_log.debug(f"写入 JSON 文件: {file_path}")

        try:
            with open(file_path, mode, encoding='utf-8') as fp:
                json.dump(data, fp, ensure_ascii=False, indent=4, **kwargs)
        except Exception as ex:
            app_log.exception(f"写入 JSON 时出错: {ex}")
            return False

        return True

    @classmethod
    def read_json_data_from_file(
        cls,
        file_path: str,
        app_log: Optional[logging.Logger] = None,
        mode: str = 'r',
        **kwargs: Any
    ) -> Optional[Any]:
        """
        从 JSON 文件中读取数据并返回解析后的对象。
        :param file_path: JSON 文件路径。
        :param app_log: 可选日志对象；若 None 则新建默认 logger。
        :param mode: 打开模式，缺省 'r'。
        :return: 解析后的对象；发生异常时返回 None。
        """
        if app_log is None:
            app_log = cls.get_logger()

        try:
            with open(file_path, mode, encoding='utf-8', **kwargs) as fp:
                app_log.debug(f"读取 JSON 文件: {file_path}")
                return json.load(fp)
        except Exception as ex:
            app_log.exception(f"读取 JSON 时出错: {ex}")
            return None
