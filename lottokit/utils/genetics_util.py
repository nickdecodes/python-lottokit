#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : nickdecodes
@Email   : nickdecodes@163.com
@Usage   :
@FileName: genetics_util.py
@DateTime: 2024/7/22 10:23
@SoftWare: PyCharm
"""

import random
import numpy as np


class GeneticsUtil:
    """
    遗传算法工具类：
      - 初始化种群与参考矩阵
      - 适应度评估（对角线、垂直对齐）
      - 锦标赛选择、两点交叉、基因变异
      - 遗传算法主循环
    """
    def __init__(
        self,
        refers: list,
        data: list,
        min_num: int,
        max_num: int,
        fitness_params: dict,
        population_size: int = 100,
        num_genes: int = 5,
        tournament_size: int = 3,
        mutation_rate: float = 0.1,
        epoch_count: int = 100
    ):
        """
        初始化遗传算法参数并生成初始种群与参考矩阵。
        :param refers: 参考数据列表，用于创建参考矩阵
        :param data: 候选基因池
        :param min_num: 基因最小值
        :param max_num: 基因最大值
        :param fitness_params: 适应度调整参数字典
        :param population_size: 种群规模
        :param num_genes: 每个个体基因数量
        :param tournament_size: 锦标赛选择规模
        :param mutation_rate: 基因突变概率
        :param epoch_count: 迭代代数
        """
        self.refers = refers
        self.data = data
        self.min_num = min_num
        self.max_num = max_num
        self.fitness_params = fitness_params
        self.population_size = population_size
        self.num_genes = num_genes
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.epoch_count = epoch_count
        # 初始化种群
        self.population = self.init_population()
        # 创建参考矩阵
        self.refers_matrix = self.create_refers_matrix()

    def init_population(self) -> list:
        """
        随机生成初始种群，每个个体从 data 中不重复抽取 num_genes 个基因，并排序。
        :return: 种群列表（二维数组）
        """
        pop = [
            np.random.choice(self.data, self.num_genes, replace=False)
            for _ in range(self.population_size)
        ]
        # 对每个个体的基因排序
        return [np.sort(row) for row in pop]

    def create_refers_matrix(self) -> list:
        """
        根据 refers 数据生成二进制参考矩阵。
        每行长度为 max_num，若 num 在 row 中则置 1，否则为 0。
        :return: 二维二进制矩阵
        """
        matrix = []
        for row in self.refers:
            vec = [0] * self.max_num
            for num in row:
                if self.min_num <= num <= self.max_num:
                    vec[num - 1] = 1
            matrix.append(vec)
        return matrix

    def check_diagonal(self, individual: list, dfs_size: tuple = (3, 3)) -> float:
        """
        检查个体基因在矩阵中对角线方向的对齐情况，返回调整后的适应度因子。
        :param individual: 基因列表
        :param dfs_size: 检查区域 (行, 列) 大小
        :return: 适应度因子
        """
        fitness_value = 1.0
        pre_index = individual[0] - 1
        rows = len(self.refers_matrix)
        dfs_row, dfs_col = dfs_size
        target = min(dfs_size)
        is_square = dfs_row == dfs_col

        for indi in individual:
            # 越界基因惩罚
            if not (self.min_num <= indi <= self.max_num):
                return 1.0

            c = indi - 1
            # 构建新矩阵：在底部添加当前基因对应行
            new_mat = [row[:] for row in self.refers_matrix]
            new_row = [0] * self.max_num
            new_row[c] = 1
            new_mat.append(new_row)

            # 仅检查上行对角线
            for dc in (-1, 1):
                count = 0
                r, col = rows, c
                while rows - dfs_row <= r <= rows and pre_index - dfs_col <= col <= c + dfs_col:
                    if new_mat[r][col] == 1:
                        count += 1
                    # 移动方向：方形或交替模式
                    if is_square:
                        r -= 1
                        col += dc
                    else:
                        # 交替移行或移列
                        pass  # 略去交替逻辑细节
                # 根据匹配数调整适应度
                if is_square and count == target:
                    fitness_value *= 0.1
                elif not is_square and count == target - 1:
                    fitness_value *= 0.3

            pre_index = c

        return fitness_value

    def check_vertical(self, individual: list) -> float:
        """
        检查个体基因在矩阵中垂直方向的对齐情况，返回调整后的适应度因子。
        :param individual: 基因列表
        :return: 适应度因子
        """
        fitness_value = 1.0
        rows = len(self.refers_matrix)

        for indi in individual:
            if not (self.min_num <= indi <= self.max_num):
                return 1.0

            c = indi - 1
            # 构建新矩阵
            new_mat = [row[:] for row in self.refers_matrix]
            new_row = [0] * self.max_num
            new_row[c] = 1
            new_mat.append(new_row)

            # 仅向上检查垂直线
            count = 0
            r = rows
            while r >= 0:
                if new_mat[r][c] == 1:
                    count += 1
                r -= 1
            # 根据垂直对齐数量调整适应度
            if count in (1, 2):
                fitness_value *= 0.1
            else:
                fitness_value *= 0.7

        return fitness_value

    def fitness(self, individual: list, dfs_size: tuple = (2, 2)) -> float:
        """
        计算个体总体适应度：
          - 重复基因或越界罚分
          - 对角线与垂直对齐检测
        :param individual: 基因列表
        :param dfs_size: 检查区域大小
        :return: 适应度值（越低越好）
        """
        # 重复基因高罚分
        if len(set(individual)) < self.num_genes:
            return 10000.0

        fitness_value = 1.0
        # 对角线检查
        fitness_value *= self.check_diagonal(individual, dfs_size)
        # 垂直检查
        fitness_value *= self.check_vertical(individual)
        return fitness_value

    def tournament_selection(self) -> np.ndarray:
        """
        锦标赛选择：随机抽取 tournament_size 个体，取适应度最高者。
        :return: 选出的个体（排序后）
        """
        contenders = random.sample(self.population, self.tournament_size)
        # 选择适应度最优（fitness 最低）
        winner = min(contenders, key=self.fitness)
        return np.sort(winner)

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        两点交叉：
          - 随机选取两个交叉点
          - 拼接父代基因段生成子代
        :return: 子代个体
        """
        a, b = sorted(random.sample(range(self.num_genes), 2))
        child = np.concatenate([
            parent1[:a],
            parent2[a:b],
            parent1[b:]
        ])
        return child

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        基因突变：
          - 以 mutation_rate 概率随机替换一个基因
          - 再排序保证个体有序
        :return: 突变后个体
        """
        if random.random() < self.mutation_rate:
            idx = random.randrange(self.num_genes)
            individual[idx] = random.choice(self.data)
        return np.sort(individual)

    def genetic(self) -> list:
        """
        遗传算法主循环：
          - 迭代 epoch_count 代
          - 每代执行选择、交叉、变异生成新种群
          - 最后返回最优个体
        :return: 最佳个体（基因列表）
        """
        for _ in range(self.epoch_count):
            new_pop = []
            # 每次产生两子代
            for _ in range(self.population_size // 2):
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                c1 = self.crossover(p1, p2)
                c2 = self.crossover(p1, p2)
                new_pop.extend([self.mutate(c1), self.mutate(c2)])
            self.population = new_pop

        # 在最终种群中选出最优个体
        best = min(self.population, key=self.fitness)
        return list(best)
