#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : nickdecodes
@Email   : nickdecodes@163.com
@Usage   :
@FileName: spider_util.py
@DateTime: 2024/7/22 10:03
@SoftWare:
"""

import re
import time
import datetime
from contextlib import contextmanager
from typing import Any, List, Generator

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class SpiderUtil:
    @staticmethod
    @contextmanager
    def web_driver(url, headless=True) -> Generator[WebDriver, Any, None]:
        url = url
        opts = Options()
        if headless:
            opts.add_argument("--headless")
        drv = webdriver.Chrome(options=opts)
        drv.get(url)
        time.sleep(2)  # 可根据页面加载情况适当调整
        try:
            yield drv
        finally:
            drv.quit()

    @staticmethod
    def dlt_sina_recent_data() -> List[List[str]]:
        all_rows: List[List[str]] = []
        url = f'https://view.lottery.sina.com.cn/lotto/pc_zst/index?lottoType=dlt&actionType=chzs&type=50&dpc=1'
        with SpiderUtil.web_driver(url) as drv:
            time.sleep(3)
            frame = drv.find_element(By.XPATH, '//*[@id="cpdata"]')
            rows = frame.find_elements(By.TAG_NAME, "tr")
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 3:
                    continue
                period = cells[0].text.strip()
                weekday = cells[2].text.strip()
                front = []
                back = []
                for cell in cells[3:-4]:
                    class_name = cell.get_attribute("class")
                    if class_name == "br02":
                        continue
                    if class_name in ["chartball01", "chartball20"]:
                        front.append(cell.text.strip())
                    elif class_name in ["chartball02"]:
                        back.append(cell.text.strip())
                    # elif class_name in ["yl02", "yl01", "bg_p"]:
                    #     if len(front) >= 35:
                    #         back.append(f'_{cell.text.strip()}_')
                    #     else:
                    #         front.append(f'_{cell.text.strip()}_')
                sum_total = cells[-4].text.strip()
                span = cells[-3].text.strip()
                zone_ratio = cells[-2].text.strip()
                odd_even_ratio = cells[-1].text.strip()
                all_rows.append([period, weekday, *front, *back, sum_total, span, zone_ratio, odd_even_ratio])
        return all_rows

    @staticmethod
    def dlt_sina_full_data() -> List[List[str]]:
        all_rows: List[List[str]] = []
        for year in range(2007, datetime.datetime.today().year + 1):
            print(year)
            url = f'https://view.lottery.sina.com.cn/lotto/pc_zst/index?lottoType=dlt&actionType=chzs&year={year}&dpc=1'
            with SpiderUtil.web_driver(url) as drv:
                time.sleep(3)
                frame = drv.find_element(By.XPATH, '//*[@id="cpdata"]')
                rows = frame.find_elements(By.TAG_NAME, "tr")
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) < 3:
                        continue
                    period = cells[0].text.strip()
                    weekday = cells[2].text.strip()
                    front = []
                    back = []
                    for cell in cells[3:-4]:
                        class_name = cell.get_attribute("class")
                        if class_name == "br02":
                            continue
                        if class_name in ["chartball01", "chartball20"]:
                            front.append(cell.text.strip())
                        elif class_name in ["chartball02"]:
                            back.append(cell.text.strip())
                        # elif class_name in ["yl02", "yl01", "bg_p"]:
                        #     if len(front) >= 35:
                        #         back.append(f'_{cell.text.strip()}_')
                        #     else:
                        #         front.append(f'_{cell.text.strip()}_')
                    # sum_total = cells[-4].text.strip()
                    # span = cells[-3].text.strip()
                    # zone_ratio = cells[-2].text.strip()
                    # odd_even_ratio = cells[-1].text.strip()
                    all_rows.append([
                        period, weekday, *front, *back,
                        # sum_total, span, zone_ratio, odd_even_ratio
                    ])
        return all_rows

    @staticmethod
    def dlt_gov_recent_data() -> List[List[str]]:
        """返回最新（第一页）记录"""
        url = 'https://www.lottery.gov.cn/kj/kjlb.html?dlt'
        with SpiderUtil.web_driver(url) as drv:
            time.sleep(3)
            frame = drv.find_element(By.XPATH, '//iframe[@id="iFrame1"]')
            drv.switch_to.frame(frame)
            content = drv.find_element(By.XPATH, '//tbody[@id="historyData"]')
            recent_data = [x.split(' ')[:9] for x in content.text.split('\n')]
        return recent_data

    @staticmethod
    def dlt_gov_full_data() -> List[List[str]]:
        """
        抓取所有分页数据：
         - 若未传 max_pages 且定义了 page_pattern，则先从页面源码里计算最大页码
         - 否则最多翻 max_pages 页，或直至无法翻页
        """
        url = 'https://www.lottery.gov.cn/kj/kjlb.html?dlt'
        full_data: List[List[str]] = []
        with SpiderUtil.web_driver(url) as drv:
            time.sleep(3)
            frame = drv.find_element(By.XPATH, '//iframe[@id="iFrame1"]')
            drv.switch_to.frame(frame)
            matches = re.findall(r'goNextPage\((\d+)\)', drv.page_source)
            page_index = [int(match) for match in matches]
            for index in range(max(page_index)):
                WebDriverWait(drv, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//tbody[@id="historyData"]'))
                )
                content = drv.find_element(By.XPATH, '//tbody[@id="historyData"]')
                full_data.extend([x.split()[:9] for x in content.text.split('\n') if len(x.split()) >= 9])

                try:
                    next_button = WebDriverWait(drv, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "/html/body/div/div/div[3]/ul/li[position()=13]"))
                    )
                except Exception as ex:
                    next_button = WebDriverWait(drv, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "/html/body/div/div/div[3]/ul/li[position()=8]"))
                    )
                next_button.click()
                time.sleep(3)
        sorted_full_data = sorted(full_data, key=lambda x: int(x[0]))
        return sorted_full_data


# === 用法示例 ===
if __name__ == "__main__":
    # 实例化爬虫（可通过 kwargs 覆盖默认任何配置项）
    spider = SpiderUtil()

    # 1. 抓取并打印“最近一页”的所有数据
    print("最近一页数据：", spider.dlt_sina_recent_data())

    # 2. 若要抓取“全部历史记录”，可使用
    # print("全部历史共", len(spider.dlt_sina_full_data()), "条")
