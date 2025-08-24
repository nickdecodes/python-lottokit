#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : nickdecodes
@Email   : nickdecodes@163.com
@Usage   :
@FileName: test.py
@DateTime: 2024/1/28 20:13
@SoftWare: 
"""

import sys
sys.path.append('..')
from lottokit import Daletou, ModelUtil

if __name__ == '__main__':
    d = Daletou()
    # d.download_data()
    # d.fetch_data()
    # d.predict(next_period=2025096, next_weekday=3, predict_type='llm', window_size=15)
    d.predict(next_period=2025096, next_weekday=3, predict_type='analyze', window_size=15)
    # d.predict(next_period=2025049, next_weekday=1, predict_type='model', window_size=15)
    # d.predict(next_period=None, next_weekday=None, predict_type='llm', window_size=15)