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
from lottokit import Daletou

if __name__ == '__main__':
    d = Daletou()
    # d.download_data()
    # d.fetch_data()
    d.predict(next_period=2025044, next_weekday=3, predict_type='analyze')