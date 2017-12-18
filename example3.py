# example3.py
# -*- coding: utf-8 -*-

from grads_rapper.grads_retrieval import create_temp_map

def exe():
    """
    東経147.0°～198.0°、北緯35.0°～47.0°、水深100ｍの領域の海水温データを取得するプログラム
    :return: 無し 
    """
    df = create_temp_map('1999/6/1', 100, 147.0, 35.0, 198.0, 47.0, 0.1, '/mnt/seadata/ts.ctl')

    print df
