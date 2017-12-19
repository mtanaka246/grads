# example4.py
# -*- coding: utf-8 -*-

from grads_rapper.grads_retrieval import create_temp_map_ex

def exe():
    """
    東経147.0°～198.0°、北緯35.0°～47.0°、水深100ｍの領域の海水温データを取得するプログラム
    （中心位置と戻り値のサイズを指定する）
    :return: 無し 
    """
    pitch = 0.1
    lon = (147.0 + 198.0) / 2.0 # 中心位置
    lat = (35.0 + 47.0) / 2.0 # 中心位置
    row_col = (1 + (47.0 - 35.0) / pitch, 1 + (198.0 - 147.0) / pitch) # 戻り値のサイズ (121, 511)
    # row_col = (1, 1)
    # row_col = (3, 3)
    # row_col = (1, 2)
    # row_col = (2, 1)
    # row_col = (0, 0)
    df = create_temp_map_ex('1999/6/1', 100, lon, lat, pitch, row_col, '/mnt/seadata/ts.ctl')

    print df
