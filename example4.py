# example4.py
# -*- coding: utf-8 -*-

from grads_rapper.grads_retrieval import create_temp_map_ex

def exe():
    """
    東経147.0°～198.0°、北緯35.0°～47.0°、水深100ｍの領域の海水温データを取得するプログラム
    :return: 無し 
    """
    pitch = 0.1
    lon = (147.0 + 198.0) / 2.0
    lat = (35.0 + 47.0) / 2.0
    row_col = ((47.0 - 35.0) / pitch, (198.0 - 147.0) / pitch) # (120, 510)
    # row_col = (121, 511)
    df = create_temp_map_ex('1999/6/1', 100, lon, lat, pitch, row_col, '/mnt/seadata/ts.ctl')

    print df
