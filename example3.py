# example3.py
# -*- coding: utf-8 -*-

from grads_rapper.grads_retrieval import create_temp_map

def exe():
    """
    東経147.0°～198.0°、北緯35.0°～47.0°、水深100ｍの領域の海水温データを取得するプログラム
    :return: 無し 
    """
    df = create_temp_map('1999/6/1', 100, 147.0, 35.0, 198.0, 47.0, 0.1, '/mnt/seadata/ts.ctl') # [121 rows x 511 columns]
    # df = create_temp_map('1999/6/1', 100, 147.0, 35.0, 197.7, 47.0, 0.1, '/mnt/seadata/ts.ctl') # [121 rows x 508 columns]
    # df = create_temp_map('1999/6/1', 100, 147.0, 35.0, 147.0, 35.0, 0.1, '/mnt/seadata/ts.ctl') # [1 rows x 1 columns]
    # df = create_temp_map('1999/6/1', 100, 147.0, 35.0, 198, 35.0, 0.3, '/mnt/seadata/ts.ctl') # [1 rows x 171 columns]
    # df = create_temp_map('1999/6/1', 100, 147.0, 35.0, 198.1, 35.0, 0.3, '/mnt/seadata/ts.ctl')  # [1 rows x 172 columns]
    # df = create_temp_map('1999/6/1', 100, 147.0, 35.0, 147.0, 47.0, 0.3, '/mnt/seadata/ts.ctl') # [41 rows x 1 columns]

    print df
