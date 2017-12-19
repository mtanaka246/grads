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

    # # "sea_temp_200_[35.0,147.0]_[47.0,198.0]_2012_07_31.dat"
    # from datetime import datetime as dt
    # import datetime
    # for depth in [200]:
    #     for year in [2010, 2011, 2012]:
    #         begin = dt.strptime("{0}/06/01".format(year), '%Y/%m/%d')
    #         date_list = [day for day in (begin + datetime.timedelta(x) for x in range(30+31))]
    #
    #         for date in date_list:
    #             file = "./output/sea_temp_{0}_[35.0,147.0]_[47.0,198.0]_{1}_{2:02}_{3:02}.dat".format(depth, date.year, date.month, date.day)
    #             # df = create_temp_map('{0}/{1}/{2}'.format(date.year, date.month, date.day), depth, 147.0, 35.0, 198.0, 47.0, 0.1, '/mnt/seadata/ts.ctl')
    #             df = create_temp_map('{0}/{1}/{2}'.format(date.year, date.month, date.day), depth, 147.0, 35.0, 197.7, 47.0, 0.1, '/mnt/seadata/ts.ctl')
    #             df.to_pickle(file)
