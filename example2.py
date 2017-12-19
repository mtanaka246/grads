# example2.py
# -*- coding: utf-8 -*-

from grads_rapper.grads_retrieval import create_temp_64_64_with_dataframe
import pandas as pd

def exe():
    # 例② create_temp_64_64_with_dataframe()
    # build_temp_64_64_with_file()をカスタマイズする

    # １．入力データの作成
    df_date_lon_lat_depth = pd.DataFrame(
        [
            ["2012/7/15", 45, 180, 100],
            ["2012/7/16", 45.1, 181, 100],
            ["2012/7/17", 45.2, 182, 100],
            ["2012/7/18", 45.3, 183, 100],
            ["2012/7/19", 45.4, 184, 100]
        ],
        columns=["date", "lat", "lon", "depth"])

    # ２．海水温情報の取得
    df_temp_64_64 = create_temp_64_64_with_dataframe(df_date_lon_lat_depth, '/mnt/seadata/ts.ctl')

    # ３．ファイルに出力
    # df_temp_64_64.to_csv("output.csv", index=False)
    print df_temp_64_64


