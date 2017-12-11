# example1.py
# -*- coding: utf-8 -*-

from grads_rapper.grads_retrieval import build_temp_64_64_with_file

def exe():
    # 例① build_temp_64_64_with_file()
    # ファイルで指定された情報から海水温を取得し、ファイルに出力
    build_temp_64_64_with_file(
        "input.csv",  # 'date', 'lat', 'lon', 'depth' で日付と座標が記されたCSVファイル（「index_col無し」を前提）
        '/mnt/seadata/ts.ctl',  # GrADSのファイル
        "output.csv"  # 出力先パス
    )
