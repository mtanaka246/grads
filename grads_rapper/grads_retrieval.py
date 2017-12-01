# grads_retrieval.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime as dt
from _widget.GrADS import GrADS
from _widget.Vector2D import Vector2D
from _widget.Vector3D import Vector3D
from _widget.PosDate import PosDate

def build_temp_64_64_with_file(src, grads_ts_ctl, dst):
    """
    src で指定された日付・位置・水深で、その位置を中心とした6.4°四方の領域内で0.1°間隔の海水温情報を取得する
    :param src: date(年月日)、lat, lon(緯度経度（2次元座標）、depth（水深）の列ラベルを含むCSV(DataFrame、index_colなし)ファイル
    :param grads_ts_ctl: GrADSのファイル
    :param dst: 学習データの出力先
    :return: なし
    """
    # 日付＆緯度経度＆水深を記録したファイルを読み込む
    df_date_lon_lat_depth = pd.read_csv(src)

    # 読み込んだ日付と緯度経度を元にGrADSから6.4°四方の領域における0.1°間隔の海水温情報を取得
    df_temp_64_64 = create_temp_64_64_with_dataframe(df_date_lon_lat_depth, grads_ts_ctl)

    # 取得した海水温情報をファイルに保存
    df_temp_64_64.to_csv(dst, index=False)

def create_temp_64_64_with_dataframe(df_date_lon_lat_depth, grads_ts_ctl):
    """
    df_date_lon_lat_depth で指定された日付・位置・水深で、その位置を中心とした6.4°四方の領域内で0.1°間隔の海水温情報を取得する
    :param src: date(年月日)、lat, lon(緯度経度（2次元座標）、depth（水深）の列ラベルを含むDataFrame
    :param grads_ts_ctl: GrADSのファイル
    :return: 海水温情報のDataFrame
    """
    def _to_pos(e):
        return Vector3D(e.lon, e.lat, e.depth)

    def _to_date(e):
        return dt.strptime(e.date, '%Y/%m/%d')

    def _to_pos_date(e):
        return PosDate(_to_pos(e), _to_date(e))

    def _to_pos_date_list(df_date_lon_lat_depth):
        return [_to_pos_date(e) for _, e in df_date_lon_lat_depth.iterrows()]

    def _read_temperature(pos_date_list, grads_ts_ctl):
        with GrADS(grads_ts_ctl, 't', 0.1, Vector2D(64, 64)) as ga:
            return np.array([ga.read(pos_date) for pos_date in pos_date_list])

    # 内部のフォーマットに変換
    pos_date_list = _to_pos_date_list(df_date_lon_lat_depth)

    # 読み込んだ日付と緯度経度を元にGrADSから6.4°四方の領域における0.1°間隔の海水温情報を取得
    temp_64_64_list = _read_temperature(pos_date_list, grads_ts_ctl)

    return pd.DataFrame(temp_64_64_list.reshape(temp_64_64_list.shape[0], temp_64_64_list.shape[1] * temp_64_64_list.shape[2]))

