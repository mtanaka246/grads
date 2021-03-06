# grads_retrieval.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from datetime import datetime as dt
from _widget.GrADS import GrADS1, GrADS2
from _widget.Vector2D import Vector2D
from _widget.Vector3D import Vector3D
from _widget.PosDate import PosDate

def build_temp_64_64_with_file(src, grads_ts_ctl, dst):
    """
    src で指定された日付・位置・水深で、その位置を中心とした6.4°四方の領域内で0.1°間隔の海水温情報を取得する
    
    （注）出力データの各レコードは緯度ごとに走査した結果であるが、低緯度から走査しているので、
    　　　上位レコードが低緯度で下位レコードが高緯度になっているので
    
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
    
    （注）出力データの各レコードは緯度ごとに走査した結果であるが、低緯度から走査しているので、
    　　　上位レコードが低緯度で下位レコードが高緯度になっているので
    
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
        with GrADS1(grads_ts_ctl, 't', 0.1, Vector2D(64, 64)) as ga:
            return np.array([ga.read(pos_date) for pos_date in pos_date_list])

    # 内部のフォーマットに変換
    pos_date_list = _to_pos_date_list(df_date_lon_lat_depth)

    # 読み込んだ日付と緯度経度を元にGrADSから6.4°四方の領域における0.1°間隔の海水温情報を取得
    temp_64_64_list = _read_temperature(pos_date_list, grads_ts_ctl)

    return pd.DataFrame(temp_64_64_list.reshape(temp_64_64_list.shape[0], temp_64_64_list.shape[1] * temp_64_64_list.shape[2]))

def create_temp_map(date, depth, lon_min, lat_min, lon_max, lat_max, pitch, grads_ts_ctl):
    """
    指定された日付、水深、領域に対して、pitchの間隔ごとの海水温を返す。
    
    （注１）出力データの各レコードは緯度ごとに走査した結果であるが、低緯度から走査しているので、
    　　　上位レコードが低緯度で下位レコードが高緯度になっているので
    （注２）結果の行数および列数はGrADSの仕様に従うため、事前に予測することは難しい
    　　　結果の次元を確定したい場合はcreate_temp_map_ex()を使用する
    　　　①実験結果からGrADSの戻り値の次元は_calc_size()で定義された式の戻り値+1
    　　　②戻り値の先頭行および先頭列の値は他のデータに比べ低い値が出力されるため除去

    :param date: '年/月/日'の書式で指定された文字列
    :param depth: 水深
    :param lon_min: 領域の経度の下限値
    :param lat_min: 領域の緯度の下限値
    :param lon_max: 領域の経度の上限値
    :param lat_max: 領域の緯度の上限値
    :param pitch: 間隔
    :param grads_ts_ctl: GrADSのファイル
    :return: 海水温情報のDataFrame
    """
    def _to_date(date):
        return dt.strptime(date, '%Y/%m/%d')

    with GrADS2(grads_ts_ctl, 't', pitch) as ga:
        return pd.DataFrame(ga.read(_to_date(date), depth, lon_min, lat_min, lon_max, lat_max))

def create_temp_map_ex(date, depth, lon, lat, pitch, row_col, grads_ts_ctl):
    """
    指定された位置（中心座標）、pitch、row_colから領域を算出し、その領域での指定日、指定水深の海水温を返す。
    
    :param date: '年/月/日'の書式で指定された文字列
    :param depth: 水深
    :param lon: 領域の中心位置の経度
    :param lat: 領域の中心位置の緯度
    :param pitch: 間隔
    :param row_col: 出力結果の次元（戻り値のshape（行、列）をここで指定する）
    :param grads_ts_ctl: GrADSのファイル
    :return: 海水温情報のDataFrame
    """
    def _to_pos(depth, lon, lat):
        return Vector3D(lon, lat, depth)

    def _to_date(date):
        return dt.strptime(date, '%Y/%m/%d')

    def _to_pos_date(date, depth, lon, lat):
        return PosDate(_to_pos(depth, lon, lat), _to_date(date))

    def _read_temperature(pos_date, grads_ts_ctl, pitch, row_col):
        with GrADS1(grads_ts_ctl, 't', pitch, row_col) as ga:
            return ga.read(pos_date)

    # 内部のフォーマットに変換
    pos_date = _to_pos_date(date, depth, lon, lat)
    temp_map = _read_temperature(pos_date, grads_ts_ctl, pitch, Vector2D(row_col[0], row_col[1]))

    return pd.DataFrame(temp_map)
