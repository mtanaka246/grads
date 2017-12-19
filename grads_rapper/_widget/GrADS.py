# GrADS.py
# -*- coding: utf-8 -*-

import os
from grads.ganum import GaNum
from Vector2D import Vector2D

class GrADS1:
    def __init__(self, ts_ctl, type, pitch, ret_shape):
        """

        :param ts_ctl: ファイルへの
        :param type: 検索するデータのタイプ - t : 海水温度
        :param pitch: サンプリング間隔
        :param ret_shape: 戻り値のshape
        """
        def calc_dist(n, pitch):
            if (n % 2) == 0:
                return pitch * (n // 2 - 0.5)
            else:
                return pitch * (n // 2)

        self._ts_ctl = ts_ctl
        self._fn = 're({0}, {1})'.format(
            type,  # 取得するサンプルデータのタイプ - t : 海水温度
            str(pitch)  # 指定領域内で、この間隔でデータをサンプリングする
        )
        self._ret_shape = ret_shape
        self._pitch = pitch
        # 領域中心から境界までの距離（°）
        self._half_dist = Vector2D(calc_dist(ret_shape.y, pitch), calc_dist(ret_shape.x, pitch))

    def __enter__(self):
        self._current_dir = os.getcwd()
        # カレントディレクトリを変更（GrADSの仕様）
        os.chdir(os.path.dirname(self._ts_ctl))
        # GrADSの起動
        self._ga = GaNum(Bin='grads -b -q')
        self._ga("open " + self._ts_ctl)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # GrADSの停止
        self._ga("close 1")
        # カレントディレクトリを戻す
        os.chdir(self._current_dir)

    def read(self, pos_date):
        def _toRange(pos, offset, offset_degree, pitch):
            return str(pos - (offset + offset_degree + pitch)) + " " + str(pos + (offset + offset_degree))

        def _toFormattedDate(date):
            return str(date.day) + date.strftime('%b').lower() + str(date.year)

        # GrADSの戻り値で領域の終端側境界にゴミ(---)が含まれることがあるので、検索領域全体を拡大して、その分を最終結果から削除する
        n = 5 # 拡大する要素数
        offset_degree = self._pitch * n  # 要素数から拡大領域を計算 - 1°= 0.1° * 5(要素数)

        self._ga("set lat {0}".format(_toRange(pos_date.pos.y, self._half_dist.y, offset_degree, self._pitch)))
        self._ga("set lon {0}".format(_toRange(pos_date.pos.x, self._half_dist.x, offset_degree, self._pitch)))
        self._ga("set lev {0}".format(pos_date.pos.z))
        self._ga("set time {0}".format(_toFormattedDate(pos_date.date)))

        # 参照 : http://www.mm.media.kyoto-u.ac.jp/wiki/PaperLovers/index.php?%B5%A4%BE%DD%2F%B5%A4%BE%DD%A1%A6%BF%E5%BB%BA%A5%C7%A1%BC%A5%BF%CA%AC%C0%CF%BB%F1%CE%C1#header
        self._ga('define x=' + self._fn)
        ret = self._ga.exp('x')

        # 先頭の行および列は値が実際より小さい値が出るようであるので、検索条件でその分の領域を拡大してここで破棄
        ret = ret[1 + n:ret.shape[0] - n, 1 + n:ret.shape[1] - n]

        assert ret.shape == (self._ret_shape.x, self._ret_shape.y), "戻り値のサイズが指定されたサイズと一致しない　：　{0} != {1}".format(ret.shape, (self._ret_shape.x, self._ret_shape.y))

        return ret

class GrADS2:
    def __init__(self, ts_ctl, type, pitch):
        self._ts_ctl = ts_ctl
        self._fn = 're({0}, {1})'.format(
            type,  # 取得するサンプルデータのタイプ - t : 海水温度
            str(pitch)  # 指定領域内で、この間隔でデータをサンプリングする
        )
        self._pitch = pitch

    def __enter__(self):
        self._current_dir = os.getcwd()
        # カレントディレクトリを変更（GrADSの仕様）
        os.chdir(os.path.dirname(self._ts_ctl))
        # GrADSの起動
        self._ga = GaNum(Bin='grads -b -q')
        self._ga("open " + self._ts_ctl)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # GrADSの停止
        self._ga("close 1")
        # カレントディレクトリを戻す
        os.chdir(self._current_dir)

    def read(self, date, depth, lon_min, lat_min, lon_max, lat_max):
        def _toRange(pos, offset, offset_degree):
            return str(pos - (offset + offset_degree)) + " " + str(pos + (offset + offset_degree))

        def _toFormattedDate(date):
            return str(date.day) + date.strftime('%b').lower() + str(date.year)

        # GrADSの戻り値で領域の終端側境界にゴミが含まれることがあるので、検索領域全体を拡大して、その分を最終結果から削除する
        n = 5 # 拡大する要素数
        offset_degree = self._pitch * n  # 要素数から拡大領域を計算 - 1°= 0.1° * 5(要素数)

        self._ga("set lat {0} {1}".format(lat_min - offset_degree - self._pitch, lat_max + offset_degree))
        self._ga("set lon {0} {1}".format(lon_min - offset_degree - self._pitch, lon_max + offset_degree))
        self._ga("set lev {0}".format(depth))
        self._ga("set time {0}".format(_toFormattedDate(date)))

        # 参照 : http://www.mm.media.kyoto-u.ac.jp/wiki/PaperLovers/index.php?%B5%A4%BE%DD%2F%B5%A4%BE%DD%A1%A6%BF%E5%BB%BA%A5%C7%A1%BC%A5%BF%CA%AC%C0%CF%BB%F1%CE%C1#header
        self._ga('define x=' + self._fn)
        ret = self._ga.exp('x')

        # 先頭の行および列は値が実際より小さい値が出るようであるので、検索条件でその分の領域を拡大してここで破棄
        ret = ret[1 + n:ret.shape[0] - n, 1 + n:ret.shape[1] - n]

        return ret

