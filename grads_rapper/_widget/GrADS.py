# GrADS.py
# -*- coding: utf-8 -*-

import os
from grads.ganum import GaNum
from Vector2D import Vector2D

class GrADS:
    def __init__(self, ts_ctl, type, pitch, ret_shape):
        """

        :param ts_ctl: ファイルへの
        :param type: 検索するデータのタイプ - t : 海水温度
        :param pitch: サンプリング間隔
        :param ret_shape: 戻り値のshape
        """
        self._ts_ctl = ts_ctl
        self._fn = 're({0}, {1})'.format(
            type,  # 取得するサンプルデータのタイプ - t : 海水温度
            str(pitch)  # 指定領域内で、この間隔でデータをサンプリングする
        )
        self._ret_shape = ret_shape
        self._pitch = pitch
        self._offset = Vector2D(ret_shape.x * pitch / 2, ret_shape.y * pitch / 2)

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
        def _toRange(dat, offset, offset_degree):
            return str(dat - offset) + " " + str(dat + (offset + offset_degree))

        def _toFormattedDate(date):
            return str(date.day) + date.strftime('%b').lower() + str(date.year)

        # GrADSの戻り値で領域の終端側境界にゴミが含まれることがあるので、領域を大きくして検索し、その分を最終結果から削除する
        offset_degree = 1.0  # 上記の拡大領域分

        # print("set lat {0}".format(_toRange(pos_date.pos.y, self._offset.y, offset_degree)))
        # print("set lon {0}".format(_toRange(pos_date.pos.x, self._offset.x, offset_degree)))
        # print("set lev {0}".format(pos_date.pos.z))
        # print("set time {0}".format(_toFormattedDate(pos_date.date)))
        # print('define x=' + self._fn)

        self._ga("set lat {0}".format(_toRange(pos_date.pos.y, self._offset.y, offset_degree)))
        self._ga("set lon {0}".format(_toRange(pos_date.pos.x, self._offset.x, offset_degree)))
        self._ga("set lev {0}".format(pos_date.pos.z))
        self._ga("set time {0}".format(_toFormattedDate(pos_date.date)))

        # 参照 : http://www.mm.media.kyoto-u.ac.jp/wiki/PaperLovers/index.php?%B5%A4%BE%DD%2F%B5%A4%BE%DD%A1%A6%BF%E5%BB%BA%A5%C7%A1%BC%A5%BF%CA%AC%C0%CF%BB%F1%CE%C1#header
        self._ga('define x=' + self._fn)
        ret = self._ga.exp('x')

        n = int(offset_degree / self._pitch if (self._pitch < offset_degree) else self._pitch)
        ret = ret[1:ret.shape[0] - n, 1:ret.shape[1] - n]

        assert ret.shape == (self._ret_shape.x, self._ret_shape.y)

        return ret

