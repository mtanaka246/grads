# -*- coding: utf-8 -*-
# BaseDataSet.py

class BaseDataSet(object):
    def batch(self, batch_id):
        """
        batch_id で指定されたバッチデータをリストで返す（呼び出し側は正規化済みのデータが帰ることを前提としている）
        :param batch_id: バッチインデックス 
        :return: batch_id 番目のバッチデータ
        """
        raise NotImplementedError()

    def size(self):
        """
        全データ数を返す
        :return: 全データ数
        """
        raise NotImplementedError()

    def shape(self):
        """
        データのshappe
        :return: (全レコード数, 1データの幅, 高さ, 深さ)のタプル
        """
        raise NotImplementedError()
