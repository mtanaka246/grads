# -*- coding: utf-8 -*-
# BaseDataSet.py

class BaseDataSet(object):
    def batch(self):
        """
        バッチデータをリストで返すジェネレータ
        （呼び出し側は正規化済みのデータが帰ることを前提としている）
        :return: バッチデータ
        """
        raise NotImplementedError()

    def batch_size(self):
        """
        1エポックのバッチ数を返す
        :return: 1エポックのバッチ数
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

    def size_per_batch(self):
        """
        1バッチのデータ数を返す
        :return: 1バッチのデータ数
        """
        raise NotImplementedError()
