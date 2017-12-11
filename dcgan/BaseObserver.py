# -*- coding: utf-8 -*-
# BaseObserver.py

class BaseObserver(object):
    def on_completed_batch_train(self, proxy, epoch_id, batch_id, counter, g_loss, d_loss, elapsed_time):
        """
        バッチ単位の学習完了後にコールされる
        :param proxy: プロキシー
        :param epoch_id: 現在のエポック番号
        :param batch_id: 現在のパッチ番号
        :param counter: 実行したバッチ処理の回数
        :param g_loss: Generator のロス
        :param d_loss: Discriminator のロス
        :param elapsed_time: 経過時間
        :return: True : 学習を継続, False : 学習を中止
        """
        raise NotImplementedError()

    def on_completed_epoch_train(self, proxy, epoch_id, batch_id, counter, elapsed_time):
        """
        エポック単位の学習完了後にコールされる
        :param proxy: プロキシー
        :param epoch_id: 現在のエポック番号
        :param batch_id: 現在のパッチ番号
        :param counter: 実行したバッチ処理の回数
        :param elapsed_time: 経過時間
        :return: True : 学習を継続, False : 学習を中止
        """
        raise NotImplementedError()
