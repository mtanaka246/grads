# -*- coding: utf-8 -*-
# FeatureModel.py

import numpy as np

class FeatureModel:
    """
    実画像を生成するためのGenerator引数を推定し、その推定値で画像を生成するクラス
    """
    def __init__(self, generator, seed=None, fitting_epoch=100):
        """
        コンストラクタ
        :param generator: GANで作成したGenrator 
        :param seed: 初期化時の乱数シード
        :param fitting_epoch: 推定エポック数
        """
        def _set_trainable(model, val):
            u = model.trainable
            model.trainable = val

            v = []

            for layer in model.layers:
                v.append(layer.trainable)
                layer.trainable = val

            return u, v

        def _recover_trainable(model, u, v):
            model.trainable = u

            for layer, val in zip(model.layers, v):
                layer.trainable = val

        def _create_model(generator):
            input_shape = (1, generator.input_shape[1], 1)

            model = Sequential()
            model.add(LocallyConnected1D(1, 1, input_shape=input_shape[1:], use_bias=False))
            model.add(Flatten())
            model.add(generator)
            # コンパイル前にgeneratorを学習対象外にする
            u, v = _set_trainable(model.layers[2], False)
            model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(lr=0.1))
            # Generator の状態を戻す
            _recover_trainable(model.layers[2], u, v)

            return model

        import keras
        from keras.models import Sequential
        from keras.layers import Flatten
        from keras.layers.local import LocallyConnected1D

        input_shape = (1, generator.input_shape[1], 1)

        self.model = _create_model(generator)

        if seed != None:
            np.random.seed(seed)  # 初期値を固定

        self.lw = np.random.uniform(-1.0, 1.0, input_shape[1]).reshape(np.shape(self.model.layers[0].get_weights()))
        self.x_all_one = np.ones(input_shape)
        self.fitting_epoch = fitting_epoch

    def _train(self, model, lw, x, y, epoch):
        model.layers[0].set_weights(lw)
        # u, v = _set_trainable(model.layers[2], False) # 念のため
        model.fit(x, y.reshape((1,) + np.shape(y)), epochs=epoch, verbose=0)
        # _update_trainable(model.layers[2], u, v)

        return np.array(model.layers[0].get_weights())

    def estimate(self, y):
        """
        引数で与えられた画像が Generator の出力となるように入力を推定する
        :param y: 画像（縦×横×深さ）
        :return: 画像に対する推定入力値
        """
        x = self._train(self.model, self.lw, self.x_all_one, y, self.fitting_epoch)

        return np.array(x).flatten()

    def replicate(self, y):
        """
        引数で与えられた画像が Generator の出力となるように入力を推定し、その値で画像を作成する
        :param y: 画像（縦×横×深さ）
        :return: 画像に対する推定入力値で生成したGeneratorの出力
        """
        x = self._train(self.model, self.lw, self.x_all_one, y, self.fitting_epoch)
        x = np.reshape(x, (1, self.model.layers[2].input_shape[1]))
        y = self.model.layers[2].predict(x)

        return np.reshape(y, self.model.output_shape[1:])
