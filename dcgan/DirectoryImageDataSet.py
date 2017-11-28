# -*- coding: utf-8 -*-
# DirectoryImageDataSet.py

from BaseDataSet import BaseDataSet
import numpy as np

class DirectoryImageDataSet(BaseDataSet):
    """
        学習実行時に指定されたディレクトリからバッチサイズ分のファイルをロードし
        正規化して、GANを実行するためのクラス
    """
    def __init__(self, dir, data_size_per_batch, data_shape):
        """
        コンストラクタ
        :param dir: ディレクトリ
        :param data_size_per_batch: 1バッチのデータ数（ファイル数）
        :param data_shape: データのshape（幅, 高さ, 深さ）
        """
        self.dir = dir
        self.data_size_per_batch = data_size_per_batch
        self.files = _dataset_files(self.dir, ["png"])
        self.data_shape = (len(self.files),) + data_shape

    def batch(self, batch_id):
        """
        batch_id で指定されたバッチデータをリストで返す（呼び出し側は正規化済みのデータが帰ることを前提としている）
        :param batch_id: バッチインデックス 
        :return: batch_id 番目のバッチデータ
        """
        first = batch_id * self.data_size_per_batch
        last = first + self.data_size_per_batch

        return self.samples(range(first, last), True)

    def size(self):
        """
        全データ数を返す
        :return: 全データ数
        """
        return self.data_shape[0]

    def shape(self):
        """
        データのshappe
        :return: (全レコード数, 1データの幅, 高さ, 深さ)のタプル
        """
        return self.data_shape

    def samples(self, ids, normalized):
        """
        ids で指定されたインデックスのデータを返す
        :param ids: データのインデックス
        :return: データ
        """
        images = np.array([_imread(file, self.data_shape[-1]) for file in self.files[ids]])

        return self.normalize(images) if normalized else images

    def normalize(self, images):
        return images / 127.5 - 1.0

    def denormalize(self, images):
        return (images + 1.0) * 127.5

def _dataset_files(root, ext):
    import itertools
    import os
    from glob import glob

    """Returns a list of all image files in the given directory"""
    return np.array(sorted(itertools.chain.from_iterable(glob(os.path.join(root, "*.{}".format(ext))) for ext in ext)))

def _imread(path, c_dim):
    import scipy

    if c_dim == 3:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
    elif c_dim == 1:
        v = scipy.misc.imread(path, mode='L').astype(np.float)
        return np.reshape(v, v.shape + (1,))
    else:
        raise  ValueError("c_dim must be one or three")
