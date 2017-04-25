# coding=utf-8
import random
import os
from common.util import download_file
import mxnet as mx
import numpy as np
import gzip
import struct

def read_data(label, image):
    """
    download and read data into numpy
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    with gzip.open(download_file(base_url + label, os.path.join('data', label))) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_file(base_url + image, os.path.join('data', image)), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(
            len(label), rows, cols)
    return (label, image)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

class PairDataIter(mx.io.DataIter):
    def __init__(self, batch_size, mode='train'):
        super(PairDataIter, self).__init__()
        assert mode in ['train', 'val']
        self.batch_size=batch_size
        self.provide_label=[('sim_label', (batch_size, 1)), ]
        self.provide_data=[('pos_data', (batch_size, 28, 28)), ('neg_data', (batch_size, 28, 28))]
        if mode == 'train':
            (self.labels, self.data) = read_data(
                'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
        else:
            (self.labels, self.data) = read_data(
                't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
        digit_indices = [np.where(self.labels == i)[0] for i in range(10)]
        self.pairs, self.sim_labels = create_pairs(self.data, digit_indices)
        self.end_idx = len(self.sim_labels) // self.batch_size
        self.count = 0

    def reset(self):
        self.count = 0
        indexes = range(len(self.pairs))
        random.shuffle(indexes)
        self.pairs = self.pairs[indexes]
        self.sim_labels = self.sim_labels[indexes]
        
    def next(self):
        if self.count == self.end_idx:
            raise StopIteration
        pair_data = self.pairs[self.count*self.batch_size:(self.count+1)*self.batch_size]
        sim_labels = self.sim_labels[self.count*self.batch_size:(self.count+1)*self.batch_size]
        self.count += 1
        return mx.io.DataBatch(
            data=[mx.nd.array(pair_data[:,0]), mx.nd.array(pair_data[:,1])],
            label=[mx.nd.array(sim_labels.reshape((self.batch_size, 1)))],
            provide_data=self.provide_data,
            provide_label=self.provide_label
        )


if __name__=='__main__':
    pair_iter = PairDataIter(batch_size=10)
    batch = pair_iter.next()
    print batch.data[0], batch.label[0]

