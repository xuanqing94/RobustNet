#!/usr/bin/env python3

import pickle
import numpy as np
from scipy.misc import imshow

def aug(fin, fout, multiple, std):
    with open(fin, 'rb') as fh:
        entry = pickle.load(fh, encoding='latin1')
        data = entry['data']
        label = entry['labels']
        data = data.reshape(-1, 3, 32, 32)
        batch, channel, h, w = data.shape
        noise = np.random.randn(multiple * batch, channel, h, w) * std
        data_aug = np.repeat(data, multiple, axis=0) + noise
        img = data_aug[0, :, :, :]
        data = np.concatenate((data, data_aug))
        label = np.repeat(label, multiple + 1, axis=0)
        entry['data'] = data
        entry['label'] = label
        with open(fout, 'wb') as fh_out:
            pickle.dump(entry, fh_out)
        print(fin)


if __name__ == "__main__":
    folder = '/home/luinx/data/cifar10-py/cifar-10-batches-py/'
    folder_dst = '/home/luinx/data/cifar10-py/cifar-10-batches-py-aug/'
    file_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    for f in file_list:
        fin = folder + f
        fout = folder_dst + f
        aug(fin, fout, 10, 255 * 0.03)

