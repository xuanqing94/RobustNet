#!/usr/bin/env python3

import pickle
import numpy as np
from scipy.misc import imshow

def aug(fin, fout, multiple, std):
    with open(fin, 'rb') as fh:
        entry = pickle.load(fh, encoding='latin1')
        data = entry['data']
        label = entry['labels']
        batch, n = data.shape
        noise = np.random.randn(multiple * batch, n) * std
        data_aug = np.clip(np.repeat(data, multiple, axis=0) + noise, 0, 255).astype(np.uint8)
        data = np.concatenate((data, data_aug))
        label = label * (multiple + 1)
        entry['data'] = data
        entry['labels'] = label
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
        aug(fin, fout, 9, 255 * 0.05)

