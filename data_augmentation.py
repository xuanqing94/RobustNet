#!/usr/bin/env python3

import pickle
import numpy as np

def aug(fn):
    with open(fn, 'rb') as fh:
        entry = pickle.load(fh, encoding='latin1')
    print(entry.shape)


if __name__ == "__main__":
    folder = '/home/luinx/'
