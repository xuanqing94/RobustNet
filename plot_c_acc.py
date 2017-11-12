#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def readf(fn):
    c = []
    acc = []
    for l in open(fn, 'r'):
        if l[0] == '#':
            continue
        items = l.split(',')
        c.append(float(items[0]))
        acc.append(float(items[1]))
    return np.array(c), np.array(acc)


if __name__ == "__main__":
    c1, acc_none = readf('./experiment/cifar10_CW_nodefense')
    c2, acc_our = readf('./experiment/cifar10_CW_ours_0.6_0.1')
    c3, acc_our_ensemble = readf('./experiment/cifar10_CW_ours_ensemble50_tmp')
    plt.semilogx(c1, 100 * acc_none, 'r', label='No defense')
    plt.semilogx(c2, 100 * acc_our, 'b', label='Ours, ensemble=1')
    plt.semilogx(c3, 100 * acc_our_ensemble, 'g', label='Ours, ensemble=50')

    plt.xlabel('c')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.show()
