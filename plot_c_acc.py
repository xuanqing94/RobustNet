#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def readf(fn):
    c = []
    acc = []
    for l in open(fn, 'r'):
        if l[0] in ('#', 'F'):
            continue
        items = l.split(',')
        c.append(float(items[0]))
        acc.append(float(items[1]))
    return np.array(c), np.array(acc)


if __name__ == "__main__":
    method="FGSM"
    dataset="cifar10"
    algo = ["ours_0_0_1", "ours_0.9_0.2_50"]
    color = ["r", "g"]
    for c, a in zip(color, algo):
        x, y = readf('./experiment/{}_{}_{}'.format(dataset, method, a))
        plt.semilogx(x, 100 * y, color=c, label=a)

    plt.xlabel('c')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.show()
