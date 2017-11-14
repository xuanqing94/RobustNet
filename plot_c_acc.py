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
    method="CW"
    dataset="cifar10"
    model = "vgg16"
    algo = ["0.9_0.2_1", "0.9_0.2_50"]
    labels = ["1-ensemble", "50-ensemble"]
    for lab, a in zip(labels, algo):
        x, y = readf('./experiment/{}_{}_ours_{}_{}'.format(dataset, method, model, a))
        plt.semilogx(x, 100 * y, label=lab)

    plt.xlabel('c')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.show()
