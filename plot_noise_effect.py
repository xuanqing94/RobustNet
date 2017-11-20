#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def readf(fn):
    c = []
    acc = []
    distort = []
    for l in open(fn, 'r'):
        if l[0] in ('#', 'F'):
            continue
        items = l.split(',')
        c.append(float(items[0]))
        acc.append(float(items[1]))
        distort.append(float(items[2]))
    return np.array(c), np.array(acc), np.array(distort)

if __name__ == "__main__":
    dataset = "cifar10"
    attack = "CW"
    net = "vgg16"
    methods = ["0_0_1", "0_0_1_tmp", "0.9_0.2_1_tmp", "0.9_0.2_1"]
    labels = ["Model 1", "Model 2", "Model 3", "Model 4"]
    files = ["./experiment/{}_{}_ours_{}_{}".format(dataset, attack, net, method) for method in methods]
    for lab,f in zip(labels, files):
        c, acc, distort = readf(f)
        plt.semilogx(c, 100*acc, label=lab)
    plt.xlabel("c")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()
