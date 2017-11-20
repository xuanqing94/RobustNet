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
    f1 = "./experiment/cifar10_CW_ours_vgg16_0.9_0.2_50"
    f2 = "./experiment/cifar10_CW_ours_vgg16_0.9_0.2_1_tmp"
    f3 = "./experiment/clean_clean_ensemble"
    legends = ["Train+Test noise", "Train noise only", "Test noise only"]
    for i, f in enumerate([f1, f2, f3]):
        c, acc, distort = readf(f)
        plt.semilogx(c, 100 * acc, label=legends[i])
    plt.xlabel("c")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc=1)
    plt.show()
