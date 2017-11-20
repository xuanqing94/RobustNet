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
    filen = "./experiment/ensemble_acc"
    n, _, acc = readf(filen)
    plt.plot(n, 100 * acc)
    plt.xlabel('#Ensemble')
    plt.ylabel('Accuracy (%)')
    plt.show()
