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
    methods = ["0.9_0.2_1", "0.9_0.2_50"]
    labels = ["1-ensemble", "50-ensemble"]
    files = ["./experiment/{}_{}_ours_{}_{}".format(dataset, attack, net, method) for method in methods]
    for lab,f in zip(labels, files):
        c, acc, distort = readf(f)
        plt.plot(distort, 100*acc, label=lab)
    plt.xlabel("Avg distortion")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("CIFAR10+VGG16")
    plt.show()
