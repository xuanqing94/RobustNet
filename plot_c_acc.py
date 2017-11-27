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
    lw = 1.5
    plt.subplot(231)
    dataset = "stl10"
    attack = "cw"
    net = "stl10_model"
    methods = ["plain", "adv", "brelu", "rse", "dd"]
    labels = ["No defense", "Adv retraining", "Robust Opt+BReLU", "RSE", "Distill"]
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
    files = ["./accuracy/{}_{}_{}_{}.acc".format(attack, dataset, net, method) for method in methods]
    for col, lab,f in zip(colors, labels, files):
        c, acc, distort = readf(f)
        plt.semilogx(c, 100*acc, label=lab, color=col, linestyle='-', lw=lw)
    plt.xlabel('C')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc=1)

    plt.subplot(234)
    for col, lab, f in zip(colors, labels, files):
        c, acc, distort = readf(f)
        plt.semilogx(c, distort, label=lab, color=col, linestyle='-', lw=lw)
    plt.xlabel('C')
    plt.ylabel('Avg. Distortion')
    plt.legend(loc=2)

    # ==================================================================================================

    plt.subplot(232)
    dataset = "cifar10"
    attack = "cw"
    net = "vgg16"
    methods = ["plain", "adv", "brelu", "rse", "dd"]
    labels = ["No defense", "Adv retraining", "Robust Opt+BReLU", "RSE", "Distill"]
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
    files = ["./accuracy/{}_{}_{}_{}.acc".format(attack, dataset, net, method) for method in methods]
    for col, lab,f in zip(colors, labels, files):
        c, acc, distort = readf(f)
        plt.semilogx(c, 100*acc, label=lab, color=col, linestyle='-', lw=lw)
    plt.xlabel('C')
    #plt.ylabel('Accuracy (%)')
    plt.legend(loc=1)

    plt.subplot(235)
    for col, lab, f in zip(colors, labels, files):
        c, acc, distort = readf(f)
        plt.semilogx(c, distort, label=lab, color=col, linestyle='-', lw=lw)
    plt.xlabel('C')
    #plt.ylabel('Avg. Distortion')
    plt.legend(loc=2)

    # ================================================================================================

    plt.subplot(233)
    dataset = "cifar10"
    attack = "cw"
    net = "resnext"
    methods = ["plain", "adv", "brelu", "rse", "dd"]
    labels = ["No defense", "Adv retraining", "Robust Opt+BReLU", "RSE", "Distill"]
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
    files = ["./accuracy/{}_{}_{}_{}.acc".format(attack, dataset, net, method) for method in methods]
    for col, lab,f in zip(colors, labels, files):
        c, acc, distort = readf(f)
        plt.semilogx(c, 100*acc, label=lab, color=col, linestyle='-', lw=lw)
    plt.xlabel('C')
    #plt.ylabel('Accuracy (%)')
    plt.legend(loc=1)

    plt.subplot(236)
    for col, lab, f in zip(colors, labels, files):
        c, acc, distort = readf(f)
        plt.semilogx(c, distort, label=lab, color=col, linestyle='-', lw=lw)
    plt.xlabel('C')
    #plt.ylabel('Avg. Distortion')
    plt.legend(loc=2)



    plt.show()
