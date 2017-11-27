#!/usr/bin/env python3

import matplotlib.pyplot as plt
from scipy.misc import imread


if __name__ == "__main__":
    methods = ("plain", "dd", "adv", "brelu", "rse")
    targets = ("bird", "car", "cat", "deer", "dog", "frog", "horse", "plane", "truck")
    fig, axs = plt.subplots(nrows=len(methods), ncols=len(targets))
    fig.patch.set_visible(False)
    for i, method in enumerate(methods):
        for j, target in enumerate(targets):
            ax = axs[i, j]
            img = imread("./images/{}_ship_{}.png".format(method, target))
            #ax.axis('off')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img, interpolation='nearest', extent=[0, 64, 0, 64])
            if j == 0:
                ax.set_ylabel(method)
            if i == len(methods) - 1:
                ax.set_xlabel(target)
    #plt.figure(dpi=120)
    #plt.show()
    plt.savefig('./gallery.png', dpi=100)
