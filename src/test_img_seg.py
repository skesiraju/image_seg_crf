#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 19 Sep 2016
# Last modified : 19 Sep 2016

"""
Feature extraction test code
"""

from __future__ import print_function
import os
import sys
import argparse

from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage as ndi

from skimage import feature
from skimage import morphology
from skimage.morphology import watershed
from skimage.feature import peak_local_max, canny
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.data import camera
from scipy import ndimage as ndi


import cv2


def extract_sift(ifile):
    """ Given image extract sift features """

    print(ifile, end=' ')
    img = cv2.imread(ifile)
    print(type(img), img.shape)

    img_gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(type(img_gr), img_gr.shape)
    cv2.imwrite('tmp/res.jpg', img_gr)

    """
    color_scales = [cv2.COLOR_BGR2HLS, cv2.COLOR_RGB2HLS, cv2.COLOR_HLS2BGR,
                    cv2.COLOR_HLS2RGB]

    for i, cs in enumerate(color_scales):
        img_cs = cv2.cvtColor(img, cs)
        cv2.imwrite('tmp/res_' + str(i) + '.jpg', img_cs)

    cv2.imwrite('tmp/ori.jpg', img)
    """

    s1 = 3
    s2 = 8
    edges1 = feature.canny(img_gr, sigma=s1)
    edges2 = feature.canny(img_gr, sigma=s2)

    print(type(edges1), edges1.shape)

    # display results
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3),
                                             sharex=True, sharey=True)

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    ax1.set_title('ori image', fontsize=20)

    ax2.imshow(img_gr, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('gray image', fontsize=20)

    ax3.imshow(edges1, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('$\sigma='+str(s1)+'$', fontsize=20)

    ax4.imshow(edges2, cmap=plt.cm.gray)
    ax4.axis('off')
    ax4.set_title('$\sigma='+str(s2)+'$', fontsize=20)

    fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                        bottom=0.02, left=0.02, right=0.98)

    plt.show()

    distance = ndi.distance_transform_edt(edges2)
    local_maxi = peak_local_max(distance, indices=False,
                                footprint=np.ones((3, 3)), labels=edges1)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=edges1)

    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7), sharex=True,
                             sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax0, ax1, ax2 = axes

    ax0.imshow(edges1, cmap=plt.cm.gray, interpolation='nearest')
    ax0.set_title('Overlapping objects')
    ax1.imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_title('Distances')
    ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
    ax2.set_title('Separated objects')

    for ax in axes:
        ax.axis('off')

    fig.tight_layout()
    plt.show()

    edge_roberts = roberts(img_gr)
    edge_sobel = sobel(img_gr)

    fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, sharey=True,
                                   subplot_kw={'adjustable': 'box-forced'})

    ax0.imshow(edge_roberts, cmap=plt.cm.gray)
    ax0.set_title('Roberts Edge Detection')
    ax0.axis('off')

    ax1.imshow(edge_sobel, cmap=plt.cm.gray)
    ax1.set_title('Sobel Edge Detection')
    ax1.axis('off')

    plt.tight_layout()


def main():
    """ main method """

    os.system("mkdir -p tmp")
    image_dir = DATA_PRE + "JPEGImages/"
    im_files = os.listdir(image_dir)
    shuffle(im_files)

    for ifile in im_files:
        ifile = '2008_006452.jpg'
        extract_sift(image_dir + ifile)
        break


if __name__ == "__main__":

    DATA_PRE = "/home/santosh/Downloads/VOCdevkit/VOC2008/"

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
