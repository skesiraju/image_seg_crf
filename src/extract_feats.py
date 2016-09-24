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
import argparse

from random import shuffle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import PIL
import cv2

import pickle


def extract_sift(ifile):
    """ Given image extract sift features """

    pass


def extract_hsv_features(ifile, pfile):
    """ Given img file, and patched img file, return hue-sat 2Dhist and
    val hist for each patch (10x10 + 10 = 110 dim feats)  """

    img = cv2.imread(ifile)
    p_img = cv2.imread(pfile)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    n_patches = PIL.Image.open(pfile).getcolors()

    hsv_feats = []
    for p in n_patches:
        patch_pixels = np.where(p_img == p[1])
        hsv_values = hsv_img[patch_pixels[0], patch_pixels[1], :]
        hs, _, _ = np.histogram2d(hsv_values[:, 0], hsv_values[:, 1], [10, 10])
        v, _ = np.histogram(hsv_values[:, 2], bins=10)
        hist_fea = np.concatenate((hs.reshape(100,), v))
        hsv_feats.append(hist_fea)

    return np.asarray(hsv_feats)


def main():
    """ main method """

    pwd = os.path.dirname(os.path.realpath(__file__)) + "/"

    image_dir = DATA_PRE + "PPMImages/"
    patch_dir = DATA_PRE + "patches/"
    feat_dir = DATA_PRE + "feats/"

    # Extract features for each image (patch wise)
    im_files = sorted(os.listdir(image_dir))

    patch_subd = os.listdir(patch_dir)

    for pd in patch_subd:
        os.system("mkdir -p " + feat_dir + pd)

    for i, ifile in enumerate(im_files):
        print("\r{0:d}/{1:d}".format(i+1, len(im_files)), end="")
        for pd in patch_subd:
            pfile = patch_dir + pd + "/" + os.path.basename(ifile)
            hsv_f = extract_hsv_features(image_dir + ifile, pfile)
            b = os.path.splitext(os.path.basename(ifile))[0]
            np.save(feat_dir + pd + "/" + b, hsv_f)
        # extract_sift(image_dir + ifile)


if __name__ == "__main__":

    DATA_PRE = "/home/santosh/Downloads/VOCdevkit/VOC2008/"

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
