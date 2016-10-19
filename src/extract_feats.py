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

# from random import shuffle
import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

import PIL
import cv2

from multiprocessing import Pool
from collections import defaultdict
from misc.io import chunkify
from crf_utils import get_RGB, EXT


def extract_sift(ifile, pfile):
    """ Given image extract sift features """

    img = cv2.imread(ifile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    kp, des = sift.detectAndCompute(gray, None)
    print('kp, des:', len(kp), des.shape)


def extrac_texture_features():

    pass


def extract_hsv_features(ifile, pfile):
    """ Given img file, and patched img file, return hue-sat 2Dhist and
    val hist for each patch (10x10 + 10 = 110 dim feats)  """

    img = get_RGB(ifile)
    p_img = get_RGB(pfile)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    pi = PIL.Image.open(pfile)

    n_patches = defaultdict(int)
    for pixel in pi.getdata():
        n_patches[pixel] += 1

    if n_patches is None:
        print("\nNo patches:", pfile)
        return None

    hsv_feats = []
    for p in n_patches:
        patch_pixels = np.where((p_img == p).all(axis=2))
        hsv_values = hsv_img[patch_pixels[0], patch_pixels[1], :]
        hs, _, _ = np.histogram2d(hsv_values[:, 0], hsv_values[:, 1], [10, 10])
        v, _ = np.histogram(hsv_values[:, 2], bins=10)
        hist_fea = np.concatenate((hs.reshape(100,), v))
        hsv_feats.append(hist_fea)

    return np.asarray(hsv_feats)


def par_feat_ext(lst):

    patch_subd = os.listdir(patch_dir)
    for i, ifile in enumerate(lst):
        if ifile[-4:] != EXT:
                continue
        print("\r{0:d}/{1:d}".format(i+1, len(lst)), end="")
        # extract_sift(ifile, pfile)
        for pd in patch_subd:
            pfile = patch_dir + pd + "/" + os.path.basename(ifile)
            hsv_f = extract_hsv_features(image_dir + ifile, pfile)
            b = os.path.splitext(os.path.basename(ifile))[0]
            if b is not None:
                np.save(feat_dir + pd + "/" + b, hsv_f)


def main():
    """ main method """

    # Extract features for each image (patch wise)
    im_files = sorted(os.listdir(image_dir))

    patch_subd = os.listdir(patch_dir)

    for pd in patch_subd:
        os.system("mkdir -p " + feat_dir + pd)

    pool = Pool(4)
    chunks = chunkify(im_files, 4)

    pool.map(par_feat_ext, chunks)
    pool.close()
    pool.join()

    # par_feat_ext(im_files)


if __name__ == "__main__":

    DATA_PRE = "/home/santosh/Downloads/VOCdevkit/VOC2008/"
    image_dir = DATA_PRE + "PPMImages/"
    patch_dir = DATA_PRE + "patches/"
    feat_dir = DATA_PRE + "feats/"

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
