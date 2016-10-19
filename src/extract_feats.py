#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh, Srikanth
# Date created : 19 Sep 2016
# Last modified : 19 Sep 2016

"""
Color feature extraction
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
from crf_utils import get_RGB, EXT, FEAT_DIR, IMAGE_DIR, PATCH_DIR


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

    patch_subd = os.listdir(PATCH_DIR)
    for i, ifile in enumerate(lst):
        if ifile[-4:] != EXT:
                continue
        print("\r{0:d}/{1:d}".format(i+1, len(lst)), end="")
        # extract_sift(ifile, pfile)
        for pd in patch_subd:
            pfile = PATCH_DIR + pd + "/" + os.path.basename(ifile)
            hsv_f = extract_hsv_features(IMAGE_DIR + ifile, pfile)
            b = os.path.splitext(os.path.basename(ifile))[0]
            if b is not None:
                np.save(FEAT_DIR + "color_feats/" + pd + "/" + b, hsv_f)


def main():
    """ main method """

    # Extract features for each image (patch wise)
    im_files = sorted(os.listdir(IMAGE_DIR))
    os.system("mkdir -p " + FEAT_DIR + "color_feats/")

    patch_subd = os.listdir(PATCH_DIR)

    for pd in patch_subd:
        os.system("mkdir -p " + FEAT_DIR + pd)

    pool = Pool(4)
    chunks = chunkify(im_files, 4)

    pool.map(par_feat_ext, chunks)
    pool.close()
    pool.join()

    # par_feat_ext(im_files)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
