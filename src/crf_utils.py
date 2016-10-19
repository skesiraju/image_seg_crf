# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:31:47 2016

@author: santosh, srikanth
"""

import cv2

DATA_PRE = "/home/santosh/Downloads/VOCdevkit/VOC2008/"
ANNOT_DIR = DATA_PRE + "Annotations/"
image_dir = DATA_PRE + "PPMImages/"
patch_dir = DATA_PRE + "patches/"
feat_dir = DATA_PRE + "feats/"
EXT = ".ppm"

N_LABELS = 20


def save_fids(fids, fname):

    with open(fname, 'w') as fpw:
        fpw.write("\n".join(fids) + "\n")


def get_RGB(fname):

    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
