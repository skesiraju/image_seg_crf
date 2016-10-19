# -*- coding: utf-8 -*-
# Created on Sat Oct 15 14:31:47 2016
# @authors: santosh, srikanth

"""
Commmon utilities
"""

import cv2
import PIL
from collections import defaultdict

DATA_PRE = "/home/santosh/Downloads/VOCdevkit/VOC2008/"
ANNOT_DIR = DATA_PRE + "Annotations/"
IMAGE_DIR = DATA_PRE + "PPMImages/"
PATCH_DIR = DATA_PRE + "patches/"
FEAT_DIR = DATA_PRE + "feats/"
LAB_DIR = DATA_PRE + "ImageSets/Main/"
SEG_DIR = DATA_PRE + "SegmentationClass_PPM/"
ETC_D = FEAT_DIR + "etc/"

EXT = ".ppm"

N_LABELS = 20

CREAM = (224, 224, 192)
BGR = (0, 0, 0)
IGN = [CREAM, BGR]


def save_fids(fids, fname):

    with open(fname, 'w') as fpw:
        fpw.write("\n".join(fids) + "\n")


def get_RGB(fname):

    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_patch_info(pfile):

    pi = PIL.Image.open(pfile)
    n_patches = defaultdict(int)
    for pixel in pi.getdata():
        n_patches[pixel] += 1
    return n_patches
