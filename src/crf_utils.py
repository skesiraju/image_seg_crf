# -*- coding: utf-8 -*-
# Created on Sat Oct 15 14:31:47 2016
# @authors: santosh, srikanth

"""
Commmon utilities
"""

import os
import cv2
from PIL import Image
from collections import defaultdict
import numpy as np

DATA_PRE = os.environ['HOME'] + "/Downloads/VOCdevkit/VOC2008/"
ANNOT_DIR = DATA_PRE + "Annotations/"
IMAGE_DIR = DATA_PRE + "PPMImages/"
PATCH_DIR = DATA_PRE + "patches/"
FEAT_DIR = DATA_PRE + "feats/"
LAB_DIR = DATA_PRE + "ImageSets/Main/"
SEG_DIR = DATA_PRE + "SegmentationClass_PPM/"
ETC_D = FEAT_DIR + "etc/"
TREE_DIR = FEAT_DIR + "tree/"

EXT = ".ppm"

N_LABELS = 20
BGR_LABEL = 21

CREAM = (224, 224, 192)
BGR = (0, 0, 0)
IGN = [CREAM, BGR]

SCALE_CONFIGS = ['s_0.5_k_500_m_50', 's_0.75_k_500_m_200', 's_1.25_k_700_m_1200']

# a_l, b_l slope and bias of logistic sigmoid at each scale level
AB_LS = [(2., 0.), (1.5, 0.), (1.5, 0.)]

# gamma_l param for patch similarity in the hierarchy
GLS = [2, 1]


def save_fids(fids, fname):

    with open(fname, 'w') as fpw:
        fpw.write("\n".join(fids) + "\n")


def get_RGB(fname):
    
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_patch_info(pfile):
    """ Returns dictionary where key is (R,G,B) and value is
    number of pixels """

    pi = Image.open(pfile)
    n_patches = defaultdict(int)
    for pixel in pi.getdata():
        n_patches[pixel] += 1
    return n_patches

def sigmoid(x, a_l, b_l):

    return 1. / (1.+ np.exp(-((a_l * x) + b_l)))

def shuffle_data_and_labels(data, labels):

    data_lab = np.concatenate((data, labels.reshape((len(labels), 1))),
                              axis=1)
    np.random.shuffle(data_lab)

    return data_lab[:, :-1], data_lab[:, -1]
