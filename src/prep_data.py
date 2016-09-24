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

import xml.etree.ElementTree


def parse_annotations(annot_dir):
    """ Parse annotations, get labels for image and the objects in it """

    labels = []
    fid_labels = {}

    afiles = os.listdir(annot_dir)
    for af in afiles:
        fid = os.path.splitext(os.path.basename(af))[0]
        root = xml.etree.ElementTree.parse(annot_dir + af).getroot()
        for child in root:
            if child.tag == 'object':
                for sub in child:
                    if sub.tag == 'name':
                        l = sub.text
                        try:
                            l_ix = labels.index(l)
                        except ValueError:
                            labels.append(l)
                            l_ix = len(labels) - 1

                        fid_labels[fid] = [l_ix]

                    elif sub.tag == 'bndbox':
                        bbox = []
                        for c in sub:
                            bbox.append(int(float(c.text)))
                        fid_labels[fid] += [bbox]

    return fid_labels, labels


def main():
    """ main method """

    pwd = os.path.dirname(os.path.realpath(__file__)) + "/"

    tmp_d = pwd + "../tmp/"
    etc_d = pwd + "../etc/"

    os.system("mkdir -p " + etc_d)
    os.system("mkdir -p " + tmp_d)

    annot_dir = DATA_PRE + "Annotations/"

    # Parse annotations and prepare fIDs labels and boundaries
    fid_labels, labels = parse_annotations(annot_dir)
    pickle.dump(fid_labels, open(etc_d + "fid_labels.pkl", "wb"))
    with open(etc_d + "labels.txt", "w") as fpw:
        fpw.write("\n".join(labels) + "\n")


if __name__ == "__main__":

    DATA_PRE = "/home/santosh/Downloads/VOCdevkit/VOC2008/"

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
