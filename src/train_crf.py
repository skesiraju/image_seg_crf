#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 30 Oct 2016
# Last modified : 30 Oct 2016

"""
Train CRF
"""

from misc.io import read_simple_flist
from crf_utils import ETC_D, PATCH_LAB_DIR, TREE_DIR
import os
import sys
import pickle
import argparse


def main():
    """ main method """

    train_fids = read_simple_flist(ETC_D + "train.flist")
    fid_obj = pickle.load(open(ETC_D + 'fid_obj.pkl', 'rb'))

    # for each image, get the patch files
    # load the patch level ground truth and the svm output
    # load the ground truth labels for each patch

    for fid in train_fids:
        fid = '2007_000032'
        print(fid, fid_obj[fid])
        patch_gt = pickle.load(open(PATCH_LAB_DIR + fid + ".pkl", "rb"))
        tree = pickle.load(open(TREE_DIR + fid + ".pkl", "rb"))

        v01 = list(tree['01'].values())
        v12k = tree['12'].keys()

        tmp = {}
        for v0 in v01:
            try:
                tmp[v0] += 1
            except KeyError:
                tmp[v0] = 1

        print(len(v01), len(tmp), len(v12k))

        break

if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description=__doc__)
    ARGS = PARSER.parse_args()
    main()
