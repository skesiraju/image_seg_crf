#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Srikanth, Santosh
# Date created : 22 Oct 2016
# Last modified : 22 Oct 2016

"""
"""

import os 
import sys 
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
import pickle 
from crf_utils import ETC_D, SCALE_CONFIGS, PATCH_DIR, EXT, TREE_DIR
from crf_utils import get_patch_info, get_RGB
from misc.io import read_simple_flist, chunkify


def find_overlap(p1, p2_info, s1_ifile, s2_ifile):
 
    s1_img = get_RGB(s1_ifile)
    s2_img = get_RGB(s2_ifile)
    p1_pixels = np.where((s1_img == p1).all(axis=2))
    p1_xy = [xy for xy in zip(p1_pixels[0], p1_pixels[1])]
    max_cnt = 0
    max_overlap_patch = None

    for p2 in p2_info:
        p2_pixels = np.where((s2_img == p2).all(axis=2))
        p2_xy = [xy for xy in zip(p2_pixels[0], p2_pixels[1])]

        cnt = len(set(p1_xy) & set(p2_xy))
        if cnt > max_cnt:
            max_overlap_patch = p2
            max_cnt = cnt
            
    return max_cnt, max_overlap_patch


def parallel_tree_construct(fids):

    for f_id in fids:
        f_dict = {}
        for i in range(len(SCALE_CONFIGS)-1):
            s1, s2 = SCALE_CONFIGS[i: i+2]
            s1_ifile = PATCH_DIR + s1 + "/" + f_id + EXT
            s2_ifile = PATCH_DIR + s2 + "/" + f_id + EXT
            p1_info = get_patch_info(s1_ifile)
            p2_info = get_patch_info(s2_ifile)
            p_dict = {}
            for p1 in p1_info:
                max_cnt, max_overlap_patch = find_overlap(p1, p2_info,
                                                          s1_ifile, s2_ifile)
                p_dict[p1] = max_overlap_patch
            f_dict[str(i) + str(i+1)] = p_dict
            

        pickle.dump(f_dict, open(TREE_DIR + f_id + ".pkl", "wb"))


def main():
    """ main method """

    os.system("mkdir -p " + TREE_DIR)

    n_jobs = int(cpu_count() / 2)
    
    flist = read_simple_flist(ETC_D + "all.flist")

    chunks = chunkify(flist, n_jobs)

    pool = Pool(n_jobs)
    pool.map(parallel_tree_construct, chunks)
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
