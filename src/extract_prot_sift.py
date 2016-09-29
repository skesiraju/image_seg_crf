# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:07:26 2016

@authors: Santosh, Srikanth
"""


import os
import argparse

import numpy as np

import pickle
import PIL
import cv2
from sklearn.cluster import MiniBatchKMeans
import warnings
from multiprocessing import Pool
from collections import defaultdict
from misc.io import chunkify, read_simple_flist


def get_descriptors(ifile):

    img = cv2.imread(ifile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()  # raidus parameters here

    kp, des = sift.detectAndCompute(gray, None)

    return kp, des


def feat_ext():

    pwd = os.path.dirname(os.path.realpath(__file__)) + "/"

    feat_d = DATA_PRE + "feats/"
    img_dir = DATA_PRE + "PPMImages/"
    lab_d = DATA_PRE + "ImageSets/Main/"
    tmp_d = pwd + "../tmp/"
    etc_d = pwd + "../etc/"

    os.system("mkdir -p " + etc_d)
    os.system("mkdir -p " + tmp_d)

    # fid_labels = pickle.load(open(etc_d + "fid_labels.pkl", "rb"))
    # labels = read_simple_flist(etc_d + "labels.txt")

    # train_fids = read_simple_flist(lab_d + "train.txt")
    # val_fids = read_simple_flist(lab_d + "val.txt")

    train_flist = sorted(read_simple_flist(lab_d + "train.txt", pre=img_dir,
                                           sfx=".ppm"))

    # val_flist = sorted(read_simple_flist(lab_d + "val.txt", pre=img_dir,
    #                                     sfx=".ppm"))

    train_fids = []
    all_d = []
    all_k = []
    file_kp_size = []

    st = 0
    for i, tf in enumerate(train_flist):

        k, d = get_descriptors(tf)

        all_d.append(d)  # append all desc into one list

        # get the file ID (unique key) of the image
        fid = os.path.splitext(os.path.basename(tf))[0]
        train_fids.append(fid)

        pickle.dump(all_k, open(feat_d + "kp/kp_" + fid + ".pkl", "wb"))

        # save file ID to no of key points in dict
        file_kp_size.append([st, st + len(k)])
        st += len(k)

    all_d = np.concatenate(all_d)
    np.save(feat_d + "sift_train.npy", all_d)
    print('all desc:', all_d.shape, 'saved.')

    with open(feat_d + "train_fids.list", "w") as fpw:
        fpw.write("\n".join(train_fids))

    file_kp_size = np.asarray(file_kp_size)
    np.save(feat_d + "kp_index.npy", file_kp_size)
    print(file_kp_size.shape, 'saved.')

    print('Done')


def cluster_feats():
    """ Cluster SIFT feats """

    sift_data = np.load(feat_dir + "sift_train.npy")
    print('SIFT data:', sift_data.shape)

    kmeans = MiniBatchKMeans(init='k-means++', n_clusters=1000,
                             batch_size=1000, n_init=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_ixs = kmeans.fit_predict(sift_data)
        np.save(feat_dir + "sift_vq.npy", c_ixs)
    print('clustering done')


def compute_patch_sift_hist(ifile, pfile, f_c_ixs):
    """ Given img file, patched img file and corresponding VQ indices of the
    keypoints, return histogram features """

    def is_kp_in_patch(pp, kp_pt):
        """ Check if key point is in patch """
        flag = False
        x_min = pp[0].min()
        x_max = pp[0].max()
        y_min = pp[1].min()
        y_max = pp[1].max()
        if kp_pt[0] >= x_min and kp_pt[0] <= x_max:
            if kp_pt[1] >= y_min and kp_pt[1] <= y_max:
                flag = True
        return flag

    # img = cv2.imread(ifile)
    fid = os.path.splitext(os.path.basename(ifile))[0]

    kpts = pickle.load(open(feat_dir + "kp/kp_" + fid + ".pkl", "rb"))

    p_img = cv2.imread(pfile)

    pi = PIL.Image.open(pfile)
    n_patches = defaultdict(int)
    for pixel in pi.getdata():
        n_patches[pixel] += 1

    if n_patches is None:
        print("\nNo patches:", pfile)
        return None

    hist_feats = []
    for p in n_patches:
        patch_pixels = np.where(p_img == p[0])
        patch_cixs = []

        for kp_ix, kp in enumerate(kpts):
            if is_kp_in_patch(patch_pixels, kp):
                patch_cixs.append(f_c_ixs[kp_ix])

        h, _ = np.histogram(patch_cixs, bins=1000)
        hist_feats.append(h)

    hist_feats = np.asarray(hist_feats)
    return hist_feats


def parallel_sift_hist_feat_ext(lst):
    """ Parallel SIFT histogram feature extraction """

    kp_index = np.load(feat_dir + "kp_index.npy")
    c_ixs = np.load(feat_dir + "sift_vq.npy")
    train_fids = read_simple_flist(feat_dir + "train_fids.list")

    patch_subd = os.listdir(patch_dir)
    for i, fid in enumerate(lst):

        ifile = image_dir + fid + EXT
        f_ix = train_fids.index(fid)
        k_ix = kp_index[f_ix]
        f_c_ixs = c_ixs[k_ix[0]:k_ix[1]]

        print("\r{0:d}/{1:d}".format(i+1, len(lst)), end="")

        for pd in patch_subd:
            pfile = patch_dir + pd + "/" + fid + EXT
            patch_hist_feats = compute_patch_sift_hist(image_dir + ifile,
                                                       pfile, f_c_ixs)

            np.save(feat_dir + "hist_feats/" + pd + "/" + fid + ".npy",
                    patch_hist_feats)


def main():
    """ main """

    patch_subd = os.listdir(patch_dir)
    for pd in patch_subd:
        os.system("mkdir -p " + feat_dir + "hist_feats/" + pd)

    if args.choice == "feats":
        feat_ext()

    elif args.choice == "cluster":
        cluster_feats()

    elif args.choice == "hist":
        train_fids = read_simple_flist(feat_dir + "train_fids.list")
        chunks = chunkify(train_fids, 4)

        pool = Pool(4)
        pool.map(parallel_sift_hist_feat_ext, chunks)
        pool.close()
        pool.join()
        # parallel_sift_hist_feat_ext(chunks[0])

    else:
        print("Invalid choice")

if __name__ == "__main__":

    DATA_PRE = "/home/santosh/Downloads/VOCdevkit/VOC2008/"
    image_dir = DATA_PRE + "PPMImages/"
    patch_dir = DATA_PRE + "patches/"
    feat_dir = DATA_PRE + "feats/"
    EXT = ".ppm"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("choice", help="feats or cluster")
    args = parser.parse_args()
    main()
