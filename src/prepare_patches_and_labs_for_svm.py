# -*- coding: utf-8 -*-

# Created on Sat Oct 15 14:28:46 2016

# @author: santosh, srikanth

"""
Parse annotations XML file, get object boundaries,
read patches boundaries, get +ve and -ve examples
for each object label and train N different SVMs
with chi^2 kernel.
"""

from crf_utils import ANNOT_DIR, feat_dir, DATA_PRE, patch_dir, save_fids
from crf_utils import get_RGB, N_LABELS
import xml.etree.ElementTree as ET
import os
import pickle
import sys
import cv2
import PIL
import numpy as np
from collections import defaultdict
from random import shuffle


def get_object_name_and_bbox(afile):

    fid = os.path.basename(afile).split(".")[0]
    box_lst = []
    obj_lst = []
    root = ET.parse(afile).getroot()

    for objc in root.findall("object"):

        box = []
        obj_name = ''
        for c1 in objc:
            if c1.tag == 'name':
                obj_name = c1.text
            elif c1.tag == "bndbox":
                for c2 in c1:
                    box.append(float(c2.text))
            else:
                pass

        box_lst.append(box)
        obj_lst.append(obj_name)

    return fid, obj_lst, box_lst


def parse_annotations(etc_d):

    fid_obj = {}  # FILEID: [OBJ_INDEX, ]
    fid_box = {}  # FILEID: [BBOX, ]
    obj_names = []  # seq of object names

    an_files = os.listdir(ANNOT_DIR)
    for i, anfile in enumerate(an_files):

        fid, obj_lst, box_lst = get_object_name_and_bbox(ANNOT_DIR + anfile)

        obj_ixs = []
        for o in obj_lst:
            if o not in obj_names:
                obj_names.append(o)
                obj_ixs.append(len(obj_names)-1)
            else:
                obj_ixs.append(obj_names.index(o))

        fid_obj[fid] = obj_ixs
        fid_box[fid] = box_lst

    """
    from pprint import pprint
    pprint(fid_obj)
    pprint(fid_box)
    pprint(obj_names)
    """

    pickle.dump(fid_obj, open(etc_d + "fid_obj.pkl", "wb"))
    pickle.dump(fid_box, open(etc_d + "fid_box.pkl", "wb"))
    pickle.dump(obj_names, open(etc_d + "obj_names.pkl", "wb"))

    print('No of fids:', len(fid_obj))
    print('No of labels:', len(obj_names))

    return fid_obj, fid_box, obj_names


def get_patch_info(pfile):

    pi = PIL.Image.open(pfile)
    n_patches = defaultdict(int)
    for pixel in pi.getdata():
        n_patches[pixel] += 1
    return n_patches


def get_me_the_label(pixels, box, labs):

    unq_lab, unq_ixs = np.unique(labs, return_index=True)

    if len(unq_lab) == 1:
        return unq_lab[0]

    else:
        return None


def get_gt_label_info(fid_box, fid_obj):
    """ Ground truth patch color (RGB) to label mapping """

    gt_files = os.listdir(SEG_DIR)
    gp_label = defaultdict(int)

    for gt_file in gt_files:

        if gt_file[-4:] != EXT:
            continue

        fid = gt_file.split(".")[0]
        gt_file = SEG_DIR + gt_file

        # 1. read the ground truth segmented file
        gt_patches = get_patch_info(gt_file)
        g_img = get_RGB(gt_file)

        for gp in gt_patches:
            if gp == CREAM:
                continue
            elif gp == BGR:
                label = N_LABELS + 1
            else:
                gt_pixels = np.where((g_img == gp).all(axis=2))
                box = fid_box[fid]
                labs = fid_obj[fid]
                label = get_me_the_label(gt_pixels, box, labs)

            if label is not None:
                gp_label[gp] = label

    return gp_label


def find_overlap_with_gt(pt_pixels, gp_label, fid):
    """ Find overlap of unsupervised patch with the ground truth patch """

    patch_label = N_LABELS + 1

    pt_xy = [(x, y) for x, y in zip(pt_pixels[0], pt_pixels[1])]
    gt_file = SEG_DIR + fid + EXT

    # 1. read the ground truth segmented file

    gt_patches = get_patch_info(gt_file)
    g_img = get_RGB(gt_file)

    for i, gp in enumerate(gt_patches):

        gt_pixels = np.where((g_img == gp).all(axis=2))

        gt_xy = [xy for xy in zip(gt_pixels[0], gt_pixels[1])]

        cnt = len(set(pt_xy) & set(gt_xy))

        if cnt >= (len(pt_xy) * 0.75):

            if gp == CREAM:
                patch_label = None

            elif gp == BGR:
                patch_label = N_LABELS + 1

            else:
                patch_label = gp_label[gp]

        else:

            if gp == CREAM:
                patch_label = N_LABELS + 1
            else:
                patch_label = int(-1 * gp_label[gp])

    return patch_label


def main():
    """ main """

    etc_d = feat_dir + "etc/"

    # 1.
    # fid_obj, fid_box, obj_names = parse_annotations(etc_d)

    fid_obj = pickle.load(open(etc_d + "fid_obj.pkl", "rb"))
    fid_box = pickle.load(open(etc_d + "fid_box.pkl", "rb"))
    obj_names = pickle.load(open(etc_d + "obj_names.pkl", "rb"))

    # 2. for each patched image in scale
    # for each patch, check

    gp_label = get_gt_label_info(fid_box, fid_obj)
    # from pprint import pprint
    # pprint(gp_label)
    # pickle.dump(gp_label, open(etc_d + "gp_label.pkl", "wb"))

    gp_label = pickle.load(open(etc_d + "gp_label.pkl", "rb"))

    gt_files = os.listdir(SEG_DIR)
    patch_subd = os.listdir(patch_dir)

    fids = [gtf.split(".")[0] for gtf in gt_files]
    shuffle(fids)

    train_size = int(len(fids) * 0.8)
    train_fids = fids[:train_size]
    test_fids = fids[train_size:]

    print("Trian, test, total:", len(train_fids), len(test_fids), len(fids))
    train_dir = DATA_PRE + "train/"
    test_dir = DATA_PRE + "test/"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    save_fids(train_fids, etc_d + "train.flist")
    save_fids(test_fids, etc_d + "test.flist")

    for fid in train_fids:

        fid = "2007_004112"
        for pd in patch_subd:

            pfile = patch_dir + pd + "/" + fid + EXT
            # print(pfile)
            p_img = get_RGB(pfile)

            n_patches = get_patch_info(pfile)
            # print("NP:", len(n_patches))

            for p in n_patches:
                # print('p:', p, 'p_img:', p_img.shape)
                patch_pixels = np.where((p_img == p).all(axis=2))
                # print('pp:', patch_pixels)

                pt_xy = [(x, y) for x, y in zip(patch_pixels[0],
                                                patch_pixels[1])]

                # find out the label of this patch_pixels
                # using annotation box and manual segmentation info
                patch_label = find_overlap_with_gt(patch_pixels, gp_label, fid)
                print("== patch ", patch_label, p, "==")

                # break
            break
        break




if __name__ == "__main__":

    SEG_DIR = DATA_PRE + "SegmentationClass_PPM/"
    EXT= ".ppm"
    CREAM = (224, 224, 192)
    BGR = (0, 0, 0)
    IGN = [CREAM, BGR]
    main()

