# -*- coding: utf-8 -*-
# Created on Sat Oct 15 14:28:46 2016
# @authors: Santosh, Srikanth

"""
Parse annotations XML file, get object boundaries,
read patches boundaries, get +ve and -ve examples
for each object label and train N different SVMs
with chi^2 kernel.
"""

from crf_utils import ANNOT_DIR, EXT, FEAT_DIR, DATA_PRE, PATCH_DIR, SEG_DIR
from crf_utils import N_LABELS, CREAM, BGR, BGR_LABEL
from crf_utils import get_RGB, save_fids, get_patch_info

import argparse
import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
from collections import defaultdict
from random import shuffle
from multiprocessing import Pool, cpu_count
from misc.io import read_simple_flist, chunkify


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
                obj_ixs.append(len(obj_names))
            else:
                obj_ixs.append(obj_names.index(o)+1)

        fid_obj[fid] = obj_ixs
        fid_box[fid] = box_lst

    pickle.dump(fid_obj, open(etc_d + "fid_obj.pkl", "wb"))
    pickle.dump(fid_box, open(etc_d + "fid_box.pkl", "wb"))
    pickle.dump(obj_names, open(etc_d + "obj_names.pkl", "wb"))

    print('No of fids:', len(fid_obj))
    print('No of labels:', len(obj_names))

    return fid_obj, fid_box, obj_names


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
                label = BGR_LABEL
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

    patch_label = BGR_LABEL

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
                patch_label = BGR_LABEL

            else:
                patch_label = gp_label[gp]

            break  # if > 75 overlap is found, return the label

        else:

            if gp == CREAM:
                patch_label = BGR_LABEL
            else:
                patch_label = int(-1 * gp_label[gp])

    return patch_label


def parallel_data_prep_svm(lst_fids):
    """ Save label information for each patch at each scale level.
    Parallel execution """

    info_dir = FEAT_DIR + "patch_label_info/"
    patch_subd = os.listdir(PATCH_DIR)

    for fid in lst_fids:

        patch_label_lst = []
        for pd in patch_subd:

            pfile = PATCH_DIR + pd + "/" + fid + EXT
            p_part = "/".join(pfile.replace(EXT, "").split("/")[-2:])
            p_img = get_RGB(pfile)
            n_patches = get_patch_info(pfile)

            for p in n_patches:

                patch_pixels = np.where((p_img == p).all(axis=2))

                # find out the label of this patch_pixels
                # using manual segmentation info
                patch_label = find_overlap_with_gt(patch_pixels, GP_LABEL, fid)
                patch_label_lst.append([patch_label, p, p_part])

                # print(patch_label, p, p_part)

        pickle.dump(patch_label_lst, open(info_dir + fid + ".pkl", "wb"))



def parallel_feat_map(info_files):
    """ Map patch labels and color and histogram features and save them
    respective sub directories. Parallel execution """

    sub_patch_dir = os.listdir(PATCH_DIR)

    for f in info_files:
        lst = pickle.load(open(info_d + f, 'rb'))

        label_dict = {}
        patch_dict = {}
        patch_feat_dict = {}
        for i, el in enumerate(lst):

            lab = el[0]  # object label 
            patch = el[1]  # patch color
            ppath = el[2]  # partial path
            p_config = ppath.split("/")[0]

            try:
                n_patches = patch_dict[ppath]

            except KeyError:                

                p_file = PATCH_DIR + ppath + EXT
                n_patches = get_patch_info(p_file)
                patch_dict[ppath] = n_patches  # put patch info in dict

            c_file = color_d + ppath + ".npy"
            h_file = hist_d + ppath + ".npy"              

            c_map = pickle.load(open(c_file.replace("npy", "pkl"), "rb"))
            h_map = pickle.load(open(h_file.replace("npy", "pkl"), "rb"))

            c_feat = np.load(c_file)[c_map.index(patch), :]
            h_feat = np.load(h_file)[h_map.index(patch), :]

            a_feat = np.concatenate((c_feat, h_feat))            

            patch_feat_dict = {}
            try:
                patch_feat_dict = label_dict[lab]
                try:
                    patch_feat_dict[p_config].append(a_feat)
                    
                except KeyError:
                    patch_feat_dict[p_config] = [a_feat]

            except KeyError:
                patch_feat_dict[p_config] = [a_feat]            
            
            label_dict[lab] = patch_feat_dict         
                         
        for lab, pf_dict in label_dict.items():

            for p_conf, feats in pf_dict.items():

                lab_d = train_d + p_conf + "/" + "l_" + str(lab) + "/"
                os.makedirs(lab_d, exist_ok=True)

                fname = lab_d + os.path.basename(f).split(".")[0] + ".npy"
                np.save(fname, np.asarray(feats))


def main():
    """ main """

    global GP_LABEL

    etc_d = FEAT_DIR + "etc/"

    if args.choice == "annot":   

        fid_obj, fid_box, obj_names = parse_annotations(etc_d)

    elif args.choice == "gt":

        fid_obj = pickle.load(open(etc_d + "fid_obj.pkl", "rb"))
        fid_box = pickle.load(open(etc_d + "fid_box.pkl", "rb"))

        GP_LABEL = get_gt_label_info(fid_box, fid_obj)
        from pprint import pprint
        pprint(GP_LABEL)
        pickle.dump(GP_LABEL, open(etc_d + "gp_label.pkl", "wb"))

    elif args.choice == "prep":

        GP_LABEL = pickle.load(open(etc_d + "gp_label.pkl", "rb"))
        print("GP labels:", len(GP_LABEL))

        if os.path.exists(etc_d + "train.flist") is False:
            print("* Generating train and test splits .....")
            gt_files = os.listdir(SEG_DIR)
            fids = [gtf.split(".")[0] for gtf in gt_files]
            shuffle(fids)

            train_size = int(len(fids) * 0.8)
            train_fids = fids[:train_size]
            test_fids = fids[train_size:]

            print("Trian, test, total:", len(train_fids), len(test_fids),
                  len(fids))

            save_fids(train_fids, etc_d + "train.flist")
            save_fids(test_fids, etc_d + "test.flist")
            save_fids(train_fids + test_fids, etc_d + "all.flist")

        else:

            train_fids = read_simple_flist(etc_d + "train.flist")
            test_fids = read_simple_flist(etc_d + "test.flist")

        info_dir = FEAT_DIR + "patch_label_info/"
        os.makedirs(info_dir, exist_ok=True)

        all_fids = train_fids + test_fids
        print("File IDs:", len(all_fids))

        n_jobs = int(cpu_count() / 2)

        if n_jobs > 10:
            n_jobs = 20

        print('n_jobs:', n_jobs)

        chunks = chunkify(all_fids, n_jobs)
        
        pool = Pool(n_jobs)
        pool.map(parallel_data_prep_svm, chunks)
        pool.close()
        pool.join()


    elif args.choice == "map":

        os.system("mkdir -p " + train_d)
        
        info_files = os.listdir(info_d)
        print("Info files:", len(info_files))
    
        n_jobs = int(cpu_count() / 2)
        chunks = chunkify(info_files, n_jobs)

        # parallel_feat_map(info_files[:2])
        
        p = Pool(n_jobs)
        p.map(parallel_feat_map, chunks)
        p.close()
        p.join()     

if __name__ == "__main__":

    GP_LABEL = {}

    hist_d = FEAT_DIR + "hist_feats/"
    color_d = FEAT_DIR + "color_feats/"
    info_d = FEAT_DIR + "patch_label_info/"
    train_d = FEAT_DIR + "train/"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("choice", help="feats or cluster")
    args = parser.parse_args()

    main()
