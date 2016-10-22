#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh, Srikanth
# Date created : 19 Oct 2016
# Last modified : 19 Oct 2016

"""
SVM with chi^2 kernel
"""

import os 
import sys 
import argparse 
import pickle
import numpy as np

from math import floor

from multiprocessing import cpu_count, Pool

from sklearn.base import TransformerMixin
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import chi2_kernel

from crf_utils import PATCH_DIR, FEAT_DIR, BGR_LABEL, N_LABELS, SCALE_CONFIGS
from crf_utils import AB_LS, sigmoid, shuffle_data_and_labels

from misc.io import chunkify


class CustomCHI(TransformerMixin):

    def __init__(self, gamma=1.0):   
        self.gamma = gamma

    def fit(self, X):
        dist = np.mean(chi2_kernel(X, gamma=-1.0))
        self.gamma = 0.5 * (1.0/dist)
        return self

    def fit_transform(self, X):
        self.fit(X)
        return chi2_kernel(X_train, gamma=self.gamma)

    def transform(self, X):
        return chi2_kernel(X_train, gamma=self.gamma)

    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        


def get_njobs(flag=True):

    nj = int(cpu_count() / 2)
    if flag and nj > 10:
        nj = 20
    return nj
    

def load_training_data(input_fpaths):

    return np.concatenate([np.load(f) for f in input_fpaths])
    

def balance_data(data1, data2_size, fx):

    if fx >= 2:
        data1 = np.repeat(data1, fx, axis=0)

    df = data2_size - data1.shape[0]
    ixs = np.zeros(data1.shape[0], dtype=int)
    ixs[:df] = np.ones(df, dtype=int)
    data11 = np.repeat(data1, ixs, axis=0)                    
    data1 = np.concatenate((data1, data11))

    return data1

def get_pos_neg_data(pos_class_dir, neg_class_dirs):

    orig_data = []
    if os.path.exists(pos_class_dir):
        pos_files = os.listdir(pos_class_dir)
        pos_fpaths = [pos_class_dir + f for f in pos_files]
        pos_data = load_training_data(pos_fpaths)

        neg_fpaths = []
        for nd in neg_class_dirs:

            if os.path.exists(nd) is False:
                continue

            neg_files = set(os.listdir(nd)) & set(pos_files)
            neg_fpaths += [nd + f for f in neg_files]

        neg_data = load_training_data(neg_fpaths)
        print('p:', pos_data.shape, 'n:', neg_data.shape)
        orig_data = np.concatenate((pos_data, neg_data))

        fx = neg_data.shape[0] / pos_data.shape[0]
        if fx > 1:
            pos_data = balance_data(pos_data, neg_data.shape[0], floor(fx))                    
        elif fx < 0:
            print("Repeating negative class data. Strange !!", 
                  pos_data.shape, neg_data.shape, pre_d)
            neg_data = balance_data(neg_data, pos_data.shape[0], floor(1/fx))
        else:
            pass

    else:
        print(pos_class_dir, "not found.")

    return pos_data, neg_data, orig_data


def grid_cv(X, y, k=5):
    """ Grid search over param space with k-fold cross validation """

    K = chi2_kernel(X)    

    pipeline = Pipeline([('clf', SVC(kernel='precomputed')), ])    

    params = {'clf__C': (1e-2, 1e-1, 1, 1e+1, 1e+2),}

    grid_search = GridSearchCV(pipeline, params, n_jobs=4,
                               verbose=0, cv=k)

    grid_search.fit(K, y)

    best_params = grid_search.best_estimator_.get_params()

    return best_params, grid_search.best_score_


def svm_with_cv(data, labels, scale_conf):
    """ SVM with chi2 kernel and 5 fold cross validation """
  
    best_params, best_cv_score = grid_cv(data, labels)
    print('CV:', best_cv_score, best_params['clf__C'])

    svm_clf = SVC(C=best_params['clf__C'], kernel='precomputed')
                   
    K = chi2_kernel(data)

    svm_clf = svm_clf.fit(K, labels)

    return svm_clf


def predict_features(train_data, test_data, svm_clf, scale_conf):

    K = chi2_kernel(test_data, train_data)

    # y_out has perpendicular distance between decision boundary and each point
    y_out = svm_clf.decision_function(K)    

    ix = SCALE_CONFIGS.index(scale_conf)
    a_l, b_l = AB_LS[ix]

    # convert the above distances into probabilities
    y_prob = sigmoid(y_out, a_l, b_l)

    return y_prob


def parallel_svm(label_lst):

    global cur_sd

    sd = cur_sd

    lab_dirs = os.listdir(TRAIN_DIR + sd)
    pre_d = TRAIN_DIR + sd + "/l_"

    for lab in label_lst:

        pos_class_dir = pre_d + str(lab) + "/"
        neg_class_dirs = [pre_d + str(-lab) + "/", 
                          pre_d + str(BGR_LABEL) + "/",
                          pre_d + str(-BGR_LABEL) + "/",
                          pre_d + "None/"]
            
        pos_data, neg_data, orig_data = get_pos_neg_data(pos_class_dir, 
                                                         neg_class_dirs)

        pos_labels = np.ones(pos_data.shape[0], dtype=int)
        neg_labels = pos_labels + 1

        data = np.concatenate((pos_data, neg_data))
        labels = np.concatenate((pos_labels, neg_labels))
            
        data, labels = shuffle_data_and_labels(data, labels)

        # print("data and labels:", data.shape, labels.shape)
        # print("orig_data:", orig_data.shape)
        # print("data and labels:", data.shape, labels.shape)

        svm_clf = svm_with_cv(data, labels, sd)
        pred = predict_features(data, orig_data, svm_clf, sd)

        # print(orig_data.shape, pred.shape)

        out_dir = SVM_MODEL_DIR + sd + "l_" + str(lab) + "/"
        os.system("mkdir -p " + out_dir)
            
        np.save(out_dir + "y_prob.npy", pred)
        pickle.dump(svm_clf, open(out_dir + 'svm_clf.pkl', 'wb'))


def main():
    """ main method """

    global cur_sd
    n_jobs = 5
    os.system("mkdir -p " + SVM_MODEL_DIR)

    scale_dirs = os.listdir(TRAIN_DIR)
    for sd in scale_dirs:

        cur_sd = sd
        n_labels = list(np.arange(1, N_LABELS + 1, dtype=int))
        chunks = chunkify(n_labels, n_jobs)
        pool = Pool(n_jobs)
        pool.map(parallel_svm, chunks)
        pool.close()
        pool.join()


if __name__ == "__main__":

    TRAIN_DIR = FEAT_DIR + "train/"
    SVM_MODEL_DIR = FEAT_DIR + "svm_models/"

    cur_sd = ''

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
