#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh, Srikanth
# Date created : 19 Oct 2016
# Last modified : 19 Oct 2016

"""
SVM with chi^2 kernel
"""

import os
import argparse
import pickle
import numpy as np

from math import floor

from multiprocessing import cpu_count, Pool

from sklearn.base import TransformerMixin
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import chi2_kernel

from crf_utils import FEAT_DIR, BGR_LABEL, N_LABELS
from crf_utils import shuffle_data_and_labels

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
        return chi2_kernel(X, gamma=self.gamma)

    def transform(self, X):
        return chi2_kernel(X, gamma=self.gamma)

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
    """ Load data from files """
    return np.concatenate([np.load(f) for f in input_fpaths])


def balance_data(data1, data2_size, fx):
    """ Balance data by repeating them """
    if fx >= 2:
        data1 = np.repeat(data1, fx, axis=0)

    df = data2_size - data1.shape[0]
    ixs = np.zeros(data1.shape[0], dtype=int)
    ixs[:df] = np.ones(df, dtype=int)
    data11 = np.repeat(data1, ixs, axis=0)
    data1 = np.concatenate((data1, data11))

    return data1


def get_pos_neg_data(pos_class_dir, neg_class_dirs):
    """ Get data from positive and negative classes """

    orig_data = []
    orig_labels = []
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
        orig_data = np.concatenate((pos_data, neg_data))
        orig_labels = np.concatenate(([1] * pos_data.shape[0],
                                      [2] * neg_data.shape[0]))

        fx = neg_data.shape[0] / pos_data.shape[0]
        if fx > 1:
            pos_data = balance_data(pos_data, neg_data.shape[0], floor(fx))
        elif fx < 0:
            print("Repeating negative class data. Strange !!",
                  pos_data.shape, neg_data.shape)
            neg_data = balance_data(neg_data, pos_data.shape[0], floor(1/fx))
        else:
            pass

    else:
        print(pos_class_dir, "not found.")

    return pos_data, neg_data, orig_data, orig_labels


def grid_cv(X, y, k=5):
    """ Grid search over param space with k-fold cross validation """

    K = chi2_kernel(X)

    pipeline = Pipeline([('clf', SVC(kernel='precomputed')), ])

    params = {'clf__C': (1e-2, 1e-1, 1, 1e+1, 1e+2), }

    grid_search = GridSearchCV(pipeline, params, n_jobs=1,
                               verbose=0, cv=k)

    grid_search.fit(K, y)

    best_params = grid_search.best_estimator_.get_params()

    return best_params, grid_search.best_score_


def svm_with_cv(data, labels):
    """ SVM with chi2 kernel and 5 fold cross validation """

    best_params, best_cv_score = grid_cv(data, labels)
    if ARGS.verbose:
        print('CV:', best_cv_score, best_params['clf__C'])

    svm_clf = SVC(C=best_params['clf__C'], kernel='precomputed')

    gram_matrix = chi2_kernel(data)

    svm_clf = svm_clf.fit(gram_matrix, labels)

    # Train a logistic regression to convert the output of
    # SVM into probabilities
    out = svm_clf.decision_function(gram_matrix)
    out = out.reshape(-1, 1)

    # print('out:', out.shape, 'labels:', labels.shape)

    lr_clf = LogisticRegression()
    lr_clf.fit(out, labels)

    if ARGS.verbose:
        lr_pred = lr_clf.predict(out)
        print("LR:", np.mean(labels == lr_pred))

    return svm_clf, lr_clf


def predict_features(train_data, test_data, test_labels, clfs):
    """ Predict class probabilites for a given set of test features.
    SVM followed by logistic regression """

    svm_clf, lr_clf = clfs

    gram_matrix = chi2_kernel(test_data, train_data)

    # y_out has perpendicular distances between decision boundary
    # and each point
    y_out = svm_clf.decision_function(gram_matrix)
    y_out = y_out.reshape(-1, 1)

    # convert the above distances into probabilities
    y_prob = lr_clf.predict_proba(y_out)

    if ARGS.verbose:
        y_pred = lr_clf.predict(y_out)
        print("LR test acc:", np.mean(y_pred == test_labels))

    return y_prob


def parallel_svm(label_lst):
    """ Train SVMs parallely """

    # lab_dirs = os.listdir(TRAIN_DIR + sd)
    pre_d = TRAIN_DIR + SCALE_DIR + "/l_"

    for lab in label_lst:

        out_dir = SVM_MODEL_DIR + SCALE_DIR + "/l_" + str(lab) + "/"
        os.system("mkdir -p " + out_dir)

        if os.path.exists(out_dir + 'y_prob.npy'):
            print(out_dir + 'y_prob.npy exists. Skipping..')
            continue

        print("SVM:", out_dir)

        pos_class_dir = pre_d + str(lab) + "/"
        neg_class_dirs = [pre_d + str(-lab) + "/",
                          pre_d + str(BGR_LABEL) + "/",
                          pre_d + str(-BGR_LABEL) + "/",
                          pre_d + "None/"]

        p_data, n_data, o_data, o_labels = get_pos_neg_data(pos_class_dir,
                                                            neg_class_dirs)

        data = np.concatenate((p_data, n_data))
        p_labels = np.ones(p_data.shape[0], dtype=int)
        n_labels = p_labels + 1
        labels = np.concatenate((p_labels, n_labels))

        data, labels = shuffle_data_and_labels(data, labels)

        print("data and labels:", data.shape, labels.shape, data.dtype)
        print("orig_data:", o_data.shape, o_labels.shape)
        # print("data and labels:", data.shape, labels.shape)

        svm_clf, lr_clf = svm_with_cv(data, labels)
        pred = predict_features(data, o_data, o_labels, (svm_clf, lr_clf))

        # print(orig_data.shape, pred.shape)

        np.save(out_dir + "y_prob.npy", pred)
        pickle.dump(svm_clf, open(out_dir + 'svm_clf.pkl', 'wb'))
        pickle.dump(lr_clf, open(out_dir + 'lr_clf.pkl', 'wb'))

        # break


def main():
    """ main method """

    global SCALE_DIR

    n_jobs = 7
    os.system("mkdir -p " + SVM_MODEL_DIR)

    scale_dirs = os.listdir(TRAIN_DIR)
    for sdir in scale_dirs:

        SCALE_DIR = sdir

        n_labels = list(np.arange(1, N_LABELS + 1, dtype=int))

        """
        chunks = chunkify(n_labels, n_jobs)
        pool = Pool(n_jobs)
        pool.map(parallel_svm, chunks)
        pool.close()
        pool.join()
        """
        parallel_svm(n_labels)
        break


if __name__ == "__main__":

    SCALE_DIR = ''
    TRAIN_DIR = FEAT_DIR + "train/"
    SVM_MODEL_DIR = FEAT_DIR + "svm_models/"

    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument("--verbose", "-v", action="store_true")
    ARGS = PARSER.parse_args()

    main()
