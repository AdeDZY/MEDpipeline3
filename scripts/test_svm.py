#!/usr/bin/env python
__author__ = 'zhuyund'

import numpy
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
import argparse
from sklearn.preprocessing import StandardScaler


def load_imtraj_test_data(fold):
    if fold != 0:
        test_list = open("/home/ubuntu/hw3/list/test_{0}.video".format(fold))
    else:
        test_list = open("/home/ubuntu/hw3/list/test.video")
    X = []
    y = []
    for line in test_list:
        video = line.strip()
        x = [0 for i in range(32768)]
        try:
            with open("/home/ubuntu/hw3/imtraj/{0}.spbof".format(video)) as f:
                line = f.readline()
                items = line.split()
                for item in items:
                    idx, v = item.split(':')
                    x[int(idx) - 1] = float(v)
        except IOError:
            print ">> {0}'s imtraj feature does not exist!".format(video)

        X.append(x)

    return X


def load_asrbof_test_data(fold):
    if fold != 0:
        test_list = open("/home/ubuntu/hw3/list/test_{0}.video".format(fold))
    else:
        test_list = open("/home/ubuntu/hw3/list/test.video")
    X = []
    y = []
    for line in test_list:
        video = line.strip()
        x = [0 for i in range(12760)]
        try:
            with open("/home/ubuntu/hw3/asr_bof/{0}.bof".format(video)) as f:
                line = f.readline()
                items = line.split()
                for item in items:
                    idx, v = item.split(':')
                    if idx > 12760:
                        continue
                    x[int(idx) - 1] = float(v)
        except IOError:
            print ">> {0}'s asr_bof feature does not exist!".format(video)

        X.append(x)

    return X

def load_test_data(feat_file_path, feat_dim, fold):
    """
    Load all test data
    :param feat_file_path: the file that contains all features.
    Each line represents a video. Line starts with video_name, than a '\t', than the feature vector
    :return: X, the test feature vectors. shape=(n_sample, n_feat)
    """
    test_list = open("/home/ubuntu/hw3/list/test_{0}.video".format(fold))
    videos = {}
    for line in test_list:
        video = line.strip()
        videos[video] = 0
    test_list.close()

    X = []
    for line in open(feat_file_path):
        line = line.strip()
        video, feats = line.split('\t')
        if video not in videos:
            continue
        if feats == '-1':
            X.append([0 for i in range(feat_dim)])
            continue

        x = [float(t) for t in feats.split(';')]
        X.append(x)

    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="path of the trained svm file")
    parser.add_argument("fold", help="fold 1,2,3, 0==all", type=int)
    parser.add_argument("feat_file", help="feature file")
    parser.add_argument("feat_dim", type=int, help="dimension of features")
    parser.add_argument("output_file", help="path to save the prediction score")
    parser.add_argument("--feat_type", "-f", choices=["sift", "imtraj", "cnn", "asr"], default="sift")
    args = parser.parse_args()

    # load model
    clf, scaler = cPickle.load(open(args.model_file, 'rb'))
    print clf.get_params()

    # load data
    if args.feat_type == 'imtraj':
        X = load_imtraj_test_data(args.fold)
    elif args.feat_type == 'asr':
        X = load_asrbof_test_data(args.fold)
    else:
        X = load_test_data(args.feat_file, args.feat_dim, args.fold)


    # predict with the log probability
    print ">> Predicting..."
    X = scaler.fit_transform(X)
    T = clf.decision_function(X)

    # write results
    outfile = open(args.output_file, 'w')
    for score in T:
        outfile.write(str(score) + '\n')
    outfile.close()
    print ">> Prediction scores written to {0}!".format(args.output_file)

# Apply the SVM model to the testing videos; Output the score for each video
if __name__ == '__main__':
    main()

