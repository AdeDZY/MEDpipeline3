#!/usr/bin/env python
__author__ = 'zhuyund'

import numpy
import cPickle
from sklearn.cluster import KMeans
import argparse
import sys, os
from os import listdir
from os.path import isfile, join


def transform_feats(km, cluster_num, feats):
    """
    transform a video's features into one bag-of-word vector
    :param km: kmeans model
    :param cluster_num: int. the number of clusters
    :param feats: features for the video. shape=(n_samples, n_features)
    :return: a feature vector for this video. shape=(1, cluster_num)
    """
    labels = km.predict(feats)
    v = [0 for i in range(cluster_num)]
    for label in labels:
        v[label] += 1
    return v


def load_feats(feat_csv_file):
    """
    load features into a matrix X
    :param feat_csv_file: path to the mfcc csv file
    :return: X. shape=(n_samples, n_features)
    """
    X = []
    for line in open(feat_csv_file):
        line = line.strip()
        x = [float(val) for val in line.split(';') if val]
        X.append(x)
    return X


# Generate k-means features for videos; each video is represented by a single vector
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_csv_dir", help="dir of all feature csv files")
    parser.add_argument("kmeans_model", help="path to the kmeans model")
    parser.add_argument("cluster_num", type=int, help="number of cluster")
    args = parser.parse_args()

    # open output file
    output_dir = "/home/ubuntu/hw3/siftbow_features/"

    # load the kmeans model
    km = cPickle.load(open(args.kmeans_model, "rb"))

    # get all keyframe SIFT feature file names
    feat_files = [f for f in listdir(args.feat_csv_dir) if isfile(join(args.feat_csv_dir, f))]
    vid2filepath = {}
    for f in feat_files:
        video_name = f.split('_')[0]
        if video_name not in vid2filepath:
            vid2filepath[video_name] = []
        vid2filepath[video_name].append(join(args.feat_csv_dir, f))

    # process each video's keyframes
    # each video will have a result file
    # each line in the file represents a keyframe
    n = 0
    for video_name in vid2filepath:

        output_file = open(join(output_dir, video_name + ".siftbow"), 'w')

        for frame_file in vid2filepath[video_name]:
            feats = load_feats(frame_file)
            if not feats:
                continue

            # transform
            v = transform_feats(km, args.cluster_num, feats)

            # write new feature
            output_str = ';'.join([str(t) for t in v])
            # output_file.write(frame_name + '\t')
            output_file.write(output_str + '\n')

        output_file.close()
        # print process
        n += 1
        if n % 50 == 0:
            print "{0} videos processed.".format(n)

    print "K-means features generated successfully! Featues are written into {0}!".format(output_dir)


if __name__ == '__main__':
    main()


