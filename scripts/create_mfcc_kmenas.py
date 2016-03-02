#!/usr/bin/env python
__author__ = 'zhuyund'

import numpy
import cPickle
from sklearn.cluster import KMeans
import argparse
import sys, os


def transform_video(km, cluster_num, mfcc):
    """
    transform a video's MFCC features into one bag-of-word vector
    :param km: kmeans model
    :param cluster_num: int. the number of clusters
    :param mfcc: MFCC features for the video. shape=(n_samples, n_features)
    :return: a feature vector for this video. shape=(1, cluster_num)
    """
    labels = km.predict(mfcc)
    v = [0 for i in range(cluster_num)]
    for label in labels:
        v[label] += 1
    return v


def load_mfcc(mfcc_csv_file):
    """
    load sampled MFCC features into a matrix X
    :param mfcc_csv_file: path to the mfcc csv file
    :return: X. shape=(n_samples, n_features)
    """
    X = []
    i = 0
    for line in open(mfcc_csv_file):
        i += 1
        if i % 10 != 0:
            continue
        x = [float(val) for val in line.split(';')]
        X.append(x)
    return X


# Generate k-means features for videos; each video is represented by a single vector
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kmeans_model", help="path to the kmeans model")
    parser.add_argument("cluster_num", type=int, help="number of cluster")
    parser.add_argument("file_list", help="the list of videos")
    parser.add_argument("--output_file_path", "-o", type=str,
                        default="/home/ubuntu/hw1/kmeans/all.vectors")
    args = parser.parse_args()

    # open output file
    output_file = open(args.output_file_path, 'w')

    # load the kmeans model
    km = cPickle.load(open(args.kmeans_model, "rb"))

    # process each video
    n = 0
    for video_name in open(args.file_list):
        video_name = video_name.strip()

        # load MFCC features
        mfcc_path = "/home/ubuntu/hw3/mfcc/{}.mfcc.csv".format(video_name.strip())

        if not os.path.exists(mfcc_path):
            print "{}'s MFCC features not exist! Write vector as -1".format(video_name)
            output_file.write(video_name + "\t-1\n")
            n += 1
            continue

        mfcc = load_mfcc(mfcc_path)

        # transform
        v = transform_video(km, args.cluster_num, mfcc)

        # write new feature
        output_str = ';'.join([str(t) for t in v])
        output_file.write(video_name + '\t')
        output_file.write(output_str + '\n')

        # output process
        n += 1
        if n % 50 == 0:
            print "{0} videos processed.".format(n)

    print "K-means features generated successfully! Featues are written into {0}!".format(args.output_file_path)


if __name__ == '__main__':
    main()


