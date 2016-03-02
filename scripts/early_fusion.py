#!/usr/bin/env python
__author__ = 'zhuyund'

import numpy
import os
from sklearn import svm
import cPickle
import sys
import argparse


def get_feat(output_file):
    f1 = open("/home/ubuntu/hw3/siftbow_features/all_avg.vectors")
    f2 = open("/home/ubuntu/hw3/cnn_fc7_features/all_avg.vectors")
    f3 = open("/home/ubuntu/hw3/mfcc_bow_pred/P00${i}_${k}.model")
    fo = open(output_file, 'w')
    while True:
        line1 = f1.readline().strip()
        line2 = f2.readline().strip()
        line3 = f3.readline().strip()
        if not line1:
            break

        vid, feat1 = line1.split('\t')
        vid, feat2 = line2.split('\t')
        vid, feat3 = line3.split('\t')

        if feat1 == '-1':
            feat1 = ';'.join(['0' for i in range(200)])
        if feat2 == '-1':
            feat2 = ';'.join(['0' for i in range(4096)])
        if feat3 == '-1':
            feat3 = ';'.join(['0' for i in range(200)])

        fo.write(vid + '\t')
        fo.write(feat1)
        fo.write(';')
        fo.write(feat2)
        fo.write(';')
        fo.write(feat3)
        fo.write('\n')

    fo.close()
    f1.close()
    f2.close()
    f3.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file")
    args = parser.parse_args()

    get_feat(args.output_file)


if __name__ == '__main__':
    main()
