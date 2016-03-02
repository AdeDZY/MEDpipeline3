#!/usr/bin/env python
__author__ = 'zhuyund'

import numpy
import os
import cPickle
import sys
import argparse
from os import listdir
from os.path import isfile, join


def avg_feats(feat_file_path):
    """
    Generate the average of features
    :param feat_file_path
    :return: vec
    """
    vec = []
    n = 0
    for line in open(feat_file_path):
        vals = line.split(';')
        for i, v in enumerate(vals):
            v = float(v)
            if len(vec) <= i:
                vec.append(v)
            else:
                vec[i] += v
        n += 1

    for i in range(len(vec)):
        vec[i] /= n

    return vec


def max_feats(feat_file_path):
    """
    Generate the max of features
    :param feat_file_path
    :return: vec
    """
    vec = []
    for line in open(feat_file_path):
        vals = line.split(';')
        for i, v in enumerate(vals):
            v = float(v)
            if len(vec) <= i:
                vec.append(v)
            else:
                vec[i] = max(vec[i], v)

    return vec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir", help="the dir of video features")
    parser.add_argument("list_file", help="a list of all videos")
    parser.add_argument("output_file_path", help="the output file")
    parser.add_argument("--method", '-m', help="pooling method", choices=["avg", "max"], default='avg')
    args = parser.parse_args()

    # output file
    output_file = open(args.output_file_path, 'w')

    # get all feature files
    vid2feat = {}
    for f in listdir(args.feat_dir):
        if not isfile(join(args.feat_dir, f)) and f.startswith('HVC'):
            continue
        video_name = f.split('.')[0]
        vid2feat[video_name] = join(args.feat_dir, f)

    # process each video
    for vid in open(args.list_file):
        vid = vid.strip()
        if vid not in vid2feat:
            print ">> Feature for {0} does not exist! use -1.".format(vid)
            output_str = vid + '\t-1\n'
            output_file.write(output_str)
            continue

        if args.method == 'avg':
            vec = avg_feats(vid2feat[vid])
        else:
            vec = max_feats(vid2feat[vid])

        output_str = ';'.join([str(t) for t in vec])
        output_file.write(vid + '\t')
        output_file.write(output_str + '\n')

    print "avg features generated successfully! Written into {0}!".format(args.output_file_path)


if __name__ == '__main__':
    main()

