#!/usr/bin/env python
import numpy as np
import sys
import caffe
import os
from os import listdir
from os.path import isfile, join
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()

    caffe.set_mode_cpu()

    # load model
    caffe_root = "/home/ubuntu/caffe/"
    model_name = "VGG_ILSVRC_19_layers"
    net = caffe.Net('{0}/models/{1}/{1}_deploy.prototxt'.format(caffe_root, model_name),
                    '{0}/models/{1}/{1}.caffemodel'.format(caffe_root, model_name),
                    caffe.TEST)

    # configure input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    # process each img
    imgs = [f for f in listdir("/home/ubuntu/hw3/keyframes2/") if isfile(join("/home/ubuntu/hw3/keyframes2/", f))]
    batch_size = 10
    i = 7450 
    while i < len(imgs):
        img_names = []
        n = batch_size
        for j in range(batch_size):
            if i + j >= len(imgs):
                n = j
                break
            img = imgs[i + j]
            img_path = join("/home/ubuntu/hw3/keyframes2/", img)
            img_names.append(img.split('.')[0])
            net.blobs['data'].data[j] = transformer.preprocess('data', caffe.io.load_image(img_path))
        net.forward()
        for j in range(n):
            feat = net.blobs['fc7'].data[j].flat
            fout = open(args.output_dir + '/{0}.feat'.format(img_names[j]), 'w')
            fout.write(';'.join([str(v) for v in feat]))
            fout.close()
        i += batch_size


if __name__ == "__main__":
    main()
