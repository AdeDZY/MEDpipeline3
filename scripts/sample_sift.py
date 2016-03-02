#!/usr/bin/env python
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import random

def main():
    sift_dir = "/home/ubuntu/hw3/sift_features/"
    files = [f for f in listdir(sift_dir) if isfile(join(sift_dir, f))]
    fout = open("select.sift", 'w')
    for f in files:
        for line in open(join(sift_dir, f)):
            r = random.random()
            if r < 0.05:
                fout.write(line)

    print ">> sampled SIFT descriptors written to select.sift!"

if __name__ == '__main__':
    main()
