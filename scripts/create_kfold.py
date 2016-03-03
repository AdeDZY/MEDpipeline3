#!/usr/bin/env python
from sklearn.cross_validation import KFold
import numpy as np

def main():
    videos = []
    lables = []
    with open("/home/ubuntu/hw3/list/train_dev") as f:
        for line in f:
            v, l = line.strip().split()
            videos.append(v)
            lables.append(l)

    kf = KFold(n=len(videos), n_folds=3, shuffle=True, random_state=0)
    i = 0
    for train_idxes, test_idxes in kf:
        fo = open("/home/ubuntu/hw3/list/train_{0}.video".format(i + 1), 'w') # write videos
        for idx in train_idxes:
            fo.write(videos[idx] + '\n')
        fo.close()

        fo = open("/home/ubuntu/hw3/list/test_{0}.video".format(i + 1), 'w') # write videos
        for idx in test_idxes:
            fo.write(videos[idx] + '\n')
        fo.close()
        
        for p in range(3):
            fo = open("/home/ubuntu/hw3/list/P00{1}_train_{0}".format(i + 1, p + 1), 'w')
            for idx in train_idxes:
                if lables[idx] == "P00{0}".format(p + 1):
                    fo.write(videos[idx] + ' ' + lables[idx] + '\n')
                else:
                    fo.write(videos[idx] + ' NULL\n')
            fo.close()

        for p in range(3):
            fo = open("/home/ubuntu/hw3/list/P00{1}_test_{0}".format(i + 1, p + 1), 'w')
            for idx in test_idxes:
                if lables[idx] == "P00{0}".format(p + 1):
                    fo.write('1\n')
                else:
                    fo.write('0\n')
            fo.close()

        for p in range(3):
            fo = open("/home/ubuntu/hw3/list/P00{1}_train".format(i + 1, p + 1), 'w')
            for idx in range(len(videos)):
                if lables[idx] == "P00{0}".format(p + 1):
                    fo.write(videos[idx] + ' ' + lables[idx] + '\n')
                else:
                    fo.write(videos[idx] + ' NULL\n')
            fo.close()
        i += 1


if __name__ == '__main__':
    main()
