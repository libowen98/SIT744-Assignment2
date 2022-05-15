# -*- coding: utf-8 -*-


import numpy as np 
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
import glob
import cv2

path = r'./train'
IMG_SIZE = 32


def create_train_data(path):
    training_data = []
    if os.path.exists('label.map'):
        with open('label.map', 'r', encoding='utf8') as fi_label:
            folders = [_.strip() for _ in fi_label.readlines()]
    else:
        with open('label.map', 'w', encoding='utf8') as fo_label:
            folders = os.listdir(path)
            fo_label.write('\n'.join([str(_) for _ in folders]))

    for i in range(len(folders)):
        print(folders[i])
        nums = 0
        for j in glob.glob(path + "/" + folders[i] + "/*"):
            # print(j)
            img = cv2.imread(j)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), i])
            nums += 1
        print(nums)
    np.save('training_{}.npz'.format(IMG_SIZE), np.asarray(training_data))


create_train_data(path)
