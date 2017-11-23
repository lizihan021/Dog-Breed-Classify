"""
Dogs Dataset
    Class wrapper for interfacing with the dataset of dog images
    Usage:
        - from data.dogs import DogsDataset
        - python -m data.dogs
"""
import numpy as np
import pandas as pd
from scipy.misc import imread, imresize
import os
from utils import get
import re

class DogsDataset:

    def __init__(self):
        # Load in all the data we need from disk

        self.test_metadata = get('testing_file')
        self.train_metadata = get('training_file')
        self.trainX, self.trainY = self._load_data('train')
        self.testX, self.testY = self._load_data('test')

    def _load_data(self, partition='train'):
        """
        Loads a single data partition from file.
        """
        if partition == 'train':
            metadata = self.train_metadata
        else:
            metadata = self.test_metadata
        print("loading %s..." % partition)
        X, y = [], []
        point_location_vector = []
        with open(metadata) as f:
            lines = f.read().splitlines()
        new_pos = np.asarray([[128-1], [128-1]])
        for line in lines[:10]:
            image = imread(os.path.join(get('image_path'), line))
            row, col, _ = image.shape
            image = imresize(image,(get('image_dim'), get('image_dim')))
            # image
            X.append(image)
            # feature vector
            filename = line.rpartition('.')[0] + '.txt'
            f = open(get('point_location_path') + '/' + filename, 'r')
            one_feature = np.zeros((2,8))
            points = f.readlines()
            for i in range(len(points)):
                point_x, point_y = points[i].split()
                point_x = int(point_x)
                point_y = int(point_y)
                one_feature[0, i] = point_x
                one_feature[1, i] = point_y
            old_pos = np.asarray([[row - 1], [col - 1]])
            trans_matrix = np.dot(new_pos, np.linalg.pinv(old_pos))
            one_feature = np.dot(trans_matrix, one_feature)
            one_feature = np.reshape(one_feature, 16, order='F')
            point_location_vector.append(one_feature)
            # dog class
            y.append(re.split('.', re.split('/', line)[0])[0])
        return np.array(X), np.array(y), np.array(point_location_vector)
        

if __name__ == '__main__':
    dogs = DogsDataset()
    print("Train:\t", len(dogs.trainX))
    print("Test:\t", len(dogs.testX))
