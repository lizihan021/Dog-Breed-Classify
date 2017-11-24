"""
Dogs Dataset
    Class wrapper for interfacing with the dataset of dog images
    Usage:
        - from data.dogs import DogsDataset
        - python -m data.dogs
"""
import numpy as np
from scipy.misc import imread, imresize
import os
from utils import get, visualize_feature_points
import re

class DogsDataset:

    def __init__(self):
        # Load in all the data we need from disk

        self.train_path = get('testing_file')
        self.test_path = get('training_file')
        self.trainX, self.trainY, self.train_features = self._load_data('train')
        self.testX, self.testY, self.test_features = self._load_data('test')

    def _load_data(self, partition='train'):
        """
        Loads a single data partition from file.
        """
        if partition == 'train':
            path = self.train_path
        else:
            path = self.test_path
        print("loading %s..." % partition)
        X, y = [], []
        point_location_vector = []
        lines = open(path).read().splitlines()

        for line in lines[:10]:
            image = imread(os.path.join(get('image_path'), line))
            row, col, _ = image.shape
            image = imresize(image,(get('image_dim'), get('image_dim')))

            # image #
            X.append(image)

            # feature vector #
            filename = line.rpartition('.')[0] + '.txt'
            f = open(get('point_location_path') + '/' + filename, 'r')
            one_feature = np.zeros(16)
            points = f.readlines()
            for i, point in enumerate(points):
                point_x, point_y = point.split()
                # I don't know why x is col and y is row...
                # but it works (maybe because image coordinates?)
                one_feature[2*i] = int(point_x)*127/(col-1)
                one_feature[2*i+1] = int(point_y)*127/(row-1)
            point_location_vector.append(one_feature)

            # dog class #
            y.append(re.split('.', re.split('/', line)[0])[0])

        X = self._normalize(np.array(X), partition)
        # example of visualization
        visualize_feature_points(X[3], point_location_vector[3], normalized=True)
        return np.array(X), np.array(y), point_location_vector

    def _normalize(self, X, is_train):
        # this will normalize the data:
        if is_train == "train":
            self.image_mean = np.mean(X, axis=(0,1,2))
            self.image_std = np.std(X, axis=(0,1,2))
        return (X - self.image_mean) / self.image_std


if __name__ == '__main__':
    dogs = DogsDataset()
    print("Train:\t", len(dogs.trainX))
    print("Test:\t", len(dogs.testX))
