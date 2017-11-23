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

    def __init__(self, num_classes=10, training=True, _all=False):
        """
        Reads in the necessary data from disk and prepares data for training.
        """
        # Load in all the data we need from disk
        self.training_metadata = get('training_file')
        self.trainX, self.trainY = self._load_data('train')
        self.testing_metadata = get('testing_file')
        self.testX = self._load_data('test')

    def _load_data(self, partition='train'):
        """
        Loads a single data partition from file.
        """
        print("loading %s..." % partition)
        Y = None
        if partition == 'test':
            X = self._get_images(self.testing_metadata)
        else:
            X, Y = self._get_images_and_labels(self.training_metadata)

    def _get_images_and_labels(self, metadata):
        """
        Fetches the data based on image filenames specified in df.
        If training is true, also loads the labels.
        """
        X, y = [], []
        with open(metadata) as f:
            lines = f.read().splitlines()
        for line in lines:
            image = imread(os.path.join(get('image_path'), line))
            image = imresize(image,(get('image_dim'), get('image_dim')))
            X.append(image)

            input = re.split('/', line)
            input = re.split('.', input[0])
            y.append(input[1])
        return np.array(X), np.array(y).astype(int)

    def _get_images(self, metadata):
        X = []
        with open(metadata) as f:
            lines = f.read().splitlines()
        for line in lines:
            image = imread(os.path.join(get('image_path'), line))
            image = imresize(image,(get('image_dim'), get('image_dim')))
            X.append(image)
        return np.array(X)


if __name__ == '__main__':
    dogs = DogsDataset(num_classes=10, _all=True)
    print("Train:\t", len(dogs.trainX))
    print("Validation:\t", len(dogs.validX))
    print("Test:\t", len(dogs.testX))
