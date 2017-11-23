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
        with open(metadata) as f:
            lines = f.read().splitlines()
        for line in lines[:10]:
            image = imread(os.path.join(get('image_path'), line))
            image = imresize(image,(get('image_dim'), get('image_dim')))
            # image
            X.append(image)
            # dog class
            y.append(re.split('.', re.split('/', line)[0])[0])
        return np.array(X), np.array(y)
