"""
EECS 445 - Introduction to Machine Learning
Fall 2017 - Project 2
Utility functions
"""
import os
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt

def get(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(get, 'config'):
        with open('config.json') as f:
            get.config = eval(f.read())
    node = get.config
    for part in attr.split('.'):
        node = node[part]
    return node

def visualize_feature_points(image, features):
    """
    Plot feature points on the image, including
    Right eye, Left eye, Nose, Right ear tip, Right ear base (inner base),
    Head top, Left ear base (inner base), Left ear tip.
    """
    # im = plt.imread(image_name)
    implot = plt.imshow(image)
    plt.scatter(x=features[::2], y=features[1::2], c='r', s=100)
    plt.show()

# if __name__ == '__main__':
#     image_name = "CU_Dogs/dogImages/006.American_eskimo_dog/American_eskimo_dog_00394.jpg"
#     features = np.array([231,77,273,76,249,103,201,3,229,36,253,32,279,32,299,1])

#     visualize_feature_points(image_name, features)
