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

