import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from dog import DogsDataset
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import optunity
import optunity.metrics
from utils import *
from sift_feature import *
from cnn_comp import *


# Initialize global variables
MODEL_WEIGHTS_FILE = get('cnn.weights_file_comp')
VALIDATION_SPLIT = get('cnn.validation_split')
BATCH_SIZE = get('cnn.batch_size')
CNN_EPOCHS = get('cnn.cnn_epochs')

# get cnn model
model = generate_model()

# define input data:
dogs = DogsDataset()
x_train, label_train, features_train = dogs.trainX, dogs.trainY, dogs.train_features
# actually for test we don't know feature test.
x_test, label_test, features_test_ground_truth = dogs.testX, dogs.testY, dogs.test_features

label_train_one = label_binarize(label_train, classes=range(1,134))
label_test_one = label_binarize(label_test, classes=range(1,134))

if not exists(MODEL_WEIGHTS_FILE):
	print("training ...")
	callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True)]
	history = model.fit(x_train, label_train_one,
	          batch_size=BATCH_SIZE,
	          epochs=CNN_EPOCHS,
	          verbose=1,
	          validation_split=VALIDATION_SPLIT,
	          callbacks=callbacks)

	max_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
	print('Min validation loss = {0:.4f} (epoch {1:d})'.format(max_val_loss, idx+1))

print("loading cnn weight ...")
model.load_weights(MODEL_WEIGHTS_FILE)

print("testing cnn performance ...")
score = model.evaluate(x_test, label_test_one, batch_size=BATCH_SIZE, verbose=0)
print('Test loss:', score[0], 'Test accuracy:', score[1])

# print("CNN predicting ...")
# features_test = model.predict(x_test, batch_size=BATCH_SIZE)









