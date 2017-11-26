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
from utils import *
from sift_feature import *
from CNN import *

# Initialize global variables
MODEL_WEIGHTS_FILE = get('cnn.weights_file')
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

if not exists(MODEL_WEIGHTS_FILE):
	print("training ...")
	callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True)]
	history = model.fit(x_train, features_train,
	          batch_size=BATCH_SIZE,
	          epochs=CNN_EPOCHS,
	          verbose=1,
	          validation_split=VALIDATION_SPLIT,
	          callbacks=callbacks)

	max_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
	print('Min validation loss = {0:.4f} (epoch {1:d})'.format(max_val_loss, idx+1))

print("testing cnn performance ...")
model.load_weights(MODEL_WEIGHTS_FILE)
score = model.evaluate(x_test, features_test_ground_truth, batch_size=BATCH_SIZE, verbose=0)
print('Test loss:', score[0], 'Test accuracy:', score[1])

print("CNN predicting ...")
features_test = model.predict(x_test, batch_size=BATCH_SIZE)

# get sift feature
print("getting sift feature ...")
new_features_train = get_new_feature(x_train, features_train)
new_features_test = get_new_feature(x_test, features_test_ground_truth)

# svm classify
<<<<<<< HEAD
print("svm training ...")
clf = SVC(kernel='linear',decision_function_shape="ovr", C=1.0, class_weight="balanced")
=======
print("SVM training ...")
#clf = SVC(kernel='rbf',decision_function_shape="ovr", C=1.0, class_weight="balanced")
clf = SVC(C=1.0, kernel='linear', class_weight='balanced')
print(new_features_train, label_train)
>>>>>>> d9b0b3c7d53ecfa010e33744212f3ed46acc41ec
clf.fit(new_features_train, label_train)
y_pred_b = clf.predict(new_features_test)

print(y_pred_b)
print(label_test)
true_num = 0
for i, pred in enumerate(y_pred_b):
	if label_test[i] == pred:
		true_num += 1
print("final acc:", true_num/len(y_pred_b))
#visualize_feature_points(x_train[2], features_train[2], normalized=True)

exit(0)