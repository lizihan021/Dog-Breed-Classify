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
import pickle
import sys
from utils import *
from sift_feature import *
from cnn import *
from svm import *


# Initialize global variables
MODEL_WEIGHTS_FILE = get('cnn.weights_file')
VALIDATION_SPLIT = get('cnn.validation_split')
BATCH_SIZE = get('cnn.batch_size')
CNN_EPOCHS = get('cnn.cnn_epochs')
np.set_printoptions(edgeitems=30)

if not (len(sys.argv)>1 and sys.argv[1] == "svm"):
	# get cnn model
	model = generate_model()

	# define input data:
	dogs = DogsDataset()
	x_train, label_train, features_train = dogs.trainX, dogs.trainY, dogs.train_features
	# actually for test we don't know feature test.
	x_test, label_test, features_test_ground_truth = dogs.testX, dogs.testY, dogs.test_features

	if not exists(MODEL_WEIGHTS_FILE) or (len(sys.argv)>1 and sys.argv[1] == "train"):
		if len(sys.argv)>1 and sys.argv[1] == "train":
			model.load_weights(MODEL_WEIGHTS_FILE)
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

	print("loading cnn weight ...")
	model.load_weights(MODEL_WEIGHTS_FILE)

	# print("testing cnn performance ...")
	# score = model.evaluate(x_test, features_test_ground_truth, batch_size=BATCH_SIZE, verbose=0)
	# print('Test loss:', score[0], 'Test accuracy:', score[1])

	print("CNN predicting ...")
	features_test = model.predict(x_test, batch_size=BATCH_SIZE)

	# for i in range(10):
	# 	visualize_face(x_test[i], features_test[i])

	# get sift feature
	print("getting sift feature ...")
	new_features_train = get_new_feature(x_train, features_train)
	new_features_test = get_new_feature(x_test, features_test)
	# new_features_test = get_new_feature(x_test, features_test_ground_truth) # used to test svm

	pickle.dump( {"a":new_features_train,"b":label_train,"c":new_features_test,"d":label_test}, open( "save.p", "wb" ) )
# svm classify
print("svm training ...")

temp_p = pickle.load( open( "save.p", "rb" ) )
new_features_train,label_train,new_features_test,label_test = temp_p["a"],temp_p["b"],temp_p["c"],temp_p["d"]
# clf = SVC(kernel='linear',decision_function_shape="ovr", C=1.0, class_weight="balanced")
# clf = SVC(C=1.0, gamma=1e-8, class_weight="balanced").fit(new_features_train, label_train)
# clf = SVC(C=1.0, kernel='linear', class_weight='balanced')
# clf.fit(new_features_train, label_train)

print(label_test)
optimal_model = SVC(C=1, kernel='linear', class_weight='balanced').fit(new_features_train, label_train)
y_pred_b = optimal_model.predict(new_features_test)

print(y_pred_b)
print("final acc:", get_svm_acc(y_pred_b, label_test))

# for gam in [1e-7]:
# 	for c in [5, 10, 50, 100, 200]:
# 		clf = SVC(C=c, gamma=gam, class_weight="balanced").fit(new_features_train, label_train)
# 		y_pred_b = clf.predict(new_features_test)
# 		print(y_pred_b)
# 		print(c, gam)
# 		print("final acc:", get_svm_acc(y_pred_b, label_test))
for i in range(13):
	plt.subplot(1,2,1)
	visualize_feature_points(x_test[i], features_test[i], normalized=True)
	plt.subplot(1,2,2)
	visualize_feature_points(x_train[list(label_train).index(y_pred_b[i])], normalized=True)
	plt.title(":" + list(label_train).index(y_pred_b[i]))
	plt.show()

exit(0)