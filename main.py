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
from cnn import *


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

print("loading cnn weight ...")
model.load_weights(MODEL_WEIGHTS_FILE)

print("testing cnn performance ...")
score = model.evaluate(x_test, features_test_ground_truth, batch_size=BATCH_SIZE, verbose=0)
print('Test loss:', score[0], 'Test accuracy:', score[1])

print("CNN predicting ...")
features_test = model.predict(x_test, batch_size=BATCH_SIZE)

for i in range(10):
	visualize_face(x_test[i], features_test[i])

# get sift feature
print("getting sift feature ...")
new_features_train = get_new_feature(x_train, features_train)
new_features_test = get_new_feature(x_test, features_test)
# new_features_test = get_new_feature(x_test, features_test_ground_truth) # used to test svm

# svm classify
print("svm training ...")
# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=new_features_train, y=label_train, num_folds=10, num_iter=2)
def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = sklearn.svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])
optimal_model = SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(new_features_train, label_train)
# clf = SVC(kernel='linear',decision_function_shape="ovr", C=1.0, class_weight="balanced")
# clf = SVC(kernel='rbf',decision_function_shape="ovr", C=1.0, class_weight="balanced")
# clf = SVC(C=1.0, kernel='linear', class_weight='balanced')
# clf.fit(new_features_train, label_train)

y_pred_b = optimal_model.predict(new_features_test)

np.set_printoptions(edgeitems=30)
print(y_pred_b)
print(label_test)
true_num = 0
for i, pred in enumerate(y_pred_b):
	if label_test[i] == pred:
		true_num += 1
print("final acc:", float(true_num)/len(y_pred_b))

#visualize_feature_points(x_train[2], features_train[2], normalized=True)

exit(0)