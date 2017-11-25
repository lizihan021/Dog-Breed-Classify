import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
from utils import *
from dog import DogsDataset
from os.path import exists
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Initialize global variables
IMAGE_SIZE = get('image_dim')
DROPOUT_CNN = get('cnn.dropout_cnn')
DROPOUT_DENSE = get('cnn.dropout_dense')
BATCH_SIZE = get('cnn.batch_size')
OPTIMIZER = get('cnn.optimizer')
MODEL_WEIGHTS_FILE = get('cnn.weights_file')
VALIDATION_SPLIT = get('cnn.validation_split')
CNN_EPOCHS = get('cnn.cnn_epochs')

# define CNN
model = Sequential()
# (128,128,3)
model.add(Conv2D(16, kernel_size=(7, 7),
                 activation='relu',
                 input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
# (122,122,16)
model.add(Conv2D(32, (5, 5), activation='relu'))
# (118,118,32)
model.add(MaxPooling2D(pool_size=(2, 2)))
# (59,59,32)
model.add(Dropout(DROPOUT_CNN))
# (59,59,32)
model.add(Conv2D(64, (5, 5), activation='relu'))
# (55,55,64)
model.add(Conv2D(64, (3, 3), activation='relu'))
# (53,53,64)
model.add(MaxPooling2D(pool_size=(2, 2)))
# (26,26,64)
model.add(Dropout(DROPOUT_CNN))
# (26,26,64)
model.add(Conv2D(256, (3, 3), activation='relu'))
# (24,24,256)
model.add(Conv2D(256, (3, 3), activation='relu'))
# (22,22,256)
model.add(MaxPooling2D(pool_size=(2, 2)))
# (11,11,256)
model.add(Dropout(DROPOUT_CNN))
# (11,11,256)
model.add(Flatten())
# 30976
model.add(Dense(1250, activation='relu'))
# 1250
model.add(Dropout(DROPOUT_DENSE))
# 1250
model.add(Dense(1000, activation='relu'))
# 1000
model.add(Dropout(DROPOUT_DENSE))
# 1000
model.add(Dense(16, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

plot_model(model, to_file='model.png')

# define input data:
dogs = DogsDataset()
x_train, label_train, features_train = dogs.trainX, dogs.trainY, dogs.train_features
x_test, lable_test, features_test = dogs.testX, dogs.testY, dogs.test_features

if not exists(MODEL_WEIGHTS_FILE):
	callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True)]
	history = model.fit(x_train, features_train,
	          batch_size=BATCH_SIZE,
	          epochs=CNN_EPOCHS,
	          verbose=2,
	          validation_split=VALIDATION_SPLIT,
	          callbacks=callbacks)

	max_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
	print('Min validation loss = {0:.4f} (epoch {1:d})'.format(max_val_loss, idx+1))

model.load_weights(MODEL_WEIGHTS_FILE)
score = model.evaluate(x_test, features_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

def get_sift(img, kp = [], mode = "gray", mask = None):
	img = denormalize_image(img)
	img = np.array(img*255, dtype="uint8")
	if mode == "gray":
		gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		kp,des = sift.compute(gray,kp)
		return des.flatten()
		# img=cv2.drawKeypoints(gray,kp, None)
		# cv2.imshow('result', img), cv2.waitKey(0)
		# cv2.destroyWindow("result")
	elif mode == "color":
		# image chnnel mask
		hist1 = cv2.calcHist([img],[0],mask,[256],[0,256]).T[0]
		hist2 = cv2.calcHist([img],[1],mask,[256],[0,256]).T[0]
		hist3 = cv2.calcHist([img],[2],mask,[256],[0,256]).T[0]
		# plt.hist(gray_img.ravel(),256,[0,256])
		# plt.title('Histogram for gray scale picture')
		# plt.show()
		return np.append(np.append(hist1, hist2), hist3)

def generate_kp(features_line):
	kp = []
	for i in range(int(len(features_line)/2)):
		tmp_x = int(features_line[2*i])
		tmp_y = int(features_line[2*i+1])
		kp.append( cv2.KeyPoint(tmp_x, tmp_y, 16) ) ###
	return kp, None

def get_new_feature(X, feature):
	new_features = []
	for x_row, f_row in zip(X, feature):
		kp, mask = generate_kp(f_row)
		sift_vec = get_sift(x_row, kp = kp)
		hist_vec = get_sift(x_row, mode = "color", mask = mask)
		# concatnate them to be the new feature.
		new_features.append( np.append(sift_vec, hist_vec) )
	return new_features

new_features_train = get_new_feature(x_train, features_train)
new_features_test = get_new_feature(x_test, features_test)

clf = SVC(kernel='rbf',decision_function_shape="ovr", C=1.0, class_weight="balanced")
clf.fit(new_features_train, label_train)
y_pred_b = clf.predict(new_features_test)
#visualize_feature_points(x_train[2], features_train[2], normalized=True)

exit(0)