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
import cv2

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

def get_sift(img, kp = [], mode = "gray"):
	if mode == "gray":
		img = denormalize_image(img)
		img = np.array(img*255, dtype="uint8")
		gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		print(kp)
		kp,des = sift.compute(gray,kp)
		print(des.shape)
		print(des)
		img=cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imshow('result', img), cv2.waitKey(0)
		cv2.destroyWindow("result")

	elif mode == "color":
		return 1

def generate_kp(features_line):
	kp = []
	left_eye = np.array([features_line[0], features_line[1]])
	right_eye = np.array([features_line[2], features_line[3]])
	nose = np.array([features_line[4], features_line[5]])
	face_line = left_eye - right_eye
	kp_scale = np.linalg.norm(face_line)/2
	kp_angle = np.degrees(np.arctan2(face_line[1], face_line[0]))
	for i in range(8):
		tmp_x = features_line[2*i]
		tmp_y = features_line[2*i+1]
		kp.append(cv2.KeyPoint(tmp_x, tmp_y, _size=kp_scale, _angle=kp_angle))

	# mask
	center = (left_eye + right_eye + nose)/3
	face_scale = np.linalg.norm(face_line)*1.5
	mask = np.zeros((128,128), dtype="uint8")

	col_min = np.maximum(int(center[0]-face_scale), 0)
	col_max = np.minimum(int(center[0]+face_scale), 127)
	row_min = np.maximum(int(center[1]-face_scale), 0)
	row_max = np.minimum(int(center[1]+face_scale), 127)

	mask[row_min:row_max,col_min:col_max] = 1
	return kp, mask

kp, mask = generate_kp(features_train[5])
img = denormalize_image(x_train[5])
img = np.array(img*255, dtype="uint8")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('result', mask*gray), cv2.waitKey(0)
get_sift(x_train[5], kp)

visualize_feature_points(x_train[2], features_train[2], normalized=True)

exit(0)