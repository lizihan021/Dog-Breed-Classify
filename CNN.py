import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
from utils import get, visualize_feature_points
from dog import DogsDataset
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
# must first load train set !!!
x_train, label_train, features_train = dogs.trainX, dogs.trainY, dogs.train_features
x_test, lable_test, features_test = dogs.testX, dogs.testY, dogs.test_features

callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True)]
history = model.fit(x_train, features_train,
          batch_size=BATCH_SIZE,
          epochs=CNN_EPOCHS,
          verbose=2,
          validation_split=VALIDATION_SPLIT,
          callbacks=callbacks)

max_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print('Min validation loss = {0:.4f} (epoch {1:d})'.format(max_val_loss, idx+1))

score = model.evaluate(x_test, features_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

def get_sift(img, kp, mode = "gray"):
	if mode == "gray":
		return 0
	elif mode == "color":
		return 1
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	sift = cv2.SIFT()
	kp = sift.detect(gray,None)
	print(kp)
	kp,des = sift.compute(gray,kp)
	print(kp)
	img=cv2.drawKeypoints(gray,kp)
	cv2.imwrite('sift_keypoints.jpg',img)




exit(0)