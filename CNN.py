from __future__ import print_function
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras.utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split

# Initialize global variables
CU_DOG_DATASETS_DIR = '../CU_Dogs'

IMAGE_SIZE = 128
DROPOUT_CNN = 0.3
DROPOUT_DENSE = 0.5
BATCH_SIZE = 64
OPTIMIZER = 'adam'
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

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=2,
#           validation_data=(x_test, y_test))

# score = model.evaluate(x_test, y_test, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])