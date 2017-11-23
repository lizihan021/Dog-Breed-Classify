from __future__ import print_function
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras.utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split

# Initialize global variables
CU_DOG_DATASETS_DIR = '../CU_Dogs'

IMAGE_SIZE = 128
DROPOUT = 0.1
BATCH_SIZE = 64
OPTIMIZER = 'adam'
# define CNN

model = Sequential()
model.add(Conv2D(16, kernel_size=(7, 7),
                 activation='relu',
                 input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

plot_model(model, to_file='model.png')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])