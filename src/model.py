import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD

class VGG11(object):

    def __init__(self, l2_reg):
        self.model = Sequential()

        self.model.add(Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=[32, 32, 1]))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding="same",))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding="same",))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding="same",))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        # self.model.add(Conv2D(512, (3, 3), activation='relu', padding="same",))
        # self.model.add(Conv2D(512, (3, 3), activation='relu', padding="same",))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        # self.model.add(Conv2D(512, (3, 3), activation='relu', padding="same",))
        # self.model.add(Conv2D(512, (3, 3), activation='relu', padding="same",))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

        decay = 5e-4 if l2_reg else 0
        sgd = SGD(lr=0.01, decay=decay, momentum=0.9, nesterov=True)
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
