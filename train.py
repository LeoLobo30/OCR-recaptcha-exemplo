# imports para treinamento de dados
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator

import cv2 as cv

DATA_TRAIN = './dataset'
SIZE_IMAGENS = (60,23)

train_image_generator = ImageDataGenerator(rescale=1./255,
                                            width_shift_range=0.1,
                                            height_shift_range=0.2,
                                            )

train_data_gen = train_image_generator.flow_from_directory(DATA_TRAIN,
    batch_size=75,
    target_size=SIZE_IMAGENS,
    color_mode='grayscale')


model = Sequential()
model.add(Conv2D(64, 3, activation='relu', input_shape=(60,23,1)))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(units=250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=60, activation='relu'))
model.add(Dense(units=34, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.summary()


model.fit_generator(train_data_gen,steps_per_epoch=50,epochs=3)

model.save('model.h5')