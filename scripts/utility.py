from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from typing import List
import numpy as np
import cv2

def getTargets(filepaths: List[str]) -> List[str]:
    labels = [fp.split('/')[-1].split('_')[0] for fp in filepaths]
    return labels

def encodeLabels(y_train: List, y_test: List):
    label_encoder = LabelEncoder()

    y_train_labels = label_encoder.fit_transform(y_train)
    y_test_labels = label_encoder.transform(y_test)

    y_train_1h = to_categorical(y_train_labels)
    y_test_1h = to_categorical(y_test_labels)

    LABELS = label_encoder.classes_
    print(f"{LABELS} -- {label_encoder.transform(LABELS)}")
    return LABELS, y_train_1h, y_test_1h

def getFeatures(filepaths: List[str]) -> np.array:
    images = []
    for imagePath in filepaths:
        image = cv2.imread(imagePath)
        images.append(image)
    return np.array(images)


def buildModel(inputShape: tuple, classes: int) -> Sequential:
    model = Sequential()
    height, width, depth = inputShape
    inputShape = (height, width, depth)
    chanDim = -1

    model.add(Conv2D(32, 3, padding="same", name='conv_32_1', input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, padding="same", name='conv_64_1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, 3, padding="same", name='conv_64_2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, 3, padding="same", name='conv_128_1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, 3, padding="same", name='conv_128_2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, 3, padding="same", name='conv_128_3'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, name='fc_1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes, name='output'))
    model.add(Activation("softmax"))
    return model