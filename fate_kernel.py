import os

import cv2
import numpy as np
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

image_paths = []
data = []
labels = []

# import os
for dirname, _, filenames in os.walk('dataset-100/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        image_paths.append(os.path.join(dirname, filename))


for imagePath in image_paths:
    # print(imagePath)
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (100, 100))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

EPOCHS = 100
INIT_LR = 1e-3
BS = 32

model = Sequential()
inputShape = (100, 100, 3)
chanDim = -1

        # CONV => RELU => POOL
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=inputShape))  # First pair "CONV -> RELU", resembling VGG16's structure
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())  # Normalizing output
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))  # Pooling layer
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))  # Second pair of "CONV -> RELU"
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())  # Normalizing output
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))  # Second pooling layer
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))  # Third pair of "CONV -> RELU"
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())  # Normalizing output
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))  # Third pooling layer
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))  # Fourth pair of "CONV -> RELU"
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())  # Normalizing output
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))  # Fourth pooling layer
model.add(Dropout(0.3))

        # first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(4096, activation='relu'))  # Fully connected layer
model.add(Dense(4096, activation='relu'))  # Fully connected layer
model.add(Dense(1024, activation='relu'))  # Fully connected layer
model.add(BatchNormalization())  # Finishing up the net

        # softmax classifier
model.add(Dense(5))
model.add(Activation("softmax"))

opt_Adam = Adam(lr=INIT_LR)
opt_sgd = SGD()
model.compile(loss="categorical_crossentropy", optimizer=opt_sgd, metrics=["accuracy"])


print("[INFO] training network...")
# model.fit(trainX, trainY, epochs=100, batch_size=8, verbose=1)
H = model.fit(
    trainX, trainY,
    steps_per_epoch=len(trainX) // 128,
    epochs=1000, verbose=1)

print("[INFO] serializing network...")
model.save("fate.model", save_format="h5")


