import os

import cv2
import tifffile as tiff
from PIL import Image, ImageDraw
from app.config import *
from app.jaccard_metrics import jaccard_coef, jaccard_coef_int
from collections import defaultdict
from keras import Input, Model, callbacks
from keras.layers import MaxPooling2D, np, Convolution2D, UpSampling2D, concatenate, BatchNormalization
from keras.optimizers import Nadam, Adam
from shapely.geometry import MultiPolygon, Polygon

class NeuralNetwork(object):

    def __init__(self):
        self.x = None
        self.y = None
        self.x_val = None
        self.y_val = None
        self.x_test = None


    def init_data(self):
        self.x = self.create_image_dataset(TRAIN_INPUT_DATA_PATH)
        self.y = self.create_image_dataset(TRAIN_OUTPUT_DATA_PATH)
        self.y = self.modify_y_set(self.y)

        self.x_val = self.create_image_dataset(VALIDATION_INPUT_DATA_PATH)
        self.y_val = self.create_image_dataset(VALIDATION_OUTPUT_DATA_PATH)
        self.y_val = self.modify_y_set(self.y_val)

        self.x_test = self.create_image_dataset(TEST_INPUT_DATA_PATH)

    def create_image_dataset(self, directory):
        dataset = []
        filenames = os.listdir(directory)
        for f in filenames:
            if f.endswith(IMAGE_FORMAT):
                image = tiff.imread(directory + f)
                if len(image) == IMAGE_SIZE:
                    dataset.append(image)
        dataset = np.array(dataset, np.float32) #/ 1024.
        return dataset


    # updates y set that each low level element is included in array
    def modify_y_set(self, y):
        new_y = y.reshape(len(y), len(y[0]), len(y[0][0]), 1)
        print(len(new_y), len(new_y[0]), len(new_y[0][0]), len(new_y[0][0][0]))
        return new_y


    def get_unet(self):
        inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

        conv1 = Convolution2D(32, (3, 3), activation='relu', padding="same")(inputs)
        conv1 = Convolution2D(32, (3, 3), activation='relu', padding="same")(conv1)
        conv1 = BatchNormalization(axis=1)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(64, (3, 3), activation='relu', padding="same")(pool1)
        conv2 = Convolution2D(64, (3, 3), activation='relu', padding="same")(conv2)
        conv2 = BatchNormalization(axis=1)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(128, (3, 3), activation='relu', padding="same")(pool2)
        conv3 = Convolution2D(128, (3, 3), activation='relu', padding="same")(conv3)
        conv3 = BatchNormalization(axis=1)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(256, (3, 3), activation='relu', padding="same")(pool3)
        conv4 = Convolution2D(256, (3, 3), activation='relu', padding="same")(conv4)
        conv4 = BatchNormalization(axis=1)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization(axis=1)(conv5)

        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
        conv6 = Convolution2D(256, (3, 3), activation='relu', padding="same")(up6)
        conv6 = Convolution2D(256, (3, 3), activation='relu', padding="same")(conv6)
        conv6 = BatchNormalization(axis=1)(conv6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Convolution2D(128, (3, 3), activation='relu', padding="same")(up7)
        conv7 = Convolution2D(128, (3, 3), activation='relu', padding="same")(conv7)
        conv7 = BatchNormalization(axis=1)(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Convolution2D(64, (3, 3), activation='relu', padding="same")(up8)
        conv8 = Convolution2D(64, (3, 3), activation='relu', padding="same")(conv8)
        conv8 = BatchNormalization(axis=1)(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Convolution2D(32, (3, 3), activation='relu', padding="same")(up9)
        conv9 = Convolution2D(32, (3, 3), activation='relu', padding="same")(conv9)
        conv9 = BatchNormalization(axis=1)(conv9)

        conv10 = Convolution2D(1, (1, 1), activation='sigmoid', padding="same")(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
        return model


    def train_network(self):
        callback = callbacks.ModelCheckpoint("../data/weights.{epoch:02d}-{val_loss:.2f}.hdf5")
        model = self.get_unet()
        model.fit(self.x, self.y, batch_size=40, epochs=35, validation_data=(self.x_val, self.y_val), callbacks=[callback])
        model.save_weights("../data/weights.h5")


    def test_network(self, weights):
        model = self.get_unet()
        model.load_weights(weights)
        predictions = model.predict(self.x_test, batch_size=10)
        # self.convert_predictions_by_threshold(predictions, 0.6)
        return predictions


    def convert_predictions_by_threshold(self, img, threshold):
        for x in np.nditer(img, op_flags=['readwrite']):
            if x > threshold:
                x[...] = 1.0
            else:
                x[...] = 0.0




