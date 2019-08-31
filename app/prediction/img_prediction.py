import os
import shutil

import math
from PIL import Image

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from app.config.main_config import PREDICTION_IMAGE_SIZE, IMAGE_SIZE, IMAGE_FORMAT
from app.net.neuralnetwork import NeuralNetwork
from app.preparation.img_cropper import ImagesCropper
import cv2



class ImagePredictor(object):
    # running vertically downwards across rows
    ROWS_AXIS = 0
    # running horizontally across columns
    COLUMNS_AXIS = 1

    def __init__(self, prediction_weights, model):
        self.cropper = ImagesCropper()
        self.net = NeuralNetwork(model)
        self.prediction_weights = prediction_weights

    # predict image mask according predictions of image and rotated image
    def predict_image_mask_tta(self, image_path):
        image = tiff.imread(image_path)#.transpose([1, 2, 0])
        res_img = self.__predict(image)

        rot_img = np.rot90(image, 1)
        res_rot_img = self.__predict(rot_img)
        res_rot_img = np.rot90(res_rot_img, 3)

        newar = np.add(res_img, res_rot_img)
        b = newar > 1
        newar = b.astype(int)

        # tiff.imshow(newar)
        # plt.show()
        return newar

    def predict_image_mask(self, image_path):
        image = tiff.imread(image_path)#.transpose([1, 2, 0])
        # image = cv2.imread(image_path)
        res_img = self.__predict(image)

        # tiff.imshow(res_img)
        # plt.show()
        return res_img

    # predicts mask for image according given prediction_weights
    def __predict(self, image):
        directory = "../tempdata/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        updated_image = self.__resize_image_for_prediction(image)
        self.cropper.crop_image('img', updated_image, directory, PREDICTION_IMAGE_SIZE)

        dataset = self.__get_dataset(updated_image, directory, shift_step=PREDICTION_IMAGE_SIZE)
        self.net.x_test = np.array(dataset, np.float32)

        predictions = []
        for weights in self.prediction_weights:
            image_prediction = self.net.test_network(weights)
            res = self.__concat_predictions(updated_image, image_prediction, shift_step=PREDICTION_IMAGE_SIZE)
            start_coord_x = (res.shape[0]-image.shape[0]) // 2
            end_coord_x = res.shape[0] - start_coord_x
            start_coord_y = (res.shape[1]-image.shape[1]) // 2
            end_coord_y = res.shape[1] - start_coord_y

            res = res[start_coord_x:end_coord_x,start_coord_y:end_coord_y]
            predictions.append(res)


        newres = np.zeros(res.shape)
        for image_prediction in predictions:
            newres = np.add(newres, image_prediction)

        shutil.rmtree(directory)
        return newres

    # resizes image for future it splitting into images with PREDICTION_IMAGE_SIZE
    # for image is added frame, after predictions this frame disappear
    # frame consists of flipped image border parts
    def __resize_image_for_prediction(self, image):
        new_img = self.__concatenate_by_axis(image, self.ROWS_AXIS)
        new_img = self.__concatenate_by_axis(new_img, self.COLUMNS_AXIS)
        return new_img

    def __concatenate_by_axis(self, image, axis):
        axis_len = image.shape[axis]
        updated_axis_len = self.__get_updated__axis_length(axis_len)

        flipped_part = self.__get_flip_part_img_top(axis_len, updated_axis_len, image, axis)
        new_img = np.concatenate((image, flipped_part), axis)

        flipped_part = self.__get_flip_part_img_bottom(axis_len, updated_axis_len, image, axis)
        new_img = np.concatenate((flipped_part, new_img), axis)
        return new_img

    def __get_updated__axis_length(self, axis_len):
        updated_len = (axis_len // PREDICTION_IMAGE_SIZE) + 1
        updated_len *= PREDICTION_IMAGE_SIZE
        updated_len += IMAGE_SIZE - PREDICTION_IMAGE_SIZE
        return updated_len

    def __get_flip_part_img_top(self, len, updated_len, image, axis):
        flip_part_len = len - (updated_len - len) // 2
        if axis == self.ROWS_AXIS:
            flipped_part = np.flip(image[flip_part_len:len, :], axis)
        elif axis == self.COLUMNS_AXIS:
            flipped_part = np.flip(image[:, flip_part_len:len], axis)
        return flipped_part

    def __get_flip_part_img_bottom(self, len, updated_len, image, axis):
        diff = math.ceil((updated_len - len) / 2)
        if axis == self.ROWS_AXIS:
            flipped_part = np.flip(image[0:diff, :], axis)
        elif axis == self.COLUMNS_AXIS:
            flipped_part = np.flip(image[:, 0:diff], axis)
        return flipped_part

    def __get_dataset(self, image, directory, shift_step=IMAGE_SIZE):
        width = image.shape[1]
        height = image.shape[0]
        dataset = []
        y_border = 0
        while y_border <= height - IMAGE_SIZE:
            x_border = 0
            while x_border <= width - IMAGE_SIZE:
                img = tiff.imread(directory + "img___x={}, y={}{}".format(x_border, y_border, IMAGE_FORMAT))
                dataset.append(img)
                x_border += shift_step
            y_border += shift_step
        return dataset

    def __concat_predictions(self, image, predictions, shift_step=IMAGE_SIZE):
        width = image.shape[1]
        height = image.shape[0]
        full_image = []
        y_border = 0
        i = 0
        while y_border <= height - IMAGE_SIZE:
            x_border = 0
            images = []
            while x_border <= width - IMAGE_SIZE:
                img = predictions[i]
                crop_start = int((IMAGE_SIZE - shift_step) / 2)
                crop_end = int(crop_start + shift_step)
                images.append(img[crop_start:crop_end, crop_start:crop_end])
                x_border += shift_step
                i += 1
            x_concat = np.concatenate(images, axis=1)
            full_image.append(x_concat)
            y_border += shift_step

        res = np.concatenate(full_image, axis=0)
        return res

    def __concat_images(self, image, directory):
        x_len = image.shape[1]
        y_len = image.shape[0]
        full_image = []
        x_border = 0
        while x_border <= x_len - IMAGE_SIZE:
            y_border = 0
            images = []
            while y_border <= y_len - IMAGE_SIZE:
                img = tiff.imread(directory + "img___x={}, y={}{}".format(x_border, y_border, IMAGE_FORMAT))
                images.append(img)
                y_border += IMAGE_SIZE
            x_concat = np.concatenate(images, axis=1)
            full_image.append(x_concat)
            x_border += IMAGE_SIZE

        res = np.concatenate(full_image, axis=0)
        return res