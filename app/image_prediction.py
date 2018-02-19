import os
import shutil

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from app.images.img_cropper import ImagesCropper, IMAGE_FORMAT, IMAGE_SIZE, PREDICTION_IMAGE_SIZE
from app.neuralnetwork import NeuralNetwork


class ImagePrediction(object):

    def __init__(self, prediction_weights):
        self.cropper = ImagesCropper()
        self.net = NeuralNetwork()
        self.prediction_weights = prediction_weights
        pass

    def predict_image_tta(self, image_path):
        image = tiff.imread(image_path).transpose([1, 2, 0])

        res_img = self.__predict(image)

        rot_img = np.rot90(image, 1)
        res_rot_img = self.__predict(rot_img)
        res_rot_img = np.rot90(res_rot_img, 3)

        newar = np.add(res_img, res_rot_img)
        # b = newar > 3
        # newar = b.astype(int)

        tiff.imshow(newar)
        plt.show()

    def predict_image(self, image_path):
        image = tiff.imread(image_path).transpose([1, 2, 0])
        res_img = self.__predict(image)

        tiff.imshow(res_img)
        plt.show()

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
        new_img = self.__concatenate_by_axis(image, 0)
        new_img = self.__concatenate_by_axis(new_img, 1)
        return new_img

    def __concatenate_by_axis(self, image, axis):
        len = image.shape[axis]
        updated_len = self.__get_updated_length(len)

        flipped_part = self.__get_flip_part_img_top(len, updated_len, image, axis)
        new_img = np.concatenate((image, flipped_part), axis)

        flipped_part = self.__get_flip_part_img_bottom(len, updated_len, image, axis)
        new_img = np.concatenate((flipped_part, new_img), axis)
        return new_img

    def __get_updated_length(self, len):
        updated_len = (len // PREDICTION_IMAGE_SIZE) + 1
        updated_len *= PREDICTION_IMAGE_SIZE
        updated_len += IMAGE_SIZE - PREDICTION_IMAGE_SIZE
        return updated_len

    def __get_flip_part_img_top(self, len, updated_len, image, axis):
        flip_part_len = len - (updated_len - len) // 2
        if axis == 0:
            flipped_part = np.flip(image[flip_part_len:len, :], axis)
        elif axis == 1:
            flipped_part = np.flip(image[:, flip_part_len:len], axis)
        return flipped_part

    def __get_flip_part_img_bottom(self, len, updated_len, image, axis):
        diff = (updated_len - len) // 2
        if axis == 0:
            flipped_part = np.flip(image[0:diff, :], axis)
        elif axis == 1:
            flipped_part = np.flip(image[:, 0:diff], axis)
        return flipped_part

    def __get_dataset(self, image, directory, shift_step=IMAGE_SIZE):
        x_len = image.shape[1]
        y_len = image.shape[0]
        dataset = []
        y_border = 0
        while y_border <= y_len - IMAGE_SIZE:
            x_border = 0
            while x_border <= x_len - IMAGE_SIZE:
                img = tiff.imread(directory + "img___x={}, y={}{}".format(x_border, y_border, IMAGE_FORMAT))
                dataset.append(img)
                x_border += shift_step
            y_border += shift_step
        return dataset

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

    def __concat_predictions(self, image, predictions, shift_step=IMAGE_SIZE):
        x_len = image.shape[1]
        y_len = image.shape[0]
        full_image = []
        y_border = 0
        i = 0
        while y_border <= y_len - IMAGE_SIZE:
            x_border = 0
            images = []
            while x_border <= x_len - IMAGE_SIZE:
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