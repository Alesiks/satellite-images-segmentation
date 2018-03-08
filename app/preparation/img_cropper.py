import csv
import random

import numpy as np
import tifffile as tiff

from app.config.main_config import IMAGE_SIZE, IMAGE_FORMAT, ROTATION_ANGLE_STEP

csv.field_size_limit(13107200);


class ImagesCropper(object):
    SAVE_IMG_THRESHOLD = 0.14

    def __init__(self):
        self.total_objects_ratio = 0
        self.total_objects_num = 0

    def crop_image(self, image_name, image, cropped_images_folder_path, shift_step=IMAGE_SIZE):
        x = 0
        x_len = image.shape[1]
        y_len = image.shape[0]

        while x <= x_len - IMAGE_SIZE:
            y = 0
            x_start = x
            x_end = x_start + IMAGE_SIZE
            while y <= y_len - IMAGE_SIZE:
                y_start = y
                y_end = y_start + IMAGE_SIZE
                tiff.imsave((cropped_images_folder_path + '{}___{}' + IMAGE_FORMAT).
                            format(image_name, "x=" + str(x_start) + ", y=" + str(y_start)),
                            image[y_start:y_end, x_start:x_end])
                y += shift_step
            x += shift_step

    def crop_image_randomly(self, image_name, image, image_mask, cropped_images_folder_path,
                            cropped_images_mask_folder_path, croppped_images_quantity, rotation_angle_stop=90, make_flip=False):
        x = len(image)
        y = len(image[0])

        coords_set = set()

        for i in range(croppped_images_quantity):
            x_start = random.randint(1, x - (IMAGE_SIZE + 2))
            y_start = random.randint(1, y - (IMAGE_SIZE + 2))
            if (x_start, y_start) in coords_set:
                continue

            coords_set.add((x_start, y_start))
            x_end = x_start + IMAGE_SIZE
            y_end = y_start + IMAGE_SIZE

            cropped_mask = image_mask[x_start:x_end, y_start:y_end]
            objects_area = np.sum(cropped_mask[:, :])
            objects_ratio = objects_area / (IMAGE_SIZE ** 2)

            if objects_ratio > self.SAVE_IMG_THRESHOLD:
                self.total_objects_num += 1
                self.total_objects_ratio += objects_ratio
                cropped_img = image[x_start:x_end, y_start:y_end]

                for angle in range(0, rotation_angle_stop, ROTATION_ANGLE_STEP):
                    self.__rotate_img_and_save(cropped_img, cropped_mask, cropped_images_folder_path,
                                               cropped_images_mask_folder_path, image_name, x_start, y_start, angle,
                                               make_flip)

    def __rotate_img_and_save(self, img, img_mask, cropped_images_folder_path, cropped_images_mask_folder_path,
                              image_name, x_start, y_start, angle, make_flip):

        rotations_angles_switcher = {
            0: 0,
            90: 1,
            180: 2,
            270: 3
        }

        rotation_num = rotations_angles_switcher.get(angle)

        rotated_img = np.rot90(img, rotation_num)
        rotated_img_mask = np.rot90(img_mask, rotation_num)

        self.__save_img(rotated_img, cropped_images_folder_path, image_name, angle, x_start, y_start, False)
        self.__save_img(rotated_img, cropped_images_mask_folder_path, image_name, angle, x_start, y_start, False)

        if make_flip:
            flipped_img = np.flip(rotated_img, 0)
            flipped_img_mask = np.flip(rotated_img_mask, 0)

            self.__save_img(flipped_img, cropped_images_folder_path, image_name, angle, x_start, y_start, True)
            self.__save_img(flipped_img_mask, cropped_images_mask_folder_path, image_name, angle, x_start, y_start, True)

    def __save_img(self, image, cropped_images_folder_path, image_name, angle, x_start, y_start, isFlipped):
        if isFlipped:
            cropped_images_folder_path += "flipped_"

        img_name = (cropped_images_folder_path + '{}___{}' + IMAGE_FORMAT). \
            format(image_name, "__rotated" + str(angle) + "_x=" + str(x_start) + ", y=" + str(y_start))

        tiff.imsave(img_name, image)

