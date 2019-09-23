import csv
import random
from typing import List, Set, Tuple, Any

import numpy as np
import tifffile as tiff

from app.config.main_config import IMAGE_SIZE, IMAGE_FORMAT, ROTATION_ANGLE_STEP
from app.entity.entities import ImageDto, ImageCoordinates, SourceAndMaskImagesDto

csv.field_size_limit(13107200);


class ImagesCropper(object):
    # running vertically downwards across rows
    ROWS_AXIS = 0
    # running horizontally across columns
    COLUMNS_AXIS = 1

    MIN_SAVE_IMG_THRESHOLD = 0.11
    MAX_SAVE_IMG_THRESHOLD = 0.89

    NUM_IMAGES_BELLOW_MIN_THRESHOLD = 5
    NUM_IMAGES_ABOVE_MAX_THRESHOLD = 2

    def __init__(self):
        self.total_objects_ratio = 0
        self.total_objects_num = 0

    def crop_image(self, image_name, image, cropped_images_folder_path, shift_step=IMAGE_SIZE):
        x = 0
        width = image.shape[self.COLUMNS_AXIS]
        height = image.shape[self.ROWS_AXIS]

        while x <= width - IMAGE_SIZE:
            y = 0
            x_start = x
            x_end = x_start + IMAGE_SIZE
            while y <= height - IMAGE_SIZE:
                y_start = y
                y_end = y_start + IMAGE_SIZE
                tiff.imsave((cropped_images_folder_path + '{}___{}' + IMAGE_FORMAT).
                            format(image_name, "x=" + str(x_start) + ", y=" + str(y_start)),
                            image[y_start:y_end, x_start:x_end])
                y += shift_step
            x += shift_step

    def crop_image_randomly(self, image_name, image, image_mask, cropped_images_folder_path,
                            cropped_images_mask_folder_path, croppped_images_quantity, rotation_angle_stop=90,
                            make_flip=False):
        x = len(image)
        y = len(image[0])

        images_bellow_min_threshold = 0
        images_above_max_threshold = 0

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

            if objects_ratio > self.MIN_SAVE_IMG_THRESHOLD and objects_ratio < self.MAX_SAVE_IMG_THRESHOLD:
                self.__crop_image(image, cropped_mask, cropped_images_folder_path, cropped_images_mask_folder_path,
                                  image_name,
                                  x_start, y_start, x_end, y_end, objects_ratio, rotation_angle_stop, make_flip)

            elif objects_ratio < self.MIN_SAVE_IMG_THRESHOLD and images_bellow_min_threshold < self.NUM_IMAGES_BELLOW_MIN_THRESHOLD:
                self.__crop_image(image, cropped_mask, cropped_images_folder_path, cropped_images_mask_folder_path,
                                  image_name,
                                  x_start, y_start, x_end, y_end, objects_ratio, rotation_angle_stop, make_flip)

                images_bellow_min_threshold += 1

            elif objects_ratio > self.MAX_SAVE_IMG_THRESHOLD and images_above_max_threshold < self.NUM_IMAGES_ABOVE_MAX_THRESHOLD:
                self.__crop_image(image, cropped_mask, cropped_images_folder_path, cropped_images_mask_folder_path,
                                  image_name,
                                  x_start, y_start, x_end, y_end, objects_ratio, rotation_angle_stop, make_flip)

                images_above_max_threshold += 1

    def __crop_image(self, image, cropped_mask, cropped_images_folder_path, cropped_images_mask_folder_path, image_name,
                     x_start, y_start, x_end, y_end, objects_ratio, rotation_angle_stop, make_flip):
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
        self.__save_img(rotated_img_mask, cropped_images_mask_folder_path, image_name, angle, x_start, y_start, False)

        if make_flip:
            flipped_img = np.flip(rotated_img, 0)
            flipped_img_mask = np.flip(rotated_img_mask, 0)

            self.__save_img(flipped_img, cropped_images_folder_path, image_name, angle, x_start, y_start, True)
            self.__save_img(flipped_img_mask, cropped_images_mask_folder_path, image_name, angle, x_start, y_start,
                            True)

    def __save_img(self, image, cropped_images_folder_path, image_name, angle, x_start, y_start, isFlipped):
        if isFlipped:
            cropped_images_folder_path += "flipped_"

        img_name = (cropped_images_folder_path + '{}___{}' + IMAGE_FORMAT). \
            format(image_name, "__rotated" + str(angle) + "_x=" + str(x_start) + ", y=" + str(y_start))

        tiff.imsave(img_name, image)


class SequentialImageCropper():
    # running vertically downwards across rows
    ROWS_AXIS = 0
    # running horizontally across columns
    COLUMNS_AXIS = 1

    def __init__(self):
        pass

    def crop(self, image, image_name, shift_step):
        x = 0
        width = image.shape[self.COLUMNS_AXIS]
        height = image.shape[self.ROWS_AXIS]

        cropped_images_list = []

        while x <= width - IMAGE_SIZE:
            y = 0
            x_start = x
            x_end = x_start + IMAGE_SIZE
            while y <= height - IMAGE_SIZE:
                y_start = y
                y_end = y_start + IMAGE_SIZE

                cropped_image = image[y_start:y_end, x_start:x_end]
                image_dto = ImageDto(
                    ('{}___{}' + IMAGE_FORMAT).format(image_name, "x=" + str(x_start) + ", y=" + str(y_start)),
                    cropped_image)

                cropped_images_list.append(image_dto)

                y += shift_step
            x += shift_step

        return cropped_images_list


class RandomImageCropper():
    MIN_SAVE_IMG_THRESHOLD = 0.11
    MAX_SAVE_IMG_THRESHOLD = 0.89

    NUM_IMAGES_BELLOW_MIN_THRESHOLD = 5
    NUM_IMAGES_ABOVE_MAX_THRESHOLD = 2

    def __init__(self):
        pass

    def crop_image_randomly(self, image_name: str, image: np.ndarray, image_mask: np.ndarray,
                            croppped_images_quantity: int) -> List[SourceAndMaskImagesDto]:
        x = len(image)
        y = len(image[0])

        images_bellow_min_threshold = 0
        images_above_max_threshold = 0

        cropped_images_list: List[SourceAndMaskImagesDto] = []
        used_coordinates_set: Set[Tuple[int, int]] = set()

        for i in range(croppped_images_quantity):

            image_coordinates = self.__get_image_coordinates(x, y, used_coordinates_set)
            if image_coordinates is None:
                continue

            used_coordinates_set.add((image_coordinates.x_start, image_coordinates.y_start))

            cropped_mask = self.__crop_image(image_mask, image_coordinates)
            objects_area = np.sum(cropped_mask[:, :])
            objects_ratio = objects_area / (IMAGE_SIZE ** 2)

            if objects_ratio > self.MIN_SAVE_IMG_THRESHOLD and objects_ratio < self.MAX_SAVE_IMG_THRESHOLD:
                new_image = self.__crop_image(image, image_coordinates)

            elif objects_ratio < self.MIN_SAVE_IMG_THRESHOLD and images_bellow_min_threshold < self.NUM_IMAGES_BELLOW_MIN_THRESHOLD:
                new_image = self.__crop_image(image, image_coordinates)
                images_bellow_min_threshold += 1

            elif objects_ratio > self.MAX_SAVE_IMG_THRESHOLD and images_above_max_threshold < self.NUM_IMAGES_ABOVE_MAX_THRESHOLD:
                new_image = self.__crop_image(image, image_coordinates)
                images_above_max_threshold += 1

            if new_image is not None:
                source_dto = ImageDto(image_name, image)
                mask_dto = ImageDto(image_name, image)
                sample = SourceAndMaskImagesDto(source_dto, mask_dto)
                cropped_images_list.append(sample)

        return cropped_images_list

    def __get_image_coordinates(self, x: int, y: int, coords_set: Set[Tuple[int, int]]) -> ImageCoordinates:
        x_start = random.randint(1, x - (IMAGE_SIZE + 2))
        y_start = random.randint(1, y - (IMAGE_SIZE + 2))
        x_end = x_start + IMAGE_SIZE
        y_end = y_start + IMAGE_SIZE

        image_coordinates = None
        if self.__is_this_part_cropped_before(coords_set, x_start, y_start) is False:
            image_coordinates = ImageCoordinates(x_start, x_end, y_start, y_end)
        return image_coordinates

    def __is_this_part_cropped_before(self, coords_set: Set[Tuple[int, int]], x_start: int, y_start: int) -> bool:
        return (x_start, y_start) in coords_set

    def __crop_image(self, image: np.ndarray, image_coordinates: ImageCoordinates) -> np.ndarray:
        return image[image_coordinates.x_start:image_coordinates.x_end,
               image_coordinates.y_start:image_coordinates.y_end]
