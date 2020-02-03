import csv
import random
from typing import List, Set, Tuple, Any

import numpy as np
import tifffile as tiff

from app.config.main_config import IMAGE_SIZE, IMAGE_FORMAT, ROTATION_ANGLE_STEP
from app.domain.img_models import ImageDto, ImageCoordinates, SourceAndMaskImagesDto

csv.field_size_limit(13107200);


class SequentialImageCropper():
    # running vertically downwards across rows
    ROWS_AXIS = 0
    # running horizontally across columns
    COLUMNS_AXIS = 1

    def __init__(self):
        pass

    def crop(self,
             image_name: str,
             image: np.ndarray,
             cropped_images_folder_path: str,
             shift_step: int = IMAGE_SIZE
             ) -> None:
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


class RandomImageCropper():
    MIN_SAVE_IMG_THRESHOLD = 0.11
    MAX_SAVE_IMG_THRESHOLD = 0.89

    NUM_IMAGES_BELLOW_MIN_THRESHOLD = 5
    NUM_IMAGES_ABOVE_MAX_THRESHOLD = 2

    def __init__(self):
        pass

    def crop(self,
             image_name: str,
             image: np.ndarray,
             image_mask: np.ndarray,
             croppped_images_quantity: int
             ) -> List[SourceAndMaskImagesDto]:
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
                new_image_name = self.__update_image_name_after_crop(image_name, image_coordinates)
                source_dto = ImageDto(new_image_name, new_image)
                mask_dto = ImageDto(new_image_name, cropped_mask)
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

    def __update_image_name_after_crop(self, image_name: str, image_coordinates: ImageCoordinates) -> str:
        new_image_name = "{}_x={},y={}".format(image_name, image_coordinates.x_start, image_coordinates.y_start)
        return new_image_name

