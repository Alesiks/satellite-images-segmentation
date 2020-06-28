import numpy as np
from typing import List

from app.config.main_config import ROTATION_ANGLE_STEP
from app.domain.img_models import SourceAndMaskImagesDto, ImageDto


class ImageRotationService():

    def __init__(self):
        self.rotation_angle_to_rotations_num = {
            0: 0,
            90: 1,
            180: 2,
            270: 3
        }

    def rotate_list(self, image_list: List[SourceAndMaskImagesDto], angle: int) -> List[SourceAndMaskImagesDto]:
        rotated_images: List[SourceAndMaskImagesDto] = []
        for image in image_list:
            for curr_angle in range(ROTATION_ANGLE_STEP, angle + ROTATION_ANGLE_STEP, ROTATION_ANGLE_STEP):
                rotated = self.rotate(image, curr_angle)
                rotated_images.append(rotated)
        return rotated_images

    def rotate(self, source_and_mask_images: SourceAndMaskImagesDto, angle: int) -> SourceAndMaskImagesDto:
        new_source_img_dto = self.__rotate_img_dto(source_and_mask_images.source, angle)
        new_mask_img_dto = self.__rotate_img_dto(source_and_mask_images.mask, angle)

        return SourceAndMaskImagesDto(new_source_img_dto, new_mask_img_dto)


    def __rotate_img_dto(self, img_dto: ImageDto, angle: int) -> ImageDto:
        new_img_name = self.__update_image_name_after_rotation(img_dto.image_name, angle)
        rotation_num = self.rotation_angle_to_rotations_num.get(angle)
        rotated_img = np.rot90(img_dto.image, rotation_num)

        return ImageDto(new_img_name, rotated_img)


    def __update_image_name_after_rotation(self, img_name: str, angle: int) -> str:
        return img_name + "_rotate_" + str(angle)


class ImageFlipService():

    def flip_list(self, image_list: List[SourceAndMaskImagesDto]) -> List[SourceAndMaskImagesDto]:
        flipped_images: List[SourceAndMaskImagesDto] = []
        for image in image_list:
            flipped = self.flip(image)
            flipped_images.append(flipped)
        return flipped_images

    def flip(self, source_and_mask_images: SourceAndMaskImagesDto) -> SourceAndMaskImagesDto:
        new_source_img_dto = self.__flip_img_dto(source_and_mask_images.source)
        new_mask_img_dto = self.__flip_img_dto(source_and_mask_images.mask)

        return SourceAndMaskImagesDto(new_source_img_dto, new_mask_img_dto)

    def __flip_img_dto(self, img_dto: ImageDto) -> ImageDto:
        new_img_name = self.__update_image_name_after_flip(img_dto.image_name)
        flipped_img = np.flip(img_dto.image, 0)

        return ImageDto(new_img_name, flipped_img)

    def __update_image_name_after_flip(self, img_name: str) -> str:
        return img_name + "_flip"
