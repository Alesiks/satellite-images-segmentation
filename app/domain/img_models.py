import numpy as np

class ImageDto():
    def __init__(self, image_name: str, image: np.ndarray):
        self.image_name = image_name
        self.image = image


class SourceAndMaskImagesDto():
    def __init__(self, source: ImageDto, mask: ImageDto):
        self.source = source
        self.mask = mask


class ImageCoordinates():
    def __init__(self, x_start: int, x_end: int, y_start: int, y_end: int):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end


class DatasetPreparationSettings():
    def __init__(self, flip: bool, rotate: int, segments_num_for_each_image: int):
        self.flip = flip
        self.rotate = rotate
        self.segments_num_for_each_image = segments_num_for_each_image

        # @property
        # def flip(self):
        #     return self.__flip
        #
        # @flip.setter
        # def flip(self, flip):
        #     self.flip = flip
