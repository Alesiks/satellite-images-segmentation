from app.config.main_config import IMAGE_FORMAT
from app.domain.img_models import ImageDto
import tifffile as tiff
import numpy as np

class LocalImageSaveService():

    def save(self, image_dto: ImageDto, directory: str, compress: bool) -> None:
        image = image_dto.image
        path = "{}{}{}".format(directory, image_dto.image_name, IMAGE_FORMAT)
        if compress:
            # image = image.reshape(len(image), len(image[0]))
            image = image * 255
            image = image.astype(np.uint8)
        tiff.imsave(path, image)