from app.config.main_config import IMAGE_FORMAT
from app.domain.img_models import ImageDto
import tifffile as tiff


class LocalImageSaveService():

    def save(self, image_dto: ImageDto, directory: str) -> None:
        image = image_dto.image
        path = "{}{}{}".format(directory, image_dto.image_name, IMAGE_FORMAT)
        tiff.imsave(path, image)