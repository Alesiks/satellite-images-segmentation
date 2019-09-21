
class ImageDto():

    def __init__(self, image_name, image):
       self.image_name = image_name
       self.image = image


class SourceAndMaskImagesDto():

    def __init__(self, source, mask):
        self.source = source
        self.mask = mask


class ImageCoordinates():

    def __init__(self, x_start, x_end, y_start, y_end):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end