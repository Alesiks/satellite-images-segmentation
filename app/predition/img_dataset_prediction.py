import os

import numpy
import tifffile as tiff

from app.config.main_config import TEST_INPUT_DATA_PATH, TEST_OUTPUT_DATA_PATH
from app.predition.img_prediction import ImagePredictor


class DatasetPredictor(object):

    def __init__(self, prediction_weights):
        self.predictor = ImagePredictor(prediction_weights)

    def predictInriaAerialDataset(self, data_path=TEST_INPUT_DATA_PATH):
        filenames = os.listdir(data_path)
        for f in filenames:
            tempRes = self.predictor.predict_image_mask(data_path + f)
            tempRes = tempRes.reshape(len(tempRes), len(tempRes[0]))
            tempRes = tempRes * 255
            tempRes = tempRes.astype(numpy.uint8)
            tiff.imsave(TEST_OUTPUT_DATA_PATH + f, tempRes)