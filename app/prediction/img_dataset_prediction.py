import os

import numpy
import tifffile as tiff

from app.config.main_config import TEST_INPUT_DATA_PATH, TEST_OUTPUT_DATA_PATH
from app.prediction.img_prediction import ImagePredictor


class DatasetPredictor(object):

    def __init__(self, prediction_weights, model):
        self.predictor = ImagePredictor(prediction_weights, model)

    def predictInriaAerialDataset(self, data_input_path=TEST_INPUT_DATA_PATH, data_output_path=TEST_OUTPUT_DATA_PATH):
        filenames = os.listdir(data_input_path)
        for f in filenames:
            tempRes = self.predictor.predict_image_mask(data_input_path + f)
            tempRes = tempRes.reshape(len(tempRes), len(tempRes[0]))
            tempRes = tempRes * 255
            tempRes = tempRes.astype(numpy.uint8)
            tiff.imsave(data_output_path + f, tempRes)