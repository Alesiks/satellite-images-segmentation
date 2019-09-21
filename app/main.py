import numpy
import tifffile

from app.config.main_config import BUILDINGS_MASK_DATA_PATH
from app.net.neuralnetwork import NeuralNetwork
from app.net.unet import UNET
from app.prediction.img_dataset_prediction import DatasetPredictor
from app.prediction.img_prediction import ImagePredictor
from app.preparation.img_preparer import ImagesPreparer

# samples_path="D:\\machine learning data\\Сhinese\\data\\0.2resolution\\tif\\2016\\"
# geojson_path="D:\\machine learning data\\Сhinese\\data\\0.2resolution\\geojson\\2016\\"

# preparer = ImagesPreparer()
# preparer.create_data_inria_aerial_images(100, False)


unet = UNET()
net = NeuralNetwork(unet.model)
net.init_data()
net.train_network()
numpy.set_printoptions(threshold=numpy.nan)

# unet = UNET()
# imgPrediction = ImagePredictor(["../data/weights.09-0.225.hdf5"], unet.model)
# img = imgPrediction.predict_image_mask("../data/minsk.jpg")
#
# unet = UNET()
# tester = DatasetPredictor(["../data/weights.17-1.951.hdf5"], unet.model)
# tester.predictInriaAerialDataset()  # "D:\\machine learning data\\Сhinese\\data\\0.2resolution\\tif\\test\\",
#                                     "D:\\machine learning data\\Сhinese\\data\\0.2resolution\\tif\\res\\3\\")