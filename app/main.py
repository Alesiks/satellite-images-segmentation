import tifffile

from app.config.main_config import BUILDINGS_MASK_DATA_PATH
from app.net.neuralnetwork import NeuralNetwork
from app.net.unet import UNET
from app.predition.img_dataset_prediction import DatasetPredictor
from app.predition.img_prediction import ImagePredictor
from app.preparation.img_preparer import ImagesPreparer

# preparer = ImagesPreparer()
# preparer.create_data_inria_aerial_images(80, False)


unet = UNET()

# # tiramisu = Tiramisu()
#
net = NeuralNetwork(unet.model)
net.init_data()
net.train_network()
# numpy.set_printoptions(threshold=numpy.nan)
#
# unet = UNET()
# imgPrediction = ImagePredictor(["../data/weights.09-0.225.hdf5"], unet.model)
# img = imgPrediction.predict_image_mask("../data/minsk.jpg")
#
# unet = UNET()
# tester = DatasetPredictor(["../data/weights.09-0.225.hdf5"], unet.model)
# tester.predictInriaAerialDataset("../data/minsk and other/", "../data/mask for minsk and other/")