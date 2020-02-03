

from app.config.main_config import BUILDINGS_MASK_DATA_PATH, BUILDINGS_DATA_PATH, TRAIN_INPUT_DATA_PATH, \
    VALIDATION_INPUT_DATA_PATH
from app.domain.img_models import DatasetPreparationSettings
from app.net.neuralnetwork import NeuralNetwork
from app.net.unet import UNET
from app.prediction.img_predictions import DatasetPredictor
from app.preparation.img_preparer import InriaDatasetPreparer

datasetSettings = DatasetPreparationSettings(flip=True, rotate=0, segments_num_for_each_image=40)
preparer = InriaDatasetPreparer(data_path=BUILDINGS_DATA_PATH,
                                ground_truth_data_path=BUILDINGS_MASK_DATA_PATH,
                                train_dataset_path=TRAIN_INPUT_DATA_PATH,
                                validation_dataset_path=VALIDATION_INPUT_DATA_PATH,
                                dataset_settings=datasetSettings)

preparer.prepare_dataset_for_training()
#
#
# unet = UNET()
# net = NeuralNetwork(unet.model)
# net.init_data()
# net.train_network()
# numpy.set_printoptions(threshold=numpy.nan)

# unet = UNET()
# imgPrediction = ImagePredictor(["../data/weights.09-0.225.hdf5"], unet.model)
# img = imgPrediction.predict_image_mask("../data/minsk.jpg")
#

#
# unet = UNET()
# tester = DatasetPredictor(["../data/weights.34-1.560.hdf5"], unet.model)
# tester.predictInriaAerialDataset()  # "D:\\machine learning data\\Сhinese\\data\\0.2resolution\\tif\\test\\",
#                                     # "D:\\machine learning data\\Сhinese\\data\\0.2resolution\\tif\\res\\3\\")