import numpy as np
from app.config import *
from app.image_prediction import ImagePrediction
from app.images.img_preparer import ImagesPreparer
from app.neuralnetwork import NeuralNetwork
import tifffile as tiff
import matplotlib.pyplot as plt


# preparer = ImagesPreparer()
# preparer.create_data(1000)
# preparer.create_data(350, True)

# net = NeuralNetwork()
# net.init_data()
# net.train_network_kfold_validation(folds_num=3)


imgPrediction = ImagePrediction(["../data/zz/exp_8__k_folds/0-weights.38-0.204.hdf5", "../data/zz/exp_8__k_folds/1-weights.39-0.215.hdf5"])
imgPrediction.predict_image_tta("../data/buildings for test/6020_1_4.tif")
# imgPrediction.predict_image_tta("../data/three_band/all images/6070_2_3.tif")

# im = tiff.imread("../data/buildings for test/exp1.tif").transpose(2,0,1)
# print(im.shape)

