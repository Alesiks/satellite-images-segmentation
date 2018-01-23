import numpy as np
from app.config import *
from app.images.img_preparer import ImagesPreparer
from app.neuralnetwork import NeuralNetwork
import tifffile as tiff
import matplotlib.pyplot as plt


preparer = ImagesPreparer()
preparer.create_data(1500)
# preparer.create_data(350, True)

net = NeuralNetwork()
net.init_data()
net.train_network()

# net.x_test = net.create_image_dataset(TEST_INPUT_DATA_PATH)
# predictions = net.test_network( "../data/weights.01-0.00.hdf5")
#
# for prediction in predictions:
#     tiff.imshow(prediction)
# plt.show()