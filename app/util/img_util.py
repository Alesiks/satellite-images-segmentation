import collections
import os

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from app.config.main_config import BUILDINGS_MASK_DATA_PATH, IMAGE_FORMAT, TRAIN_OUTPUT_DATA_PATH


class ImageDiscoverUtil(object):

    def __init__(self):
        pass

    def get_buildings_distribution(self, images_mask_path="../" + BUILDINGS_MASK_DATA_PATH):
        self.files = os.listdir(images_mask_path)
        dictionary = {}

        for f in self.files:
            if f.endswith(IMAGE_FORMAT):
                img = tiff.imread(images_mask_path + f)
                buildings_area = np.sum(img[:, :]) #/ 255
                buildings_area /= (img.shape[0] * img.shape[1])
                buildings_area = round(buildings_area, 2)
                buildings_area = (int) (buildings_area * 100)

                if dictionary.get(buildings_area) is None :
                    dictionary[buildings_area] = 1
                else :
                    dictionary[buildings_area] += 1

        od = collections.OrderedDict(sorted(dictionary.items()))
        plt.plot(od.keys(), od.values(), 'ro')
        plt.xticks(np.arange(min(od.keys()), max(od.keys()) + 1, 2))
        plt.yticks(np.arange(min(od.values()), max(od.values()) + 1, int((max(od.values())-min(od.values())) / 40)))
        # plt.figure(figsize=(1280/120, 960/120), dpi=120)
        plt.savefig('../../data/data_distribution.png')



np.set_printoptions(threshold=np.nan)

imageUtil = ImageDiscoverUtil()
# imageUtil.get_buildings_distribution()

imageUtil.get_buildings_distribution(images_mask_path=TRAIN_OUTPUT_DATA_PATH)