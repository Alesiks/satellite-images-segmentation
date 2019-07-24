import csv
import json
import os

import cv2
import gdal
import numpy as np
import shapely
import shapely.wkt
import tifffile as tiff
import matplotlib.pyplot as plt


from app.config.main_config import BUILDINGS_DATA_PATH, IMAGE_FORMAT, BUILDINGS_MASK_DATA_PATH, \
    VALIDATION_INPUT_DATA_PATH, TRAIN_INPUT_DATA_PATH, TRAIN_POLYGONS_PATH, GRID_SIZES_PATH, \
    VALIDATION_OUTPUT_DATA_PATH, TRAIN_OUTPUT_DATA_PATH
from app.preparation.img_cropper import ImagesCropper


class ImagesPreparer(object):
    TRAIN_IMAGES_TO_VALIDATION_RATIO = 7

    def __init__(self, object_type=1, samples_path=BUILDINGS_DATA_PATH, geojson_path=BUILDINGS_MASK_DATA_PATH):
        self.object_type = object_type
        self.samples_path = samples_path
        self.filenames = os.listdir(samples_path)
        self.samples_geojson_path = geojson_path
        self.cropper = ImagesCropper()

    def create_data_chinese_images(self, data_set_size, is_validation=False):
        num_images_added_to_train_set = 0
        for f in self.filenames:
            if f.endswith(IMAGE_FORMAT):
                image_id = f[:-4]
                original_image = tiff.imread((self.samples_path + '{}' + IMAGE_FORMAT).format(image_id))
                image_mask = self.get_mask_for_chinese_images((self.samples_path + '{}' + IMAGE_FORMAT).format(image_id),
                                                         (self.samples_geojson_path + '{}' + ".geojson").format(image_id))

                if is_validation and num_images_added_to_train_set == self.TRAIN_IMAGES_TO_VALIDATION_RATIO:
                    num_images_added_to_train_set = 0

                    self.cropper.crop_image_randomly(image_id, original_image, image_mask, VALIDATION_INPUT_DATA_PATH,
                                                     VALIDATION_OUTPUT_DATA_PATH,
                                                     data_set_size, 180)
                else:
                    self.cropper.crop_image_randomly(image_id, original_image, image_mask, TRAIN_INPUT_DATA_PATH,
                                                     TRAIN_OUTPUT_DATA_PATH,
                                                     data_set_size, 180)
                    num_images_added_to_train_set += 1


    def get_mask_for_chinese_images(self, image_path, image_geojson_path):
        ds = gdal.Open(image_path)
        width = ds.RasterXSize
        height = ds.RasterYSize
        xmin = ds.GetGeoTransform()[0]
        ymax = ds.GetGeoTransform()[3]
        step = 0.2;
        ymin = ymax - height * step
        xmax = xmin + width * step

        shape = (width, height)

        with open(image_geojson_path) as json_content:
            data = json.load(json_content)

            polys = []
            for sh in data['features']:
                if sh['geometry']['coordinates'] and sh['geometry']['coordinates'][0]:
                    geom = np.array(sh['geometry']['coordinates'][0][0])
                    geom_fixed = self.__get_mask(shape, geom, xmin, ymin, xmax, ymax)
                    pts = geom_fixed.astype(int)
                    polys.append(pts)

        mask = np.zeros(shape)
        cv2.fillPoly(mask, polys, 1)
        # mask = mask.astype(bool)
        mask = mask.reshape(len(mask), len(mask[0]), 1)
        # tifffile.imsave("D://test1.jpg", mask)
        # plt.imshow(mask)
        # plt.show()
        return mask


    def __get_mask(self, shape, point, xMin, yMin, xMax, yMax):
        w, h = shape

        dX = xMax - xMin
        dY = yMax - yMin

        x, y = point[:, 0], point[:, 1]

        # w_ = w * (w / (w + 1))
        xp = ((x - xMin) / dX) * shape[0]

        # h_ = h * (h / (h + 1))
        yp = ((y - yMin) / dY) * shape[1]

        return np.concatenate([xp[:, None], yp[:, None]], axis=1)


    def create_data_inria_aerial_images(self, data_set_size,is_validation=False):
        num_images_added_to_train_set = 0
        for f in self.filenames:
            if f.endswith(IMAGE_FORMAT):
                image_id = f[:-4]
                original_image = tiff.imread((BUILDINGS_DATA_PATH + '{}' + IMAGE_FORMAT).format(image_id))
                image_mask = tiff.imread((BUILDINGS_MASK_DATA_PATH + '{}' + IMAGE_FORMAT).format(image_id)) / 255
                if is_validation and num_images_added_to_train_set == self.TRAIN_IMAGES_TO_VALIDATION_RATIO:
                    num_images_added_to_train_set = 0
                    self.cropper.crop_image_randomly(image_id, original_image, image_mask, VALIDATION_INPUT_DATA_PATH,
                                                     VALIDATION_OUTPUT_DATA_PATH,
                                                     data_set_size)
                else:
                    self.cropper.crop_image_randomly(image_id, original_image, image_mask, TRAIN_INPUT_DATA_PATH,
                                                     TRAIN_OUTPUT_DATA_PATH,
                                                     data_set_size)
            num_images_added_to_train_set += 1

        # print(self.cropper.total_objects_ratio)

    # code below is used for preparing data from kaggle contest (Dstl Satellite Imagery Feature Detection)
    # the code below mostly was taken from competitors of Dstl contest
    def create_data_dstl(self, data_set_size, is_validation=False):
        for f in self.filenames:
            if f.endswith(IMAGE_FORMAT):
                image_id = f[:-4]
                original_image = tiff.imread((self.samples_path + '{}' + IMAGE_FORMAT).format(image_id)).transpose(
                    [1, 2, 0])

                x_max, y_min = self.__load_grid_sizes(image_id)
                x_scaler, y_scaler = self.__get_scalers(original_image.shape[:2], x_max, y_min)

                train_polygons = self.__load_train_polygons(image_id, self.object_type)
                train_polygons_scaled = shapely.affinity.scale(train_polygons, xfact=x_scaler, yfact=y_scaler,
                                                               origin=(0, 0, 0))
                train_mask = self.__mask_for_polygons(train_polygons_scaled, original_image.shape[:2])
                if is_validation:
                    self.cropper.crop_image_randomly(image_id, original_image, train_mask, VALIDATION_INPUT_DATA_PATH,
                                                     VALIDATION_OUTPUT_DATA_PATH, data_set_size)
                else:
                    self.cropper.crop_image_randomly(image_id, original_image, train_mask, TRAIN_INPUT_DATA_PATH,
                                                     TRAIN_OUTPUT_DATA_PATH, data_set_size)
                    if self.cropper.total_objects_num == 0:
                        self.cropper.total_objects_num = 1
                    print(self.cropper.total_objects_num, "  ", self.cropper.total_objects_ratio, "  ",
                          (self.cropper.total_objects_ratio / self.cropper.total_objects_num))

    def __load_grid_sizes(self, image_id):
        for _im_id, _x, _y in csv.reader(open(GRID_SIZES_PATH)):
            if _im_id == image_id:
                return (float(_x), float(_y))

    def __get_scalers(self, im_size, x_max, y_min):
        h, w = im_size  # they are flipped so that mask_for_polygons works correctly
        w_ = w * (w / (w + 1))
        h_ = h * (h / (h + 1))
        return w_ / x_max, h_ / y_min

    def __load_train_polygons(self, image_id, object_type):
        for _im_id, _poly_type, _poly in csv.reader(open(TRAIN_POLYGONS_PATH)):
            if _im_id == image_id and int(_poly_type) == object_type:
                train_polygons = shapely.wkt.loads(_poly)
                return train_polygons

    # build image mask by polygons
    def __mask_for_polygons(self, polygons, image_size):
        img_mask = np.zeros(image_size, np.uint8)
        if not polygons:
            return img_mask
        int_coords = lambda x: np.array(x).round().astype(np.int32)
        exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
        interiors = [int_coords(pi.coords) for poly in polygons
                     for pi in poly.interiors]
        cv2.fillPoly(img_mask, exteriors, 1)
        cv2.fillPoly(img_mask, interiors, 0)
        return img_mask

