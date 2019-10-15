import csv
import json
import os

import cv2
import gdal
import numpy as np
import shapely
import shapely.wkt

import tifffile as tiff


from app.config.main_config import BUILDINGS_DATA_PATH, IMAGE_FORMAT, BUILDINGS_MASK_DATA_PATH, \
    VALIDATION_INPUT_DATA_PATH, TRAIN_INPUT_DATA_PATH, TRAIN_POLYGONS_PATH, GRID_SIZES_PATH, \
    VALIDATION_OUTPUT_DATA_PATH, TRAIN_OUTPUT_DATA_PATH
from app.domain.img_augumentations import ImageRotationService, ImageFlipService
from app.domain.img_models import DatasetPreparationSettings
from app.domain.img_uploadings import LocalImageSaveService
from app.preparation.img_cropper import RandomImageCropper


class InriaDatasetPreparer():

    def __init__(self,
                 data_path: str,
                 ground_truth_data_path: str,
                 train_dataset_path: str,
                 validation_dataset_path: str,
                 dataset_settings: DatasetPreparationSettings):
        self.data_path = data_path
        self.ground_truth_data_path = ground_truth_data_path
        self.train_dataset_path = train_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.dataset_settings = dataset_settings

        self.image_cropper = RandomImageCropper()
        self.image_rotator = ImageRotationService()
        self.image_flipper = ImageFlipService()
        self.image_saver = LocalImageSaveService()

    def prepare_dataset_for_training(self) -> None:
        self.__create_directory_for_masks_images(self.train_dataset_path)
        filenames = os.listdir(self.data_path)

        for f in filenames:
            if f.endswith(IMAGE_FORMAT):
                image_id = f[:-4]
                original_image = tiff.imread((self.data_path + '{}' + IMAGE_FORMAT).format(image_id))
                image_mask = tiff.imread((self.ground_truth_data_path + '{}' + IMAGE_FORMAT).format(image_id)) / 255

                cropped_images_list = self.image_cropper.crop(image_id, original_image, image_mask,
                                                              self.dataset_settings.segments_num_for_each_image)
                if self.dataset_settings.flip is True:
                    flipped_images = self.image_flipper.flip_list(cropped_images_list)
                    cropped_images_list.extend(flipped_images)

                if self.dataset_settings.rotate > 0:
                    rotated_images = self.image_rotator.rotate_list(cropped_images_list, self.dataset_settings.rotate)
                    cropped_images_list.extend(rotated_images)

            for source_and_mask in cropped_images_list:
                self.image_saver.save(source_and_mask.source, self.train_dataset_path)
                self.image_saver.save(source_and_mask.mask, self.train_dataset_path + "mask\\")

    def __create_directory_for_masks_images(self, train_directory: str) -> None:
        mask_directory_path = train_directory + "mask\\"
        if not os.path.exists(mask_directory_path):
            os.makedirs(mask_directory_path)



# todo rewrite class to work in new object domain
class ChineseDatasetPreparer():
    def create_data_chinese_images(self, data_set_size, is_validation=False):
        num_images_added_to_train_set = 0
        for f in self.filenames:
            if f.endswith(IMAGE_FORMAT):
                image_id = f[:-4]
                original_image = tiff.imread((self.samples_path + '{}' + IMAGE_FORMAT).format(image_id))
                image_mask = self.get_mask_for_chinese_images(
                    (self.samples_path + '{}' + IMAGE_FORMAT).format(image_id),
                    (self.samples_geojson_path + '{}' + ".geojson").format(image_id))

                if is_validation and num_images_added_to_train_set == self.TRAIN_IMAGES_TO_VALIDATION_RATIO:
                    num_images_added_to_train_set = 0

                    self.cropper.crop_image_randomly(image_id, original_image, image_mask, VALIDATION_INPUT_DATA_PATH,
                                                     VALIDATION_OUTPUT_DATA_PATH,
                                                     data_set_size, 90)
                else:
                    self.cropper.crop_image_randomly(image_id, original_image, image_mask, TRAIN_INPUT_DATA_PATH,
                                                     TRAIN_OUTPUT_DATA_PATH,
                                                     data_set_size, 90, make_flip=True)
                    num_images_added_to_train_set += 1

    def get_mask_for_chinese_images(self, image_path, image_geojson_path):
        ds = gdal.Open(image_path)
        width = ds.RasterXSize
        height = ds.RasterYSize
        xmin = ds.GetGeoTransform()[0]
        ymin = ds.GetGeoTransform()[3]
        step = 0.2
        ymax = ymin - height * step
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
        # tiff.imshow(mask)
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

# todo rewrite class to work in new object domain
class DSTLDatasetPreparer():
    def prepare_dataset_for_training(self, data_set_size, is_validation=False):
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







