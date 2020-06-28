import cv2
from PIL import Image, ImageOps
import numpy as np
import sys
import matplotlib.pyplot as plt

import imageio

# TODO REFACTOR

def place_gt_onto_source_image_0(source_path: str, gt_path):
    img = cv2.imread(source_path)
    img_gt = cv2.imread(gt_path)
    img_gt_invert = cv2.bitwise_not(img_gt)
    res= np.where(img_gt == (0, 0, 0), img,(0,0,200))

    pic = imageio.imread(source_path)
    arr = img_gt == (255, 255, 255)
    arr[:,:,0] = 0
    arr[:, :, 1] = 0
    # arr[:, :, 2] = 2
    img[arr] = 255  # full intensity to those pixel's R channel
    cv2.imwrite('/home/ales/programming/satellite-images-segmentation/data/test/opencv_bitwise_and.jpg', img)


    # # dst = cv2.bitwise_and(img, img_gt_invert)
    # cv2.imwrite('/home/ales/programming/satellite-images-segmentation/data/test/opencv_bitwise_and.jpg', res)
    #
    #
    # Conv_hsv_Gray = cv2.cvtColor(img_gt_invert, cv2.COLOR_BGR2GRAY)
    #
    # res, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #
    # img_gt_invert[mask == 255] = [0,128,0]
    #
    # cv2.imwrite('/home/ales/programming/satellite-images-segmentation/data/test/opencv_bitwise_and1.jpg', img_gt_invert)
    # dst = cv2.bitwise_and(img, img_gt_invert)
    # cv2.imwrite('/home/ales/programming/satellite-images-segmentation/data/test/opencv_bitwise_and2.jpg', dst)



def place_gt_onto_source_image(source_path: str, gt_path):
    picture = Image.open(gt_path)
    # img_gt_invert = ImageOps.invert(im)
    # img_gt_invert.save('/home/ales/programming/satellite-images-segmentation/data/test/temp.jpg', quality=95)

    img = picture.convert("RGBA")

    pixdata = img.load()

    # Clean the background noise, if color != white, then set to black.

    for y in range(0, img.size[1]):
        for x in range(0, img.size[0]):
            if pixdata[x, y] == (255, 100, 255, 300):
                pixdata[x, y] = (0, 0, 0, 255)
    img.save('/home/ales/programming/satellite-images-segmentation/data/test/opencv_bitwise_and.rgb', 'RGBA')

    # cv2.imwrite('/home/ales/programming/satellite-images-segmentation/data/test/opencv_bitwise_and.jpg', picture)


    # img = cv2.imread(source_path)
    # img_gt = cv2.imread('/home/ales/programming/satellite-images-segmentation/data/test/temp.jpg')
    #
    # dst = cv2.bitwise_and(img, img_gt)
    #
    #
    # cv2.imwrite('/home/ales/programming/satellite-images-segmentation/data/test/opencv_bitwise_and.jpg', dst)

    # print(img.shape)
    # print(img_gt.shape)

def place():
    pic = imageio.imread('/home/ales/programming/satellite-images-segmentation/data/test/bellingham2.tif')

    pic[1000:1500, :, 2] = 255  # full intensity to those pixel's R channel
    plt.imsave('/home/ales/programming/satellite-images-segmentation/data/test/opencv_bitwise_and000.jpg', pic)
    # plt.show()

place()

place_gt_onto_source_image_0(
    '/home/ales/programming/satellite-images-segmentation/data/test/bellingham2.tif',
    '/home/ales/programming/satellite-images-segmentation/data/test/predict/bellingham2.tif'
)
