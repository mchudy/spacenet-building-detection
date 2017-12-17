import tensorflow as tf
import pandas as pd
from osgeo import gdal, osr, ogr
import numpy as np
import json
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from tqdm import trange
import datetime
import time
import math
import io
import os
from unet import Network


dataset = 'AOI_1_RIO'
script_path = os.path.dirname(os.path.realpath(__file__))
number_of_images = 6940


def rel_path(path):
    return os.path.join(script_path, path)


def get_mask_image_path(img_number):
    return rel_path(f'../data/rio/masks/AOI_1_RIO_img{img_number}_mask_visible.tif')


def get_3band_image_path(img_number, scaled=False):
    if scaled:
        return rel_path(f'../data/rio/scaled/3band_AOI_1_RIO_img{img_number}.tif')
    return rel_path(f'../data/rio/3band/3band_AOI_1_RIO_img{img_number}.tif')


def get_8band_image_path(img_number, scaled=False):
    if scaled:
        return rel_path(f'../data/rio/scaled/8band_AOI_1_RIO_img{img_number}.tif')
    return rel_path(f'../data/rio/8band/8band_AOI_1_RIO_img{img_number}.tif')


def rescale_images():
    os.makedirs(rel_path('../data/rio/scaled'), exist_ok=True)

    for i in trange(1, number_of_images + 1):
        img_file = get_3band_image_path(i)
        image_3band = gdal.Open(img_file)
        gdal.Warp(rel_path(f'../data/rio/scaled/3band_AOI_1_RIO_img{i}.tif'), image_3band, width=Network.IMAGE_WIDTH, height=Network.IMAGE_HEIGHT)

        img_file = get_8band_image_path(i)
        image_8band = gdal.Open(img_file)
        gdal.Warp(rel_path(f'../data/rio/scaled/8band_AOI_1_RIO_img{i}.tif'), image_8band, width=Network.IMAGE_WIDTH, height=Network.IMAGE_HEIGHT)


def create_building_mask(rasterSrc, vectorSrc, npDistFileName='',
                            noDataValue=0, burn_values=1):

    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    srcRas_ds = gdal.Open(rasterSrc)
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize

    memdrv = gdal.GetDriverByName('GTiff')
    dst_ds = memdrv.Create(npDistFileName, cols, rows, 1, gdal.GDT_Byte,
                           options=['COMPRESS=LZW'])
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(noDataValue)
    gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[burn_values])
    dst_ds = 0


def generate_masks():
    os.makedirs(rel_path('../data/rio/masks'), exist_ok=True)

    for i in trange(1, number_of_images + 1):
        img_file = get_3band_image_path(i, scaled=True)
        geojson_file = rel_path(f'../data/rio/vectordata/geojson/Geo_AOI_1_RIO_img{i}.geojson')
        visible_mask_file = rel_path(f'../data/rio/masks/AOI_1_RIO_img{i}_mask_visible.tif')
        create_building_mask(img_file, geojson_file, npDistFileName=visible_mask_file, burn_values=255)


def preprocess_data():
    rescale_images()
    generate_masks()


if __name__ == '__main__':
    preprocess_data()
