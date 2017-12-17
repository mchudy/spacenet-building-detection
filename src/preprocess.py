import tensorflow as tf
from osgeo import gdal, osr, ogr
import numpy as np
import json
from tqdm import trange
import os
from unet import Network


dataset = 'AOI_1_RIO'
script_path = os.path.dirname(os.path.realpath(__file__))
number_of_images = 6940


def rel_path(path):
    return os.path.join(script_path, path)


def get_mask_image_path(img_number):
    return rel_path(f'../data/rio/masks/AOI_1_RIO_img{img_number}_mask.tif')


def get_3band_image_path(img_number, scaled=False):
    if scaled:
        return rel_path(f'../data/rio/scaled/3band_AOI_1_RIO_img{img_number}.tif')
    return rel_path(f'../data/rio/3band/3band_AOI_1_RIO_img{img_number}.tif')


def get_8band_image_path(img_number, scaled=False):
    if scaled:
        return rel_path(f'../data/rio/scaled/8band_AOI_1_RIO_img{img_number}.tif')
    return rel_path(f'../data/rio/8band/8band_AOI_1_RIO_img{img_number}.tif')


def is_image_incomplete(image):
    incomplete = True
    for i in range(1, 4):
        band = image.GetRasterBand(i)
        band_array = band.ReadAsArray()
        size = band.XSize * band.YSize
        zeros_count = np.count_nonzero(band_array==0)
        if zeros_count < size / 2:
            incomplete = False

    return incomplete


def filter_incomplete_images(perm):
    print('Filtering incomplete images...')

    result = []
    for i in trange(len(perm)):
        current_image_no = perm[i]
        img_file = get_3band_image_path(current_image_no)
        image_3band = gdal.Open(img_file)
        if not is_image_incomplete(image_3band):
            result.append(current_image_no)

    print(f'{len(result)}/{len(perm)} images left')
    return result


def rescale_images(perm):
    print('Scaling images...')
    os.makedirs(rel_path('../data/rio/scaled'), exist_ok=True)

    for i in trange(len(perm)):
        img_no = perm[i]
        img_file = get_3band_image_path(img_no)

        image_3band = gdal.Open(img_file)
        is_image_incomplete(image_3band)
        gdal.Warp(rel_path(f'../data/rio/scaled/3band_AOI_1_RIO_img{i}.tif'), image_3band, width=Network.IMAGE_WIDTH, height=Network.IMAGE_HEIGHT)

        img_file = get_8band_image_path(img_no)
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


def generate_masks(perm):
    print('Generating masks...')
    os.makedirs(rel_path('../data/rio/masks'), exist_ok=True)

    for i in trange(len(perm)):
        image_no = perm[i]
        img_file = get_3band_image_path(i, scaled=True)
        geojson_file = rel_path(f'../data/rio/vectordata/geojson/Geo_AOI_1_RIO_img{image_no}.geojson')
        mask_file = rel_path(f'../data/rio/masks/AOI_1_RIO_img{i}_mask.tif')
        create_building_mask(img_file, geojson_file, npDistFileName=mask_file, burn_values=255)


def preprocess_data():
    np.random.seed(5)
    perm = np.random.permutation(range(1, number_of_images + 1))
    perm = filter_incomplete_images(perm)
    rescale_images(perm)
    generate_masks(perm)


if __name__ == '__main__':
    preprocess_data()
