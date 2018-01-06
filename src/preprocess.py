import tensorflow as tf
from osgeo import gdal, osr, ogr, gdalnumeric
import numpy as np
import json
from tqdm import trange
import os
from unet import Network
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import sys


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGES_COUNT = 6940


def rel_path(path):
    return os.path.join(SCRIPT_PATH, path)


path_to_spacenet_utils = rel_path('../utilities/python')
sys.path.extend([path_to_spacenet_utils])
from spaceNetUtilities import geoTools as gT


def create_dist_map(rasterSrc, vectorSrc, npDistFileName='',
                           noDataValue=0, burn_values=1,
                           dist_mult=1, vmax_dist=64):

    '''
    Create building signed distance transform from Yuan 2016
    (https://arxiv.org/pdf/1602.06564v1.pdf).
    vmax_dist: absolute value of maximum distance (meters) from building edge
    Adapted from createNPPixArray in labeltools
    '''

    ## open source vector file that truth data
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    ## extract data from src Raster File to be emulated
    ## open raster file that is to be emulated
    srcRas_ds = gdal.Open(rasterSrc)
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize

    geoTrans, poly, ulX, ulY, lrX, lrY = gT.getRasterExtent(srcRas_ds)
    transform_WGS84_To_UTM, transform_UTM_To_WGS84, utm_cs \
                                        = gT.createUTMTransform(poly)
    line = ogr.Geometry(ogr.wkbLineString)
    line.AddPoint(geoTrans[0], geoTrans[3])
    line.AddPoint(geoTrans[0]+geoTrans[1], geoTrans[3])

    line.Transform(transform_WGS84_To_UTM)
    metersIndex = line.Length()

    memdrv = gdal.GetDriverByName('MEM')
    dst_ds = memdrv.Create('', cols, rows, 1, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(noDataValue)

    gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[burn_values])
    srcBand = dst_ds.GetRasterBand(1)

    memdrv2 = gdal.GetDriverByName('MEM')
    prox_ds = memdrv2.Create('', cols, rows, 1, gdal.GDT_Int16)
    prox_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    prox_ds.SetProjection(srcRas_ds.GetProjection())
    proxBand = prox_ds.GetRasterBand(1)
    proxBand.SetNoDataValue(noDataValue)

    opt_string = 'NODATA='+str(noDataValue)
    options = [opt_string]

    gdal.ComputeProximity(srcBand, proxBand, options)

    memdrv3 = gdal.GetDriverByName('MEM')
    proxIn_ds = memdrv3.Create('', cols, rows, 1, gdal.GDT_Int16)
    proxIn_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    proxIn_ds.SetProjection(srcRas_ds.GetProjection())
    proxInBand = proxIn_ds.GetRasterBand(1)
    proxInBand.SetNoDataValue(noDataValue)
    opt_string2 = 'VALUES='+str(noDataValue)
    options = [opt_string, opt_string2]
    #options = ['NODATA=0', 'VALUES=0']

    gdal.ComputeProximity(srcBand, proxInBand, options)

    proxIn = gdalnumeric.BandReadAsArray(proxInBand)
    proxOut = gdalnumeric.BandReadAsArray(proxBand)

    proxTotal = proxIn.astype(float) - proxOut.astype(float)
    proxTotal = proxTotal*metersIndex
    proxTotal *= dist_mult

    # clip array
    proxTotal = np.clip(proxTotal, -1*vmax_dist, 1*vmax_dist)

    if npDistFileName != '':
        # save as numpy file since some values will be negative
        np.save(npDistFileName, proxTotal)
        #cv2.imwrite(npDistFileName, proxTotal)


def plot_dist_transform(input_image, pixel_coords, dist_image,
                        figsize=(8,8), plot_name='', add_title=False,
                        colorbar=True,
                        poly_face_color='orange', poly_edge_color='red',
                        poly_nofill_color='blue', cmap='bwr'):
    '''Explore distance transform'''

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                        figsize=(3*figsize[0], figsize[1]))

    #fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(2*figsize[0], figsize[1]))
    mind, maxd = np.round(np.min(dist_image),2), np.round(np.max(dist_image),2)

    if add_title:
        suptitle = fig.suptitle(plot_name.split('/')[-1], fontsize='large')

    # create patches
    patches = []
    patches_nofill = []
    if len(pixel_coords) > 0:
        # get patches
        for coord in pixel_coords:
            patches_nofill.append(Polygon(coord, facecolor=poly_nofill_color,
                                          edgecolor=poly_edge_color, lw=3))
            patches.append(Polygon(coord, edgecolor=poly_edge_color, fill=True,
                                   facecolor=poly_face_color))
        p0 = PatchCollection(patches, alpha=0.25, match_original=True)
        p1 = PatchCollection(patches, alpha=0.75, match_original=True)
        #p2 = PatchCollection(patches_nofill, alpha=0.75, match_original=True)

    #if len(patches) > 0:
    #    p0 = PatchCollection(patches, alpha=0.25, match_original=True)
    #    p1 = PatchCollection(patches, alpha=0.75, match_original=True)


    # ax0: raw image
    ax0.imshow(input_image)
    if len(patches) > 0:
        ax0.add_collection(p0)
    ax0.set_title('Input Image + Ground Truth Buildings')

    ## truth polygons
    #zero_arr = np.zeros(input_image.shape[:2])
    ## set background to white?
    ##zero_arr[zero_arr == 0.0] = np.nan
    #ax1.imshow(zero_arr, cmap=cmap)
    #if len(patches) > 0:
    #    ax1.add_collection(p1)
    #ax1.set_title('Ground Truth Building Outlines')

    # transform
    cbar_pointer = ax1.imshow(dist_image)
    dist_suffix = " (min=" + str(mind) + ", max=" + str(maxd) + ")"
    ax1.set_title("Yuan 2016 Distance Transform" + dist_suffix)

    # overlay buildings on distance transform
    ax2.imshow(dist_image)
    # truth polygons
    if len(patches) > 0:
        ax2.add_collection(p1)
    # truth mask
    #ax2.imshow(z, cmap=palette, alpha=0.5,
    #       norm=matplotlib.colors.Normalize(vmin=0.5, vmax=0.9, clip=False))
    ax2.set_title("Ground Truth Polygons Overlaid on Distance Transform")

    if colorbar:
        #from mpl_toolkits.axes_grid1 import make_axes_locatable
        #divider = make_axes_locatable(ax2)
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        #fig.colorbar(cbar_pointer, cax=cax, orientation='vertical')
        left, bottom, width, height = [0.38, 0.85, 0.24, 0.03]
        cax = fig.add_axes([left, bottom, width, height])
        fig.colorbar(cbar_pointer, cax=cax, orientation='horizontal')

    #plt.axis('off')
    plt.tight_layout()
    if add_title:
        suptitle.set_y(0.95)
        fig.subplots_adjust(top=0.96)
    #plt.show()

    if len(plot_name) > 0:
        plt.savefig(plot_name)


def get_mask_image_path(img_number):
    return rel_path(f'../data/rio/masks/AOI_1_RIO_img{img_number}_mask.tif')


def get_distance_transform_image_path(img_number):
    return rel_path(f'../data/rio/dist_transforms/AOI_1_RIO_img{img_number}_mask.tif')


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


def generate_distance_transforms(perm):
    print('Generating distance transforms...')
    os.makedirs(rel_path('../data/rio/dist_transforms'), exist_ok=True)

    for i in trange(len(perm)):
        image_no = perm[i]
        img_file = get_3band_image_path(i, scaled=True)
        geojson_file = rel_path(f'../data/rio/vectordata/geojson/Geo_AOI_1_RIO_img{image_no}.geojson')
        mask_file = rel_path(f'../data/rio/dist_transforms/AOI_1_RIO_img{i}_mask.tif')
        create_dist_map(img_file, geojson_file, npDistFileName=mask_file, burn_values=255)


def preprocess_data():
    np.random.seed(5)
    perm = np.random.permutation(range(1, IMAGES_COUNT + 1))
    perm = filter_incomplete_images(perm)
    rescale_images(perm)
    generate_masks(perm)
    generate_distance_transforms(perm)
    #plot_dist_transform(get_3band_image_path(1), [], get_distance_transform_image_path(1))


if __name__ == '__main__':
    preprocess_data()
