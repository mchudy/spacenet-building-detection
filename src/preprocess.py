import tensorflow as tf
from osgeo import gdal, osr, ogr, gdalnumeric
import numpy as np
import json
from tqdm import trange
import os
from unet import Network
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from greedy_clustering import FindAllClusters
import matplotlib.pyplot as plt
import sys
import pandas as pd
import cv2
import skimage.draw
import skimage.io
import re
from skimage import measure


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGES_COUNT = 6940


def rel_path(path):
    return os.path.join(SCRIPT_PATH, path)


path_to_spacenet_utils = rel_path('../utilities/python')
sys.path.extend([path_to_spacenet_utils])
from spaceNetUtilities import geoTools as gT


def convert_geotiff_to_array(image_number, scaled=False):
    image_3band = gdal.Open(get_3band_image_path(image_number, scaled))
    channels = image_3band.RasterCount
    mul_img = np.zeros((image_3band.RasterXSize, image_3band.RasterYSize, channels), dtype='float')

    for band in range(0, image_3band.RasterCount):
        mul_img[:,:,band] = image_3band.GetRasterBand(band+1).ReadAsArray().astype(float) / 255.0

    return mul_img


def convert_target_to_array(image_number):
    image = gdal.Open(get_mask_image_path(image_number))
    channels = image.RasterCount
    mul_img = np.zeros((image.RasterXSize, image.RasterYSize, channels), dtype='float')
    mul_img[:,:, 0] = image.GetRasterBand(1).ReadAsArray().astype(float) / 255.0

    return mul_img



def create_dist_map(rasterSrc, vectorSrc, npDistFileName='',
                           noDataValue=0, burn_values=255,
                           dist_mult=1, vmax_dist=64):

    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

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

    gdal.ComputeProximity(srcBand, proxInBand, options)

    proxIn = gdalnumeric.BandReadAsArray(proxInBand)
    proxOut = gdalnumeric.BandReadAsArray(proxBand)

    proxTotal = proxIn.astype(float) - proxOut.astype(float)
    proxTotal = proxTotal*metersIndex
    proxTotal *= dist_mult

    proxTotal = np.clip(proxTotal, -1*vmax_dist, 1*vmax_dist)

    if npDistFileName != '':
        np.save(npDistFileName, proxTotal)


def plot_dist_transform(input_image, dist_image, figsize=(8,8), plot_name='',  mask_image=''):

    _, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(3*figsize[0], figsize[1]))

    ax0.imshow(input_image[:,:,:3])
    ax0.set_title('Input image')
    ax1.set_title('Ground truth')
    ax1.imshow(mask_image[:,:,0], cmap='gray')
    ax2.imshow(dist_image)
    ax2.set_title("Distance Transform")
    plt.show()

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


def rescale_image(source_image, target_image):
    os.system(f'gdalwarp -ts 256 256 -r cubic -overwrite {source_image} {target_image} > /dev/null')


def rescale_images(perm):
    print('Scaling images...')
    os.makedirs(rel_path('../data/rio/scaled'), exist_ok=True)

    for i in trange(len(perm)):
        img_no = perm[i]
        rescale_image(get_3band_image_path(img_no), rel_path(f'../data/rio/scaled/3band_AOI_1_RIO_img{i}.tif'))
        rescale_image(get_8band_image_path(img_no), rel_path(f'../data/rio/scaled/8band_AOI_1_RIO_img{i}.tif'))


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
        create_dist_map(img_file, geojson_file, npDistFileName=mask_file, vmax_dist=32)


def CreateGeoJSON ( fn, cluster ):
    os.makedirs(rel_path('../output/geojson'), exist_ok=True)

    memdrv = gdal.GetDriverByName ('MEM')
    src_ds = memdrv.Create('',cluster.shape[1],cluster.shape[0],1)
    src_ds.SetGeoTransform([0, 1, 0, 0, 0, 1])
    band = src_ds.GetRasterBand(1)
    band.WriteArray(cluster)
    dst_layername = "BuildingID"
    drv = ogr.GetDriverByName("geojson")

    if os.path.exists(rel_path('../output/geojson/' + fn  + ".geojson")):
        drv.DeleteDataSource(rel_path('../output/geojson/' + fn + ".geojson"))

    dst_ds = drv.CreateDataSource ( rel_path('../output/geojson/' + fn + ".geojson"))
    dst_layer = dst_ds.CreateLayer( dst_layername, srs=None )
    dst_layer.CreateField( ogr.FieldDefn("DN", ogr.OFTInteger) )
    gdal.Polygonize( band  , None, dst_layer, 0, ['8CONNECTED=8'], callback=None )


def FixGeoJSON( fn ):
    buf_dist = 0.0
    dst_layername = "BuildingID"
    drv = ogr.GetDriverByName("geojson")
    dst_ds = drv.Open ( rel_path('../output/geojson/' + fn + ".geojson"))
    dst_layer = dst_ds.GetLayer(0)
    if os.path.exists(rel_path('../output/geojson/buffer' + fn + ".geojson")):
        drv.DeleteDataSource(rel_path('../output/geojson/buffer' + fn + ".geojson"))
    adst_ds = drv.CreateDataSource ( rel_path('../output/geojson/buffer' + fn + ".geojson"))
    adst_layer = adst_ds.CreateLayer( dst_layername, srs=None )
    adst_layer.CreateField( ogr.FieldDefn("DN", ogr.OFTInteger) )

    for i in range(dst_layer.GetFeatureCount()):
        f = dst_layer.GetFeature(i)
        clusternumber = f.GetField("DN")
        f.SetGeometry(f.GetGeometryRef().Buffer(buf_dist))
        if 0 == f.GetField("DN"):
            dst_layer.DeleteFeature(i) #not supported by geoJSON driver now
        else:
            adst_layer.CreateFeature(f)


def ParseGeoJSON( fn, perm,transpose=False ):
    with open(rel_path('../output/geojson/' + fn)) as f:
        polygon_list = json.load(f)['features']
        if len(polygon_list) == 0:
            yield '{},-1,POLYGON EMPTY,1'.format(fn)
        else:
            img_shape = (256, 256)
            check_img = np.zeros(img_shape)
            for polygon in polygon_list:
                dn = polygon['properties']['DN']
                coords_raw = polygon['geometry']['coordinates'][0]
                if isinstance(coords_raw[0][0], (list,)):
                    continue
                pp = [
                        [int(p[1]) for p in coords_raw],
                        [int(p[0]) for p in coords_raw]
                    ]
                rr, cc = skimage.draw.polygon(pp, img_shape)
                check_img[rr,cc] = 1
                if transpose:
                    coords = ','.join((str(co[1]*440/256) + ' ' + str(co[0]*408/256) + ' 0' for co in coords_raw))
                else:
                    coords = ','.join((str(co[0]*440/256) + ' ' + str(co[1]*408/256) + ' 0' for co in coords_raw))
                image_name = fn[6:-8]
                match = re.search('img(\\d+)$', image_name)
                img_no = int(match.groups()[0])
                img_no = perm[img_no]
                image_name = f'AOI_1_RIO_img{img_no}'
                yield '{},{},"POLYGON (({}))",{}'.format(image_name, dn, coords, dn)


def merge_results(path, out_filepath, perm, transpose=False):
    with open(out_filepath, 'w') as fw:
        fw.write('ImageId,BuildingId,PolygonWKT_Pix,Confidence\n')
        for fn in [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and 'buffer' in f]:
            lines = ParseGeoJSON(fn, perm, transpose)
            for li in lines: fw.write(li + '\n')


def create_truth_csv(imgs):
    truth_df = pd.read_csv(rel_path('../data/rio/vectordata/summarydata/AOI_1_RIO_polygons_solution_3band.csv'))
    with open(rel_path('../output/geojson/truth.csv'), 'w') as fw:
        fw.write('ImageId,BuildingId,PolygonWKT_Pix,PolygonWKT_Geo\n')
        for img_no in imgs:
            image_id = 'AOI_1_RIO_img' + str(img_no)
            image_df = truth_df[truth_df['ImageId'] == image_id]
            image_df['ImageId'] = 'AOI_1_RIO_img' + str(img_no)
            csv = image_df.to_csv(index=False, header=False)
            fw.write(csv)


def get_permutation():
    np.random.seed(5)
    perm = np.random.permutation(range(1, IMAGES_COUNT + 1))
    perm = filter_incomplete_images(perm)
    return perm


def preprocess_data():
    perm = np.load(rel_path('../results/perm.npy'))
    # perm = get_permutation()
    #rescale_images(perm)
    #generate_masks(perm)
    generate_distance_transforms(perm)

    # img_no = 0
    # # plot_dist_transform(
    # #     convert_geotiff_to_array(img_no, scaled=True),
    # #     [],
    # #     np.load(get_distance_transform_image_path(img_no) + '.npy'),
    # #     mask_image=convert_target_to_array(img_no))
    # intensity = np.load(get_distance_transform_image_path(img_no) + '.npy')
    # xmax, xmin = intensity.max(), intensity.min()
    # intensity = 2*(intensity - xmin)/(xmax - xmin) - 1
    # print(intensity)

    # cluster = measure.find_contours(intensity, 0)
    # #print(cluster)
    # cluster = FindAllClusters(intensity)
    # print(cluster)
    # CreateGeoJSON('AOI_1_RIO_img' + str(img_no), cluster)
    # FixGeoJSON('AOI_1_RIO_img' + str(img_no))
    # create_truth_csv([perm[img_no]])
    # merge_results(rel_path('../output/geojson'), rel_path('../output/geojson/result.csv'), perm)
    #print(ccc)


if __name__ == '__main__':
    preprocess_data()
