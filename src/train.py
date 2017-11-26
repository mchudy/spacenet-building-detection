import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import pandas as pd
from osgeo import gdal, osr, ogr
import numpy as np
import json
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from tqdm import trange

dataset = 'AOI_1_RIO'
script_path = os.path.dirname(os.path.realpath(__file__))
number_of_images = 6940


def rel_path(path):
    return os.path.join(script_path, path)


def latlon2pixel(lat, lon, input_raster='', targetsr='', geom_transform=''):
    sourcesr = osr.SpatialReference()
    sourcesr.ImportFromEPSG(4326)

    geom = ogr.Geometry(ogr.wkbPoint)
    geom.AddPoint(lon, lat)

    if targetsr == '':
        src_raster = gdal.Open(input_raster)
        targetsr = osr.SpatialReference()
        targetsr.ImportFromWkt(src_raster.GetProjectionRef())
    coord_trans = osr.CoordinateTransformation(sourcesr, targetsr)
    if geom_transform == '':
        src_raster = gdal.Open(input_raster)
        transform = src_raster.GetGeoTransform()
    else:
        transform = geom_transform

    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]
    geom.Transform(coord_trans)
    x_pix = (geom.GetPoint()[0] - x_origin) / pixel_width
    y_pix = (geom.GetPoint()[1] - y_origin) / pixel_height

    return (x_pix, y_pix)


def geojson_to_pixel_arr(raster_file, geojson_file, pixel_ints=True):

    with open(geojson_file) as f:
        geojson_data = json.load(f)

    src_raster = gdal.Open(raster_file)
    targetsr = osr.SpatialReference()
    targetsr.ImportFromWkt(src_raster.GetProjectionRef())

    geom_transform = src_raster.GetGeoTransform()

    latlons = []
    types = []
    for feature in geojson_data['features']:
        coords_tmp = feature['geometry']['coordinates'][0]
        type_tmp = feature['geometry']['type']
        latlons.append(coords_tmp)
        types.append(type_tmp)

    pixel_coords = []
    latlon_coords = []
    for i, (poly_type, poly0) in enumerate(zip(types, latlons)):

        if poly_type.upper() == 'MULTIPOLYGON':
            for poly in poly0:
                poly=np.array(poly)

                if len(poly.shape) == 3 and poly.shape[0] == 1:
                    poly = poly[0]

                poly_list_pix = []
                poly_list_latlon = []
                for coord in poly:
                    lon, lat, z = coord
                    px, py = latlon2pixel(lat, lon, input_raster=src_raster,
                                         targetsr=targetsr,
                                         geom_transform=geom_transform)
                    poly_list_pix.append([px, py])
                    poly_list_latlon.append([lat, lon])

                if pixel_ints:
                    ptmp = np.rint(poly_list_pix).astype(int)
                else:
                    ptmp = poly_list_pix
                pixel_coords.append(ptmp)
                latlon_coords.append(poly_list_latlon)

        elif poly_type.upper() == 'POLYGON':
            poly=np.array(poly0)

            if len(poly.shape) == 3 and poly.shape[0] == 1:
                poly = poly[0]

            poly_list_pix = []
            poly_list_latlon = []
            for coord in poly:
                lon, lat, z = coord
                px, py = latlon2pixel(lat, lon, input_raster=src_raster,
                                     targetsr=targetsr,
                                     geom_transform=geom_transform)
                poly_list_pix.append([px, py])
                poly_list_latlon.append([lat, lon])

            if pixel_ints:
                ptmp = np.rint(poly_list_pix).astype(int)
            else:
                ptmp = poly_list_pix
            pixel_coords.append(ptmp)
            latlon_coords.append(poly_list_latlon)

        else:
            print("Unknown shape type in coords_arr_from_geojson()")
            return

    return pixel_coords, latlon_coords


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

    return


def plot_truth_coords(input_image, mask_image, pixel_coords,
                  figsize=(8,8),
                  poly_face_color='orange',
                  poly_edge_color='red', poly_nofill_color='blue', cmap='bwr'):

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(2*figsize[0], figsize[1]))

    patches = []
    patches_nofill = []
    if len(pixel_coords) > 0:
        for coord in pixel_coords:
            patches_nofill.append(Polygon(coord, facecolor=poly_nofill_color,
                                          edgecolor=poly_edge_color, lw=3))
            patches.append(Polygon(coord, edgecolor=poly_edge_color, fill=True,
                                   facecolor=poly_face_color))
        p0 = PatchCollection(patches, alpha=0.50, match_original=True)
        p2 = PatchCollection(patches_nofill, alpha=0.75, match_original=True)

    ax0.imshow(input_image)
    if len(patches) > 0:
        ax0.add_collection(p0)

    ax1.imshow(mask_image, cmap=cmap)

    plt.tight_layout()
    plt.show()

    return patches, patches_nofill


def get_3band_image_path(img_number):
    return rel_path(f'../data/rio/3band/3band_AOI_1_RIO_img{img_number}.tif')


def get_8band_image_path(img_number):
    return rel_path(f'../data/rio/8band/8band_AOI_1_RIO_img{img_number}.tif')


def generate_masks():
    os.makedirs(rel_path('../data/rio/masks'), exist_ok=True)

    for i in trange(1, number_of_images + 1):
        img_file = get_3band_image_path(i)
        geojson_file = rel_path(f'../data/rio/vectordata/geojson/Geo_AOI_1_RIO_img{i}.geojson')
        mask_file = rel_path(f'../data/rio/masks/AOI_1_RIO_img{i}_mask.tif')
        visible_mask_file = rel_path(f'../data/rio/masks/AOI_1_RIO_img{i}_mask_visible.tif')

        create_building_mask(img_file, geojson_file, npDistFileName=mask_file, burn_values=1)
        create_building_mask(img_file, geojson_file, npDistFileName=visible_mask_file, burn_values=255)


def convert_geotiff_to_array(path):
    mul_ds = gdal.Open(path)
    channels = mul_ds.RasterCount
    mul_img = np.zeros((mul_ds.RasterXSize, mul_ds.RasterYSize, channels), dtype='float')
    #geoTf   = np.asarray(mul_ds.GetGeoTransform())

    for band in range(0, channels):
        mul_img[:,:,band] = mul_ds.GetRasterBand(band+1).ReadAsArray().transpose().astype(float) / 255.0

    return mul_img


if __name__ == '__main__':

    band3_images_path = rel_path('../data/rio/3band')
    band8_images_path = rel_path('../data/rio/8band')

    #generate_masks()
    image_array = convert_geotiff_to_array(get_3band_image_path(30))
    print(image_array)
    #patches, patches_no_fill = geojson_to_pixel_arr(img_file, geojson_file)
    #plot_truth_coords(plt.imread(img_file), plt.imread(mask_file), patches)


#band3_solution_path = rel_path('../data/rio/vectordata/summarydata/AOI_1_RIO_polygons_solution_3band.csv')
#band8_solution_path = rel_path('../data/rio/vectordata/summarydata/AOI_1_RIO_polygons_solution_8band.csv')
#ImageId,BuildingId,PolygonWKT_Pix,PolygonWKT_Geo
#band3_solution_df = pd.read_csv(band3_solution_path)
#band8_solution_df = pd.read_csv(band8_solution_path)

#ds8 = gdal.Open(band8_images_path+ '/8band_' + dataset + '_img' + str(1) + '.tif')
#print(ds8)

#x8band = tf.placeholder(tf.float32, shape=[None, FLAGS.ws, FLAGS.ws, 8])
#x3band = tf.placeholder(tf.float32, shape=[None, scale * FLAGS.ws, scale * FLAGS.ws, 3])


