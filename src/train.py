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
from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import datetime
import time
import math
import io
import random


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


def generate_masks():
    os.makedirs(rel_path('../data/rio/masks'), exist_ok=True)

    for i in trange(1, number_of_images + 1):
        img_file = get_3band_image_path(i, scaled=True)
        geojson_file = rel_path(f'../data/rio/vectordata/geojson/Geo_AOI_1_RIO_img{i}.geojson')
        visible_mask_file = rel_path(f'../data/rio/masks/AOI_1_RIO_img{i}_mask_visible.tif')
        create_building_mask(img_file, geojson_file, npDistFileName=visible_mask_file, burn_values=255)


def convert_geotiff_to_array(image_number, scaled=False, only_rgb=False):
    image_3band = gdal.Open(get_3band_image_path(image_number, scaled))
    image_8band = gdal.Open(get_8band_image_path(image_number, scaled))
    channels = image_3band.RasterCount + image_8band.RasterCount
    mul_img = np.zeros((image_3band.RasterXSize, image_3band.RasterYSize, channels), dtype='float')

    for band in range(0, image_3band.RasterCount):
        mul_img[:,:,band] = image_3band.GetRasterBand(band+1).ReadAsArray().transpose().astype(float) / 255.0

    for band in range(0, image_8band.RasterCount):
        output_band = image_3band.RasterCount + band
        mul_img[:,:,output_band] = image_8band.GetRasterBand(band+1).ReadAsArray().transpose().astype(float) / 255.0

    return mul_img


def convert_target_to_array(image_number):
    image = gdal.Open(get_mask_image_path(image_number))
    channels = image.RasterCount
    mul_img = np.zeros((image.RasterXSize, image.RasterYSize, channels), dtype='float')
    mul_img[:,:, 0] = image.GetRasterBand(1).ReadAsArray().transpose().astype(float) / 255.0

    return mul_img


def rescale_images():
    os.makedirs(rel_path('../data/rio/scaled'), exist_ok=True)

    for i in trange(1, number_of_images + 1):
        img_file = get_3band_image_path(i)
        image_3band = gdal.Open(img_file)
        gdal.Warp(rel_path(f'../data/rio/scaled/3band_AOI_1_RIO_img{i}.tif'), image_3band, width=Network.IMAGE_WIDTH, height=Network.IMAGE_HEIGHT)

        img_file = get_8band_image_path(i)
        image_8band = gdal.Open(img_file)
        gdal.Warp(rel_path(f'../data/rio/scaled/8band_AOI_1_RIO_img{i}.tif'), image_8band, width=Network.IMAGE_WIDTH, height=Network.IMAGE_HEIGHT)


class Network:
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    IMAGE_CHANNELS = 11

    def __init__(self, layers=None, per_image_standardization=True, batch_norm=True, skip_connections=True):
        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_1_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=skip_connections))

            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_2_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_2_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=skip_connections))

            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_3_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_3', skip_connection=skip_connections))

            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_4_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_4_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_4'))

        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        if per_image_standardization:
            list_of_images_norm = tf.map_fn(tf.image.per_image_standardization, self.inputs)
            net = tf.stack(list_of_images_norm)
        else:
            net = self.inputs

        # ENCODER
        for layer in layers:
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())

        layers.reverse()
        Conv2d.reverse_global_variables()

        # DECODER
        for layer in layers:
            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])

        self.segmentation_result = tf.layers.dense(inputs=tf.sigmoid(net), units=1)

        self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = tf.reduce_mean(correct_pred)

            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()


def draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network, batch_num):
    n_examples_to_plot = 12
    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10))

    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(test_inputs[example_i][:,:,:3])
        axs[1][example_i].imshow(test_targets[example_i][:,:,0], cmap='gray')
        axs[2][example_i].imshow(
            np.reshape(test_segmentation[example_i][:,:,0], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]), cmap='gray')

        test_image_thresholded = np.array(
            [0 if x < 0.5 else 255 for x in test_segmentation[example_i].flatten()])
        axs[3][example_i].imshow(
            np.reshape(test_image_thresholded, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]),
            cmap='gray')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = 'image_plots/'
    if not os.path.exists(IMAGE_PLOT_DIR):
        os.makedirs(IMAGE_PLOT_DIR)

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    return buf


def jaccard(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)

    return intersection.sum() / float(union.sum())


def train():
    BATCH_SIZE = 25
    IMAGES_COUNT = 800
    TEST_IMAGES_COUNT = 100
    EPOCHS = 50
    BATCHES_IN_EPOCH = int(math.floor(IMAGES_COUNT / BATCH_SIZE))

    network = Network()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(os.path.join('models', network.description, timestamp))

    all_inputs = []
    all_targets = []
    batch_pointer = 0

    test_inputs = []
    test_targets = []

    for i in range(1, IMAGES_COUNT + 1):
        all_inputs.append(convert_geotiff_to_array(i, scaled=True))
        all_targets.append(convert_target_to_array(i))

    for i in range(IMAGES_COUNT + 1, IMAGES_COUNT + 1 + TEST_IMAGES_COUNT):
        test_inputs.append(convert_geotiff_to_array(i, scaled=True))
        test_targets.append(convert_target_to_array(i))

    def next_batch(batch_pointer):
        inputs = []
        targets = []

        for i in range(BATCH_SIZE):
            inputs.append(all_inputs[batch_pointer + i])
            targets.append(all_targets[batch_pointer + i])

        batch_pointer += BATCH_SIZE
        return np.array(inputs), np.array(targets)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Running session...')
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        test_accuracies = []
        global_start = time.time()
        for epoch_i in range(EPOCHS):
            batch_pointer = 0
            permutation = np.random.permutation(IMAGES_COUNT)
            all_inputs = [all_inputs[i] for i in permutation]
            all_targets = [all_targets[i] for i in permutation]

            for batch_i in range(BATCHES_IN_EPOCH):
                batch_num = epoch_i * BATCHES_IN_EPOCH + batch_i + 1

                start = time.time()
                batch_inputs, batch_targets = next_batch(batch_pointer)
                batch_inputs = np.reshape(batch_inputs,
                                          (BATCH_SIZE, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
                batch_targets = np.reshape(batch_targets,
                                           (BATCH_SIZE, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))

                cost, _ = sess.run([network.cost, network.train_op],
                                   feed_dict={network.inputs: batch_inputs, network.targets: batch_targets,
                                              network.is_training: True})
                end = time.time()
                print('{}/{}, epoch: {}, cost: {}, batch time: {}'.format(batch_num,
                                                                          EPOCHS * BATCHES_IN_EPOCH,
                                                                          epoch_i, cost, end - start))

                if batch_num % 10 == 0 or batch_num == EPOCHS * BATCHES_IN_EPOCH:
                    print('Testing...')
                    test_inputs = np.reshape(test_inputs, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
                    test_targets = np.reshape(test_targets, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                    summary, test_accuracy = sess.run([network.summaries, network.accuracy],
                                                    feed_dict={network.inputs: test_inputs,
                                                                network.targets: test_targets,
                                                                network.is_training: False})

                    test_segmentation = sess.run(network.segmentation_result, feed_dict={
                        network.inputs: np.reshape(test_inputs, [TEST_IMAGES_COUNT, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 11])})

                    for i in range(len(test_targets)):
                        flattened = test_targets[i].flatten()
                        test_segmentation_binarized = np.array([0 if x < 0.5 else 255 for x in test_segmentation[i].flatten()])
                        if not any(test_segmentation_binarized) and not any(flattened):
                            print(f'Test case {i}: empty')
                        else:
                            jaccard_index = jaccard(test_targets[i].flatten(), test_segmentation_binarized)
                            print(f'Test case {i}: {jaccard_index}')

                    print('Step {}, test accuracy: {}'.format(batch_num, test_accuracy))
                    test_accuracies.append((test_accuracy, batch_num))
                    print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                    max_acc = max(test_accuracies)
                    print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                    print("Total time: {}".format(time.time() - global_start))

                    n_examples = 8
                    sample = random.sample(range(TEST_IMAGES_COUNT), n_examples) + [2,3,4,5]
                    test_plot_buf = draw_results(test_inputs[sample], test_targets[sample], test_segmentation[sample], test_accuracy, network,
                                                 batch_num)

                    # image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)
                    # image = tf.expand_dims(image, 0)

                    # if test_accuracy >= max_acc[0]:
                    #    checkpoint_path = os.path.join('models', network.description, timestamp, 'model.ckpt')
                    #    saver.save(sess, checkpoint_path, global_step=batch_num)


def preprocess_data():
    rescale_images()
    generate_masks()


if __name__ == '__main__':
    #preprocess_data()
    train()


#patches, patches_no_fill = geojson_to_pixel_arr(img_file, geojson_file)
#plot_truth_coords(plt.imread(img_file), plt.imread(mask_file), patches)
#band3_solution_path = rel_path('../data/rio/vectordata/summarydata/AOI_1_RIO_polygons_solution_3band.csv')
#band8_solution_path = rel_path('../data/rio/vectordata/summarydata/AOI_1_RIO_polygons_solution_8band.csv')
#ImageId,BuildingId,PolygonWKT_Pix,PolygonWKT_Geo
#band3_solution_df = pd.read_csv(band3_solution_path)
#band8_solution_df = pd.read_csv(band8_solution_path)
