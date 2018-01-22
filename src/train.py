import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from osgeo import gdal, osr, ogr
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import datetime
import time
import math
import io
import random
from unet import Network
import preprocess
import csv
import pandas as pd
from greedy_clustering import FindAllClusters
from tqdm import trange
from skimage import measure


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
pd.options.mode.chained_assignment = None


BATCH_SIZE = 1
IMAGES_COUNT = 5000
TEST_IMAGES_COUNT = 200
EPOCHS = 5
TEST_PERIOD = 200
BATCHES_IN_EPOCH = int(math.floor(IMAGES_COUNT / BATCH_SIZE))
TEST_BATCH_SIZE = 10
TEST_BATCH_COUNT = int(math.floor(TEST_IMAGES_COUNT / TEST_BATCH_SIZE))


def rel_path(path):
    return os.path.join(SCRIPT_PATH, path)


def convert_geotiff_to_array(image_number, scaled=False):
    image_3band = gdal.Open(preprocess.get_3band_image_path(image_number, scaled))
    image_8band = gdal.Open(preprocess.get_8band_image_path(image_number, scaled))
    channels = image_3band.RasterCount + image_8band.RasterCount
    mul_img = np.zeros((image_3band.RasterXSize, image_3band.RasterYSize, channels), dtype='float')

    for band in range(0, image_3band.RasterCount):
        mul_img[:,:,band] = image_3band.GetRasterBand(band+1).ReadAsArray().transpose().astype(float) / 255.0

    for band in range(0, image_8band.RasterCount):
        output_band = image_3band.RasterCount + band
        mul_img[:,:,output_band] = image_8band.GetRasterBand(band+1).ReadAsArray().transpose().astype(float) / 255.0

    return mul_img


def convert_target_to_array(image_number):
    image = gdal.Open(preprocess.get_mask_image_path(image_number))
    channels = image.RasterCount
    mul_img = np.zeros((image.RasterXSize, image.RasterYSize, channels), dtype='float')
    mul_img[:,:, 0] = image.GetRasterBand(1).ReadAsArray().transpose().astype(float) / 255.0

    return mul_img


def convert_dist_transform_to_array(image_number):
    transformed = np.load(preprocess.get_distance_transform_image_path(image_number) + '.npy')
    normalized = 2*(transformed-transformed.min())/(transformed.max()-transformed.min()) - 1
    return np.nan_to_num(normalized)


def draw_results(test_inputs, test_targets, test_segmentation, network, timestamp, batch_num):
    n_examples_to_plot = len(test_inputs)
    _, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10))

    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(test_inputs[example_i][:,:,:3])
        axs[1][example_i].imshow(test_targets[example_i][:,:,0])
        # axs[2][example_i].imshow(test_segmentation[example_i][:,:,0])
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

    os.makedirs(rel_path(f'../image_plots/{timestamp}/' ), exist_ok=True)
    plt.savefig('{}/batch_{}.jpg'.format(rel_path(f'../image_plots/{timestamp}'), batch_num))
    return buf


def create_truth_csv(imgs):
    os.makedirs(rel_path('../output/geojson'), exist_ok=True)
    truth_df = pd.read_csv(rel_path('../data/rio/vectordata/summarydata/AOI_1_RIO_polygons_solution_3band.csv'))
    with open(rel_path('../output/geojson/truth.csv'), 'w') as fw:
        fw.write('ImageId,BuildingId,PolygonWKT_Pix,PolygonWKT_Geo\n')
        for img_no in imgs:
            image_id = 'AOI_1_RIO_img' + str(img_no)
            image_df = truth_df[truth_df['ImageId'] == image_id]
            image_df['ImageId'] = 'AOI_1_RIO_img' + str(img_no)
            csvv = image_df.to_csv(index=False, header=False)
            fw.write(csvv)


def train():
    network = Network()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(os.path.join('models', network.description, timestamp))
    os.makedirs(rel_path('../results/csv'), exist_ok=True)

    # perm = preprocess.get_permutation()
    # np.save(rel_path('../results/perm'), perm)

    perm = np.load(rel_path('../results/perm.npy'))

    all_inputs = []
    all_targets = []
    batch_pointer = 0

    original_test_img_no = []

    for i in range(IMAGES_COUNT + 1, IMAGES_COUNT + 1 + TEST_IMAGES_COUNT):
        original_test_img_no.append(perm[i])

    create_truth_csv(original_test_img_no)

    for i in range(IMAGES_COUNT):
        all_inputs.append(i)
        all_targets.append(i)


    def next_batch(batch_pointer):
        inputs = []
        targets = []

        for i in range(BATCH_SIZE):
            input_image_array = convert_geotiff_to_array(all_inputs[batch_pointer + i], scaled=True)
            mask_array = convert_target_to_array(all_inputs[batch_pointer + i])
            inputs.append(input_image_array)
            targets.append(mask_array)

        batch_pointer += BATCH_SIZE
        return np.array(inputs), np.array(targets)


    def next_test_batch(test_batch_num):
        test_inputs = []
        test_targets = []
        for i in range(IMAGES_COUNT + 1 + test_batch_num * TEST_BATCH_SIZE, IMAGES_COUNT + 1 + (test_batch_num + 1) * TEST_BATCH_SIZE):
            test_inputs.append(convert_geotiff_to_array(i, scaled=True))
            test_targets.append(convert_target_to_array(i))
        test_inputs = np.reshape(np.array(test_inputs), (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
        test_targets = np.reshape(np.array(test_targets), (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
        return test_inputs, test_targets


    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('Running session...')
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

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
                                                                          epoch_i + 1, cost, end - start))

                batch_inputs = []
                batch_targets = []

                if batch_num % TEST_PERIOD == 0 or batch_num == EPOCHS * BATCHES_IN_EPOCH:
                    print('Testing...')

                    test_accuracy = 0
                    test_cost = 0
                    test_segmentation = np.array([], dtype=np.dtype(float)).reshape(0, 256, 256, 1)

                    for i in trange(TEST_BATCH_COUNT):
                        batch_inputs, batch_targets = next_test_batch(i)
                        batch_test_cost, batch_test_accuracy, batch_test_segmentation = sess.run([network.cost, network.accuracy, network.segmentation_result],
                                feed_dict={network.inputs: batch_inputs,
                                            network.targets: batch_targets,
                                            network.is_training: False})
                        test_cost += batch_test_cost
                        test_accuracy += batch_test_accuracy
                        print(batch_test_cost)
                        test_segmentation = np.concatenate((test_segmentation, batch_test_segmentation))

                        if i < 2 or test_cost <= -0.3:
                            draw_results(batch_inputs, batch_targets, batch_test_segmentation, network, timestamp, f'{batch_num}_{i}')

                        # for j in trange(TEST_BATCH_SIZE):
                        #     result = test_segmentation[j]
                        #     result = result.reshape(256, 256)
                        #     cluster = FindAllClusters(result)
                        #     print(f'{j} clustered')
                        #     img_no = IMAGES_COUNT + 1 + i * TEST_BATCH_SIZE + j
                        #     preprocess.CreateGeoJSON('AOI_1_RIO_img' + str(img_no), cluster)
                        #     preprocess.FixGeoJSON('AOI_1_RIO_img' + str(img_no))

                    test_accuracy /= TEST_BATCH_COUNT
                    test_cost /= TEST_BATCH_COUNT

                    print('Step {}, test accuracy: {}'.format(batch_num, test_accuracy))
                    print('Cost {}'.format(test_cost))
                    test_accuracies.append((test_cost, batch_num))
                    print("Costs in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                    max_acc = max(test_accuracies)
                    print("Best cost: {} in batch {}".format(max_acc[0], max_acc[1]))
                    print("Total time: {}".format(time.time() - global_start))

                    # preprocess.merge_results(rel_path('../output/geojson'), rel_path('../output/geojson/result' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + '.csv'), perm)
                    # preprocess.merge_results(rel_path('../output/geojson'), rel_path('../output/geojson/result' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + 'tr.csv'), perm, transpose=True)

                    if test_accuracy >= max_acc[0]:
                       checkpoint_path = os.path.join('models', network.description, timestamp, 'model.ckpt')
                       saver.save(sess, checkpoint_path, global_step=batch_num)

                    csv_data=[epoch_i + 1, batch_num, cost, test_cost, test_accuracy]
                    with open(rel_path('../results/csv/' + timestamp + '.csv'), 'a') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(csv_data)

if __name__ == '__main__':
    train()
