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


dataset = 'AOI_1_RIO'
script_path = os.path.dirname(os.path.realpath(__file__))
number_of_images = 6940


def rel_path(path):
    return os.path.join(script_path, path)


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


def convert_geotiff_to_array(image_number, scaled=False, only_rgb=False):
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
    BATCH_SIZE = 1
    IMAGES_COUNT = 5000
    TEST_IMAGES_COUNT = 200
    EPOCHS = 10
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

        all_inputs.append(i)
        all_targets.append(i)

    for i in range(IMAGES_COUNT + 1, IMAGES_COUNT + 1 + TEST_IMAGES_COUNT):
        test_inputs.append(convert_geotiff_to_array(i, scaled=True))
        test_targets.append(convert_target_to_array(i))

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

    with tf.Session() as sess:
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

                if batch_num % 10 == 0 or batch_num == EPOCHS * BATCHES_IN_EPOCH:
                    print('Testing...')
                    test_inputs = np.reshape(test_inputs, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
                    test_targets = np.reshape(test_targets, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                    summary, test_accuracy = sess.run([network.summaries, network.accuracy],
                                                    feed_dict={network.inputs: test_inputs,
                                                                network.targets: test_targets,
                                                                network.is_training: False})

                    # for i in range(len(test_targets)):
                    #     flattened = test_targets[i].flatten()
                    #     test_segmentation_binarized = np.array([0 if x < 0.5 else 255 for x in test_segmentation[i].flatten()])
                    #     if not any(test_segmentation_binarized) and not any(flattened):
                    #         print(f'Test case {i}: empty')
                    #     else:
                    #         jaccard_index = jaccard(test_targets[i].flatten(), test_segmentation_binarized)
                    #         print(f'Test case {i}: {jaccard_index}')

                    print('Step {}, test accuracy: {}'.format(batch_num, test_accuracy))
                    test_accuracies.append((test_accuracy, batch_num))
                    print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                    max_acc = max(test_accuracies)
                    print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                    print("Total time: {}".format(time.time() - global_start))

                    n_examples = 8
                    sample = random.sample(range(TEST_IMAGES_COUNT), n_examples) + [2,3,4,5]
                    test_segmentation = sess.run(network.segmentation_result, feed_dict={
                        network.inputs: np.reshape(test_inputs[sample], [n_examples + 4, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 11])})
                    test_plot_buf = draw_results(test_inputs[sample], test_targets[sample], test_segmentation, test_accuracy, network,
                                                 batch_num)

                    if test_accuracy >= max_acc[0]:
                       checkpoint_path = os.path.join('models', network.description, timestamp, 'model.ckpt')
                       saver.save(sess, checkpoint_path, global_step=batch_num)

if __name__ == '__main__':
    train()
