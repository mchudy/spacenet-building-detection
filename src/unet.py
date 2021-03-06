import time
import os
import pandas as pd
import tensorflow as tf


def conv_conv_pool(input_, n_filters, training, name, pool=True, activation=tf.nn.relu):
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(net, F, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upsample_concat(inputA, input_B, name):
    upsample = upsampling_2D(inputA, size=(2, 2), name=name)
    return tf.concat([upsample, input_B], axis=-1, name="concat_{}".format(name))


def upsampling_2D(tensor, name, size=(2, 2)):
    H, W, _ = tensor.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name="upsample_{}".format(name))


def IOU_(y_pred, y_true):
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)


def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 1, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[0]]))
    return tf.div(exponential_map,tensor_sum_exp)


class Network:
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    IMAGE_CHANNELS = 11

    def __init__(self, layers=None, per_image_standardization=False, batch_norm=False, skip_connections=True):
        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        with tf.device('/gpu:0'):
            net = self.inputs * 2 - 1

            conv1, pool1 = conv_conv_pool(net, [32, 32], self.is_training, name=1)
            conv2, pool2 = conv_conv_pool(pool1, [64, 64], self.is_training, name=2)
            conv3, pool3 = conv_conv_pool(pool2, [128, 128], self.is_training, name=3)
            conv4, pool4 = conv_conv_pool(pool3, [256, 256], self.is_training, name=4)
            conv5 = conv_conv_pool(pool4, [512, 512], self.is_training, name=5, pool=False)

            up6 = upsample_concat(conv5, conv4, name=6)
            conv6 = conv_conv_pool(up6, [256, 256], self.is_training, name=6, pool=False)

            up7 = upsample_concat(conv6, conv3, name=7)
            conv7 = conv_conv_pool(up7, [128, 128], self.is_training, name=7, pool=False)

            up8 = upsample_concat(conv7, conv2, name=8)
            conv8 = conv_conv_pool(up8, [64, 64], self.is_training, name=8, pool=False)

            up9 = upsample_concat(conv8, conv1, name=9)
            conv9 = conv_conv_pool(up9, [32, 32], self.is_training, name=9, pool=False)

            self.segmentation_result = tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=tf.nn.sigmoid, padding='same')

            self.cost = -IOU_(self.segmentation_result, self.targets)
            #self.cost = tf.reduce_sum ( tf.pow( tf.subtract(self.segmentation_result, self.targets),2))
            global_step = tf.train.get_or_create_global_step()
            self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=0.01).minimize(self.cost)

            with tf.name_scope('accuracy'):
                argmax_probs = tf.round(self.segmentation_result)
                correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
                self.accuracy = tf.reduce_mean(correct_pred)
                tf.summary.scalar('accuracy', self.accuracy)
                tf.summary.scalar('cost', self.cost)

            with tf.name_scope('cost'):
                tf.summary.scalar('cost', self.cost)

            self.summaries = tf.summary.merge_all()
