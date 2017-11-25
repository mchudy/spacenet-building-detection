import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import pandas as pd
import gdal
import numpy as np

dataset = 'AOI_1_RIO'
script_path = os.path.dirname(os.path.realpath(__file__))

band3_solution_path = os.path.join(script_path, '../data/rio/vectordata/summarydata/AOI_1_RIO_polygons_solution_3band.csv')
band8_solution_path = os.path.join(script_path, '../data/rio/vectordata/summarydata/AOI_1_RIO_polygons_solution_8band.csv')

band3_images_path = os.path.join(script_path, '../data/rio/3band')
band8_images_path = os.path.join(script_path, '../data/rio/8band')

#ImageId,BuildingId,PolygonWKT_Pix,PolygonWKT_Geo
band3_solution_df = pd.read_csv(band3_solution_path)
band8_solution_df = pd.read_csv(band8_solution_path)

ds8 = gdal.Open(band8_images_path+ '/8band_' + dataset + '_img' + str(1) + '.tif')
print(ds8)
#x8band = tf.placeholder(tf.float32, shape=[None, FLAGS.ws, FLAGS.ws, 8])
#x3band = tf.placeholder(tf.float32, shape=[None, scale * FLAGS.ws, scale * FLAGS.ws, 3])
