#coding=utf8

import glob
import os.path
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.platform import gfile

INPUT_DATA = 'data/flower_photos/'
OUTPUT_FILE = 'data/flower_photos/flower_processed_data.npy'

VALIDATION_PERCENTTAGE = 10
TEST_PERCENTAGE = 10

def create_image_lists(sess,test_percentage,validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_path = True

    training_images = []
    testing_images = []
    training_labels = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    for sub_dir in sub_dirs:
        if is_root_path:
            is_root_path = False
            continue
        extensions = ['jpg','jpeg','JPG','JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in tqdm(extensions):
            file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
            if not file_list: continue

            for file_name in tqdm(file_list):
                image_raw_data = gfile.FastGFile(file_name,'rb').read()
                image = tf.image.decode_jpeg(image_raw_data)
                if image.dtype != tf.float32:
                    image = tf.image.convert_image_dtype(image,dtype=tf.float32)
                    image = tf.image.resize_images(image,[299,299])
                    image_value = sess.run(image)

                chance = np.random.randint(100)
                if chance<validation_percentage:
                    validation_images.append(image_value)
                    validation_labels.append(current_label)
                elif chance<(test_percentage+validation_percentage):
                    testing_images.append(image_value)
                    testing_labels.append(current_label)
                else:
                    training_images.append(image_value)
                    training_labels.append(current_label)
            current_label +=1
        state = np.random.get_state()
        np.random.shuffle(training_images)
        np.random.set_state(state)
        np.random.shuffle(training_labels)
        return np.asarray([training_images,training_labels,
                           validation_images,validation_labels,
                           testing_images,testing_labels])

def main():
    with tf.Session() as sess:
        precessed_data = create_image_lists(sess,TEST_PERCENTAGE,VALIDATION_PERCENTTAGE)
        np.save(OUTPUT_FILE,precessed_data)

if __name__=='__main__':
    main()