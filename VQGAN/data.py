"""

Functions to support input of data for training and testing.

Author: Simon Thomas
Date: 2021-07-15

"""

import tensorflow as tf
import os


AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_data_set(data_directory=None, file_names=None, img_dim=4, batch_size=12, auto=False):
    """
    Creates a data set using the files in the data directory. If
    file_names is specified, only these file names will be used
    in the data set.
    :param data_directory: the path to the data directory.
    :param file_names: a list of full path file names
    :param img_dim: the size of the image
    :param batch_size: the batch size. Can be updated as ds = ds.batch(new_batch_size)
    :return: data set (ds)
    :return: n
    """
    if not file_names:
        file_names = [os.path.join(data_directory, file) for file in os.listdir(data_directory)]

    def parse_image(filename):
        """
        Reads the image and returns the image, noise and constant tensors
        :param filename: the image to load
        :return: style batch - ( image, noise_img, constant)
        """
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if image.shape[0] != img_dim:
            image = tf.image.resize(image, [img_dim, img_dim])
        
        # scale between -1 and 1
        image = 2*image - 1
        
        if auto:
            return image, image
        return image #, image #, noise_image

    n = len(file_names)
    ds = tf.data.Dataset.from_tensor_slices(file_names)
    ds = ds.shuffle(buffer_size=n, seed=1234, reshuffle_each_iteration=False)
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = ds.repeat()  # loop forever
    return ds, n
