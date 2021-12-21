"""

A collection of utilities


Author: Simon Thomas
Date: 2021-07-15

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard, Callback



def deprocess(x):
    return (x + 1) / 2.

class SingleValueTracker(tf.keras.metrics.Metric):

    def __init__(self, name, **kwargs):
        super(SingleValueTracker, self).__init__(name=name, **kwargs)
        self.value = tf.Variable(0.0,
                                 name='w',
                                 trainable=False,
                                 dtype=tf.float32,
                                 aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                 synchronization=tf.VariableSynchronization.ON_READ
                                 )
    def update_state(self, value):
        self.value.assign(value)

    def result(self):
        return self.value
    
    

class Progress(Callback):
    """
    Inspired by # https://gist.github.com/soheilb/c5bf0ba7197caa095acfcb69744df756
    """
    def __init__(self,  data_tensor, output_dir, run_num=0):
        self.x = data_tensor
        self.n = self.x.shape[0]
        self.output_dir = output_dir
        self.run = run_num
        super(Progress, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        
        # generate image and reconstructions
        x_pred = self.model(self.x)
        canvas = self.create_canvas(self.x, x_pred, self.n)
        canvas = deprocess(canvas).clip(0, 1)
        self.save_canvas(canvas, epoch)


        # Save weights
        self.model.vq_vae.save_weights(f"./weights/run_{self.run}_z1024_vqvae.h5")
        self.model.discriminator.save_weights(f"./weights/run_{self.run}_z1024_disc.h5")

        
    def create_row(self, tensor, split_n):
        return np.hstack([_[0] for _ in np.split(tensor, split_n)])

    def create_canvas(self, x, x_pred, split_n):
        return np.vstack([self.create_row(x, split_n), self.create_row(x_pred, split_n)])

    def save_canvas(self, canvas, epoch):
        plt.imsave(os.path.join(self.output_dir, f"./{epoch:04d}.jpg"), canvas)

