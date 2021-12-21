import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.activations import softplus as f
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dropout

import numpy as np


def z_norm(y_true, y_pred):
    return tf.reduce_mean(tf.norm(y_pred, axis=-1))
    
def reconstruction_loss(y_true, y_pred):
    return K.mean((y_true - y_pred)**2)

def normalize(v):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))
    
def create_perceptual_model():
    """
    According to https://arxiv.org/pdf/1707.09405.pdf the following
    5 layers are used in the VGG19 model:
        [‘conv1 2’, ‘conv2 2’, conv3 2’, ‘conv4 2’, ‘conv5 2’]
    :return:
    """
    vgg = VGG19(include_top=False, weights="imagenet")
    # Get input
    feat_in = vgg.get_input_at(0)

    # Get features
    features_1 = vgg.get_layer("block1_conv2").output
    features_2 = vgg.get_layer("block2_conv2").output
    features_3 = vgg.get_layer("block3_conv2").output
    features_4 = vgg.get_layer("block4_conv2").output
    features_5 = vgg.get_layer("block5_conv2").output
    
    norm_1 = normalize(features_1)
    norm_2 = normalize(features_2)
    norm_3 = normalize(features_3)
    norm_4 = normalize(features_4)
    norm_5 = normalize(features_5)
    
    norm_1 = Dropout(0.5)(norm_1)
    norm_2 = Dropout(0.5)(norm_2)
    norm_3 = Dropout(0.5)(norm_3)
    norm_4 = Dropout(0.5)(norm_4)
    norm_5 = Dropout(0.5)(norm_5)
    
    out_1 = Conv2D(1, 1, name="linear_transform_1", use_bias=False)(norm_1)
    out_2 = Conv2D(1, 1, name="linear_transform_2", use_bias=False)(norm_2)
    out_3 = Conv2D(1, 1, name="linear_transform_3", use_bias=False)(norm_3)
    out_4 = Conv2D(1, 1, name="linear_transform_4", use_bias=False)(norm_4)
    out_5 = Conv2D(1, 1, name="linear_transform_5", use_bias=False)(norm_5)
    

    model = Model(inputs=[feat_in], outputs=[out_1, out_2, out_3, out_4, out_5])
    
    # Load weights
    w1 = np.expand_dims(np.load(f"./weights/perceptual_similarity_1x1_weights/block_1.npy"), axis=-1)
    w2 = np.expand_dims(np.load(f"./weights/perceptual_similarity_1x1_weights/block_2.npy"), axis=-1)
    w3 = np.expand_dims(np.load(f"./weights/perceptual_similarity_1x1_weights/block_3.npy"), axis=-1)
    w4 = np.expand_dims(np.load(f"./weights/perceptual_similarity_1x1_weights/block_4.npy"), axis=-1)
    w5 = np.expand_dims(np.load(f"./weights/perceptual_similarity_1x1_weights/block_5.npy"), axis=-1)
    
    model.get_layer("linear_transform_1").set_weights(w1)
    model.get_layer("linear_transform_2").set_weights(w2)
    model.get_layer("linear_transform_3").set_weights(w3)
    model.get_layer("linear_transform_4").set_weights(w4)
    model.get_layer("linear_transform_5").set_weights(w5)
    
    return model


class PerceptualLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.perceptual_model = create_perceptual_model()
        super(PerceptualLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="perceptual_loss")
        
    def call(self, x_true, x_pred):
    
        # Rescale -> (-1,1) -> (0-1) -> (0, 255.)
        x_true = ((x_true + 1) / 2.) * 255.
        x_pred = ((x_pred + 1) / 2.) * 255.

        # Preprocess
        x_true = preprocess_input(x_true)
        x_pred = preprocess_input(x_pred)

        f_true = self.perceptual_model(x_true)
        f_pred = self.perceptual_model(x_pred)
        
        p_loss = 0
        for ft,fp in zip(f_true, f_pred):
            p_loss += tf.reduce_mean(tf.square(ft - fp))

        return p_loss
 
    
class L2(tf.keras.losses.Loss):
    def __init__(self):
        super(L2, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="l2")

    def call(self, y_true, y_pred):
        loss = tf.sqrt((y_true - y_pred)**2)
        return loss
    

class DRAD(tf.keras.losses.Loss):
    """
    Discriminator - Relativistic Average Discriminator
    """
    def __init__(self):
        super(DRAD, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="dns")

    def call(self, d_real, d_fake):

        logits_diff_real_fake = d_real - tf.reduce_mean(d_fake, axis=0, keepdims=True)
        logits_diff_fake_real = d_fake - tf.reduce_mean(d_real, axis=0, keepdims=True)

        loss_dis_real = tf.reduce_mean(
            binary_crossentropy(y_true=tf.ones_like(d_fake),
                                y_pred=logits_diff_real_fake,
                                from_logits=True))

        loss_dis_fake = tf.reduce_mean(
            binary_crossentropy(y_true=tf.zeros_like(d_fake),
                                y_pred=logits_diff_fake_real,
                                from_logits=True))

        loss = loss_dis_real + loss_dis_fake

        return loss

    

class GRAD(tf.keras.losses.Loss):
    """
    Generator - Relativistic Average Discriminator
    """
    def __init__(self):
        super(GRAD, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="gns")

    def call(self, d_real, d_fake):

        logits_diff_real_fake = d_real - tf.reduce_mean(d_fake, axis=0, keepdims=True)
        logits_diff_fake_real = d_fake - tf.reduce_mean(d_real, axis=0, keepdims=True)

        # Generator loss.
        loss_gen_real = tf.reduce_mean(
            binary_crossentropy(y_true=tf.ones_like(d_fake),
                                y_pred=logits_diff_fake_real,
                                from_logits=True))

        loss_gen_fake = tf.reduce_mean(
            binary_crossentropy(y_true=tf.zeros_like(d_fake),
                                y_pred=logits_diff_real_fake,
                                from_logits=True))

        loss = loss_gen_real + loss_gen_fake

        return loss
    

class DNS(tf.keras.losses.Loss):
    """
    Discriminator - Non-saturating loss
    """
    def __init__(self):
        super(DNS, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="dns")

    def call(self, d_real, d_fake):
        loss = f(-d_real) + f(d_fake)
        return loss


class GNS(tf.keras.losses.Loss):
    """
    Generator - Non-saturating loss
    """
    def __init__(self):
        super(GNS, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="gns")

    def call(self, g_real, g_fake):
        loss = f(-g_fake)
        return loss

    
    
class DS(tf.keras.losses.Loss):
    """
    Discriminator - Saturating Loss (Vanilla GAN)
    """
    def __init__(self):
        super(DS, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="dns")

    def call(self, d_real, d_fake):

        loss_dis_real = tf.reduce_mean(
            binary_crossentropy(y_true=tf.ones_like(d_fake),
                                y_pred=d_real,
                                from_logits=True))

        loss_dis_fake = tf.reduce_mean(
            binary_crossentropy(y_true=tf.zeros_like(d_fake),
                                y_pred=d_fake,
                                from_logits=True))

        loss = loss_dis_real + loss_dis_fake

        return loss

    
class GS(tf.keras.losses.Loss):
    """
    Generator - Saturating Loss (Vanilla GAN)
    """
    def __init__(self):
        super(GS, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="gns")

    def call(self, d_real, d_fake):

        # Generator loss.
        loss = tf.reduce_mean(
            binary_crossentropy(y_true=tf.ones_like(d_fake),
                                y_pred=d_fake,
                                from_logits=True))

        return loss
    

    
class DH(tf.keras.losses.Loss):
    """
    Discriminator - Hinge Loss
    """
    def __init__(self):
        super(DH, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="dns")

    def call(self, d_real, d_fake):

        loss_dis_real = tf.reduce_mean(tf.keras.activations.relu(1. - d_real))
        loss_dis_fake = tf.reduce_mean(tf.keras.activations.relu(1. + d_fake))

        loss = 0.5 * (loss_dis_real + loss_dis_fake)

        return loss

    
class GH(tf.keras.losses.Loss):
    """
    Generator - Hinge Loss
    """
    def __init__(self):
        super(GH, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="gns")

    def call(self, d_real, d_fake):

        # Generator loss
        loss = -tf.reduce_mean(d_fake)

        return loss
    

# Create losses
l2 = L2()

discriminator_loss = DNS()
generator_loss = GNS()

# discriminator_loss = DS()
# generator_loss = GS()

#discriminator_loss = DH()
#generator_loss = GH()

#discriminator_loss = DRAD()
#generator_loss = GRAD()

perceptual_loss = PerceptualLoss()

