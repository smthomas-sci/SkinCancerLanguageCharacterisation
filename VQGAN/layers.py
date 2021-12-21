"""


Contains the layers for a VQ-GAN

Author: Simon Thomas
Date: 2021-07-15

"""

import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import (Layer, AveragePooling2D, UpSampling2D, Conv2D, Input,
                                     GlobalAveragePooling2D, LeakyReLU)
from tensorflow_addons.layers import GroupNormalization as Normalize
from tensorflow.keras import Model

class DownSample(Layer):
    """
    Downsampling layers that wraps and AveragePooling2D operation.
    """
    def __init__(self, **kwargs):
        super(DownSample, self).__init__(**kwargs)
        self.downsample = AveragePooling2D()
        
    def call(self, inputs):
        x = self.downsample(inputs)
        return x
    
    
class UpSample(Layer):
    """
    And upsample layer that includes a convolution and an activation.
    """
    def __init__(self, filters, **kwargs):
        super(UpSample, self).__init__(**kwargs)
        self.filters = filters
        self.upsample = UpSampling2D()
        self.upconv = Conv2D(filters=filters, kernel_size=(3, 3), padding="same")
        self.act = Swish()
        
    def call(self, inputs):
        x = self.upsample(inputs)
        x = self.upconv(x)
        return self.act(x)
    

class Quantizer(Layer):
    """
    A vector quantization layer that follows the paper:
    
    "Neural Discrete Representation Learning (https://arxiv.org/abs/1711.00937)"
    
    Inspired by:
    - https://github.com/CompVis/taming-transformers/taming/modules/vqvae/quantize.py (Torch)
    - https://github.com/iomanker/VQVAE-TF2/VectorQuantizer.py (TF2)
    
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=1., initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.initializer = initializer
        self._commitment_cost = commitment_cost
        super(Quantizer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights.
        self._w = self.add_weight(name='embedding',
                                  shape=(self.embedding_dim, self.num_embeddings),
                                  initializer=self.initializer,
                                  trainable=True)

        # Finalize building.
        super(Quantizer, self).build(input_shape)

    def call(self, x):
        
        z_e = x
        # Flatten input except for last dimension.
        flat_inputs = K.reshape(x, (-1, self.embedding_dim))

        # Calculate distances of input to embedding vectors.
        distances = (K.sum(flat_inputs**2, axis=1, keepdims=True)
                     - 2 * K.dot(flat_inputs, self._w)
                     + K.sum(self._w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = K.argmax(-distances, axis=1)
        encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
        
        # Quantize 
        z_q = self.get_embeddings(encoding_indices)
        
        # Calculate loss
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(z_q) - z_e) ** 2)
        q_latent_loss = tf.reduce_mean((z_q - tf.stop_gradient(z_e)) ** 2)
        latent_loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Straight through
        z_q = z_e + tf.stop_gradient(z_q - z_e)
        
        return encoding_indices, z_q, z_e, latent_loss

    @property
    def embeddings(self):
        return self._w

    def get_embeddings(self, encoding_indices):
        """
        This recieves the indices and returns the learned embedding
        """
        w = K.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, K.cast(encoding_indices, "int64"))
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
    

class Swish(Layer):
    """
    A switch activation layer.
    """
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.keras.activations.swish(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape    
    
    
    
class ResidualBlock(Model):
    """
    Residual block using group normalisation and swish.
    """
    def __init__(self, filters, rate=(1,1), **kwargs):
        """
        :param filters: the number of convolution filters (fixed 3x3 size)
        """
        super(ResidualBlock, self).__init__(**kwargs)
        # Attributes
        self.filters = filters
        
        # Trainable Layers
        self.match = Conv2D(filters=filters, kernel_size=(1, 1), padding="same")
        
        self.conv1 = Conv2D(filters=filters, kernel_size=(3, 3), padding="same", dilation_rate=rate)
        self.norm1 = Normalize()
        self.act1 = Swish()
        
        self.conv2 = Conv2D(filters=filters, kernel_size=(3, 3), padding="same", dilation_rate=rate)
        self.norm2 = Normalize()
        self.act2 = Swish()
            

    def call(self, inputs, **kwargs):
        
        x = inputs
        
        h = self.match(x)
        # Convolution 1
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        # Convolution 2
        x = self.conv2(x)
        
        # Make residual
        x += h
        
        x = self.norm2(x)
        x = self.act2(x)
        
        return x    
    
    
    
class AttnBlock(Layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.norm = Normalize()
        self.q = Conv2D(filters=channels, kernel_size=(1, 1), padding="same")
        self.k = Conv2D(filters=channels, kernel_size=(1, 1), padding="same")
        self.v = Conv2D(filters=channels, kernel_size=(1, 1), padding="same")
        self.proj_out = Conv2D(filters=channels, kernel_size=(1, 1), padding="same")
                
    def forward(self, inputs):
        h_ = inputs
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = tf.shape(q)
        q = tf.reshape(q, (b,c,h*w))
        q = tf.transpose(q, perm=[0, 2, 1]) # b,hw,c

        k = tf.reshape(k, (b,c,h*w)) # b,c,hw
  
        w_ = K.batch_dot(q, k) # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = tf.keras.activations.softmax(w_, axis=2)

        # attend to values
        v = tf.reshape(v, (b,c,h*w))
        w_ = tf.transpose(w_, perm=[0,2,1]) # b,hw,hw (first hw of k, second of q)
        h_ = K.batch_dot(v, w_) # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = tf.reshape(h_, (b,c,h,w))

        h_ = self.proj_out(h_)

        return x+h_
    
