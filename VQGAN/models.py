"""

Contains models used in the VQ-GAN.

Author: Simon Thomas
Date: 2021-07-15

"""
import sys

import tensorflow as tf

from VQGAN.layers import *
from VQGAN.losses import *
from VQGAN.utils import SingleValueTracker


def build_encoder(input_dim, embedding_dim, input_channels=3, factor=4, filters=128):
    """
    Builds the encoder model.
    
    param: input_dim - the size of the input e.g. 256.
    param: embedding_dim - the dimension of embedding layer e.g. 512
    param: input_channels - the depth of the input. default = 3
    param: factor - the downsampling factor. default = 4 i.e. 256 -> 16
    param: filters - the number of filters in each block. default =128
    """
    encoder_in = Input((input_dim, input_dim, input_channels))
    
    x = encoder_in
    
    # Conv In
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding="same")(x)
    x = Swish()(x)
    
    # Residual Blocks
    for i in range(factor):
        x = ResidualBlock(filters*2**i)(x)
        x = DownSample()(x)
    
    # Attention
    x = AttnBlock(filters*2**i)(x)
    
    # Middle
    x = ResidualBlock(filters*2**i)(x)
    x = AttnBlock(filters*2**i)(x)
    x = ResidualBlock(filters*2**i)(x)
    
    # Conv Out
    encoder_out = Conv2D(filters=embedding_dim, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    
    return Model(inputs=[encoder_in], outputs=[encoder_out], name="Encoder")



def build_decoder(input_dim, embedding_dim, output_channels=3, factor=4, filters=128):
    """
    
    Builds the decoder model
    
    param: input_dim - the size of the input e.g. 16.
    param: embedding_dim - the dimension of embedding layer e.g. 512
    param: output_channels - the depth of the input. default = 3
    param: factor - the upsampling factor. default = 4, i.e. 16 -> 256
    param: filters - the number of filters in each block. default = 128
    """
    decoder_in = Input(shape=(input_dim, input_dim, embedding_dim))
    
    x = decoder_in
    
    filters = filters * 2**(factor-1)
    
    # Conv In
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding="same")(x)
    x = Swish()(x)
    
    # Attention
    x = ResidualBlock(filters)(x)
    x = AttnBlock(filters)(x)
    x = ResidualBlock(filters)(x)
    
    # Residual Blocks
    for i in range(factor):
        x = ResidualBlock(filters//2**i)(x)
        x = UpSample(filters//2**i)(x)
      
    # Final residual
    x = ResidualBlock(filters//2**i)(x)
    
    decoder_out = Conv2D(filters=output_channels, kernel_size=(3, 3),
                         strides=(1, 1), padding="same", activation="linear")(x)
    
    return Model(inputs=[decoder_in], outputs=[decoder_out], name="Decoder")


def build_vqvae(encoder, decoder, quantizer):
    """
    
    Builds the vqvae model
    
    The latent loss is calculated inside the quantization for simplicity
    of use.
    
    param: encoder - an already built encoder model
    param: decoder - an already built decoder model
    param: quantizer - an already built quantizer layer
    
    """
    vq_vae_input = encoder.input
    
    x = vq_vae_input
    z_e = encoder(x)
    encoding_indices, z_q, z_e, latent_loss = quantizer(z_e)
    
    vq_vae_output = decoder(z_q)
    
    return Model(inputs=[vq_vae_input], outputs=[vq_vae_output, z_q, z_e, latent_loss], name="VQ-VAE")
    

    
class VQGAN(Model):
    def __init__(self, input_dim, embedding_dim, num_embeddings, input_channels=3, output_channels=3, factor=4, filters=128, **kwargs):
        super(VQGAN, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.lowest_dim = input_dim // 2**factor
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.factor = factor
        self.filters = filters
        
        # Build models
        self.encoder = build_encoder(self.input_dim, self.embedding_dim, self.input_channels, self.factor, self.filters)
        self.decoder = build_decoder(self.lowest_dim, self.embedding_dim, self.output_channels, self.factor, self.filters)
        self.quantizer = Quantizer(self.embedding_dim, self.num_embeddings)
        self.vq_vae = build_vqvae(self.encoder, self.decoder, self.quantizer)
        
        # Build GAN
        self.discriminator = self.build_discriminator()
        

    def build_discriminator(self):
        """
        This is the standard patchGAN layout, following closely
        the pix2pix model at https://www.tensorflow.org/tutorials/generative/pix2pix
        """
        initializer = tf.random_normal_initializer(0., 0.02)
        
        discriminator_in = Input((self.input_dim, self.input_dim, self.input_channels))
        
        x = discriminator_in
        
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2,2), padding="same", use_bias=False, kernel_initializer=initializer)(x)
        x = Normalize()(x)
        x = LeakyReLU(0.2)(x)
        
        x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2,2), padding="same", use_bias=False, kernel_initializer=initializer)(x)
        x = Normalize()(x)
        x = LeakyReLU(0.2)(x)
        
        x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2,2), padding="same", use_bias=False, kernel_initializer=initializer)(x)
        x = Normalize()(x)
        x = LeakyReLU(0.2)(x)
        
        x = Conv2D(filters=512, kernel_size=(4, 4), strides=(1,1), use_bias=False, kernel_initializer=initializer)(x)
        x = Normalize()(x)
        x = LeakyReLU(0.2)(x)
        
        # Real or fake?
        x = Conv2D(filters=1, kernel_size=(4, 4), kernel_initializer=initializer)(x)
        
        # Pool predictions
        discriminator_out = GlobalAveragePooling2D()(x)
        
        return Model(inputs=[discriminator_in], outputs=[discriminator_out], name="discriminator")
    
    def compile(self, vq_optimizer, gan_optimizer, warm_up_steps=1_000, commitment_cost=0.25, disc_weight=1.0, γ=0.1, **kwargs):
        """
        Overrides the compile step.
        
        param: vq_optimizer - the optimizer for the decoder, encoder and quantizer
        param: gan_optimizer - the optimizer for the discriminator
        param: warm_up_steps - the number of batches only training the reconstruction loss
        param: commitment_cost - the weighting for the latent loss component
        param: disc_weight - the weight for the gan loss component
        param: γ - the gamma weight of the gan loss (additional modifier - can be dynamic)
        
        """
        super(VQGAN, self).compile(**kwargs)
        self.optimizer_gan = gan_optimizer
        self.optimizer_vq = vq_optimizer
        self.commitment_cost = commitment_cost
        self.disc_weight = disc_weight
        self.δ = 1e-6
        self.warm_up = warm_up_steps
        self.γ = γ
          
        # get trainable params
        self.θ_vqvae =  self.encoder.trainable_weights + \
                        self.decoder.trainable_weights + \
                        self.quantizer.trainable_weights
        self.θ_discr = self.discriminator.trainable_weights

        # Create loss trackers
        self.d_loss_tracker = tf.keras.metrics.Mean(name="loss_d")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="loss_g")
        self.latent_loss_tracker = tf.keras.metrics.Mean(name="loss_latent")
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="loss_vq")
        self.zq_norm_tracker = tf.keras.metrics.Mean(name="zq_norm")
        self.ze_norm_tracker = tf.keras.metrics.Mean(name="ze_norm")
        self.gamma_tracker = SingleValueTracker(name="γ")
                
        
    def calculate_adaptive_weight(self, p_loss, gan_loss, x_pred):
        # Get gradients
        per_grads = tf.gradients(p_loss, x_pred)
        gan_grads = tf.gradients(gan_loss, x_pred)

        # Calculate ratio
        γ = tf.sqrt(tf.reduce_sum(tf.square(per_grads))) / tf.sqrt((tf.reduce_sum(tf.square(gan_grads))) + self.δ)
        
        return γ
         
    def test_step(self, batch):
        batch_size = batch.shape[0]

        if not batch_size:
            batch_size = 1
            
            
        x_real = batch
        

        # ----------------------- #
        #  Evaluate Discriminator #
        # ----------------------- #
        x_pred, z_q, z_e, loss_latent = self.vq_vae(x_real)

        fake_pred = self.discriminator(x_pred)

        real_pred = self.discriminator(x_real)

        loss_d = discriminator_loss(real_pred, fake_pred)
            
        # Check if still in warm-up?
        d_weight = 1. if self.optimizer_vq.iterations > self.warm_up else 0.
            
        loss_d *= d_weight

        # --------------------- #
        #  Evaluate  Generator  #
        # --------------------- #
                
            
        # Preds, zs & latent loss
        x_pred, z_q, z_e, loss_latent = self.vq_vae(x_real)
        
        # Recon loss & Perceptual loss
        loss_l1 = K.mean(tf.abs(x_real - x_pred))
        loss_perceptual = perceptual_loss(x_real, x_pred)
                        
        loss_vq = loss_perceptual + loss_l1 
                        
        # Compute final loss
        loss = loss_vq + self.commitment_cost * loss_latent
            
        # Don't run discriminator
        if self.optimizer_vq.iterations < self.warm_up:
            γ = 0.
            loss_gan = 0.
        else:
            # Discriminator loss
            fake_pred = self.discriminator(x_pred)
            loss_gan = generator_loss(None, fake_pred)
            
            γ = self.γ #self.calculate_adaptive_weight(loss_vq, loss_gan, x_pred)
            loss += γ * loss_gan * self.disc_weight
            

        # ---- METRICS ---- #

        loss = {
                "val_loss_vq":  loss_vq,
                "val_loss_latent": loss_latent,
                "val_zq_norm": z_norm(None, z_q),
                "val_ze_norm": z_norm(None, z_e),
                "val_loss_d": loss_d,
                "val_loss_g": loss_gan,
        }

        return loss


    def train_step(self, batch):
        batch_size = batch.shape[0]

        if not batch_size:
            batch_size = 1
            
            
        x_real = batch
        

        # --------------------- #
        #  Update Discriminator #
        # --------------------- #
        x_pred, z_q, z_e, loss_latent = self.vq_vae(x_real)

        with tf.GradientTape() as tape:

            fake_pred = self.discriminator(x_pred)

            real_pred = self.discriminator(x_real)

            loss_d = discriminator_loss(real_pred, fake_pred)
            
            # Check if still in warm-up?
            d_weight = 1. if self.optimizer_vq.iterations > self.warm_up else 0.
            
            loss_d *= d_weight

        gradients = tape.gradient(loss_d, self.θ_discr)
        self.optimizer_gan.apply_gradients(zip(gradients, self.θ_discr))
            
        # --------------------- #
        #    Update Generator   #
        # --------------------- #
                
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:
            
            # Preds, zs & latent loss
            x_pred, z_q, z_e, loss_latent = self.vq_vae(x_real)
        
            # Recon loss & Perceptual loss
            loss_l1 = K.mean(tf.abs(x_real - x_pred))
            loss_perceptual = perceptual_loss(x_real, x_pred)
                        
            loss_vq = loss_perceptual + loss_l1 
                        
            # Compute final loss
            loss = loss_vq + self.commitment_cost * loss_latent
            
            # Don't run discriminator
            if self.optimizer_vq.iterations < self.warm_up:
                γ = 0.
                loss_gan = 0.
            else:
                # Discriminator loss
                fake_pred = self.discriminator(x_pred)
                loss_gan = generator_loss(None, fake_pred)
            
                γ = self.γ #self.calculate_adaptive_weight(loss_vq, loss_gan, x_pred)
                loss += γ * loss_gan * self.disc_weight
            
        gradients = tape.gradient(loss, self.θ_vqvae)
        self.optimizer_vq.apply_gradients(zip(gradients, self.θ_vqvae))
        
        
        # ---------------------------- METRICS ---------------------------- #
        
        # Update loss trackers
        self.vq_loss_tracker.update_state(loss_vq)
        self.latent_loss_tracker.update_state(loss_latent)
        self.zq_norm_tracker.update_state(z_norm(None, z_q))
        self.ze_norm_tracker.update_state(z_norm(None, z_e))
        self.d_loss_tracker.update_state(loss_d)
        self.g_loss_tracker.update_state(loss_gan)
        self.gamma_tracker.update_state(γ)
        
        
        loss = {
                "loss_vq":  self.vq_loss_tracker.result(),
                "loss_latent": self.latent_loss_tracker.result(),
                "zq_norm": self.zq_norm_tracker.result(),
                "ze_norm": self.ze_norm_tracker.result(),
                "loss_d": self.d_loss_tracker.result(),
                "loss_g": self.g_loss_tracker.result(),
                "γ" : self.gamma_tracker.result()
        }
        return loss
        
        
    def call(self, inputs):
        return self.vq_vae(inputs)[0]
    
    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker, self.vq_loss_tracker,
                self.zq_norm_tracker, self.ze_norm_tracker, self.gamma_tracker]
    

