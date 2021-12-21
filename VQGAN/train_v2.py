
# ------------------------ # 
import sys
sys.path.insert(1, '../')
# ------------------------ # 

import os
import pickle

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

from VQGAN.models import *
from VQGAN.utils import *
from VQGAN.data import *


DATA_DIR = "/scratch/imb/Simon/VQGAN/data/train_data"               # Where the data is stored (.jpg)
VALIDATION_DIR = "/scratch/imb/Simon/VQGAN/data/validation_data"    # Where the val data is stored
BATCH_SIZE = 24                                                     # Batch size, shared across all GPUs
IMG_DIM = 256                                                       # Size of the input image
WARM_UP = 10                                                        # Number of warm-up epochs

# N is the number of files in the directory
train_gen, N = create_data_set(data_directory=DATA_DIR, img_dim=IMG_DIM, batch_size=BATCH_SIZE)
val_gen, val_N = create_data_set(data_directory=VALIDATION_DIR, img_dim=IMG_DIM, batch_size=BATCH_SIZE)

# Create Strategy
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


with strategy.scope():

    vq_gan = VQGAN(input_dim=256,
                   embedding_dim=256,
                   num_embeddings=2048,
                   input_channels=3,
                   output_channels=3,
                   factor=4,    # 4 = 16x16, 5 = 8x8
                   filters=64   # 64
                  )
    
        
    vq_gan.compile(tf.keras.optimizers.Adam(0.00002, beta_1=0.5, beta_2=0.9), # originaly 0.00001
                   tf.keras.optimizers.Adam(0.00002, beta_1=0.5, beta_2=0.9),
                   warm_up_steps=int(N // BATCH_SIZE * WARM_UP),
                   disc_weight=1,
                   γ=1.,
                   commitment_cost=1.0)
    


for prog_batch in val_gen:
    break


RUN_NUM = 5
OUT_DIR = "./output"
os.system(f"mkdir -p {OUT_DIR}")

progress = Progress(prog_batch, OUT_DIR, RUN_NUM)


history = vq_gan.fit(train_gen,
                     epochs=500,
                     steps_per_epoch=N // BATCH_SIZE,
                     callbacks=[progress],
                     initial_epoch=0,
                     validation_data=val_gen,
                     validation_steps=val_N // BATCH_SIZE
                     )


history = vq_gan.history

run_data = history.history

pickle.dump(run_data, open(f'run_{RUN_NUM}_data.pkl', 'wb'))

n = len(history.history["zq_norm"])

for key in history.history:
    plt.plot(range(n), history.history[key], label=key)

plt.legend()
plt.savefig(f"./run_{RUN_NUM}_z1024_history.png")
plt.close()



