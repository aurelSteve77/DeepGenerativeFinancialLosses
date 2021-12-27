# For inference

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam

from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot


# load model 
model = load_model('/content/drive/MyDrive/BNPP_Project/outcoder_gan_submission2/best_0.86_0.017/_generator_model_step_499.h5')

#Generation of random noise to feed the generator
#z = tf.random.normal((46, 4 ),seed =10)



# loading the noise used to train our model
df = pd.read_csv('/content/drive/MyDrive/BNPP_Project/outcoder_gan_submission2/best_0.86_0.017/noise_499.csv', header = None)
df = df.iloc[2:410,:]
df.to_csv('/content/drive/MyDrive/BNPP_Project/outcoder_gan_submission2/best_0.86_0.017/noise.csv', index=None, header= None)
z= df.to_numpy()
#print(z)


# checking performance of the trained model
# prediction
gen_data = model.predict(z)
gen_data = pd.DataFrame(gen_data)
print(gen_data)
gen_data.to_csv('/content/drive/MyDrive/BNPP_Project/outcoder_gan_submission2/best_0.86_0.017/gen.csv',index = None, header=None)
gen_data.to_csv('/content/gen.csv',header=None)