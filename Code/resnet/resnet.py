import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import models,layers
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2


IMAGE_SIZE=227
BATCH_SIZE=32
CHANNELS=3

dataset=tf.keras.preprocessing.image_dataset_from_directory(
    r"/rwthfs/rz/cluster/home/jo444146/techlabsWS22/Driver Drowsiness Dataset (DDD)",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names=dataset.class_names
print("class names: ",class_names)

for image_batch,label_batch in dataset.take(1):
    print('image batch shape: ',image_batch.shape)
    print('label_batch: ',label_batch.numpy())

for image_batch,label_batch in dataset.take(1):
    print('image_bacth shape: ',image_batch[0].shape)

train_size=0.8
train_ds=dataset.take(1045)
test_ds=dataset.skip(1045)
val_size=0.1
val_ds=test_ds.take(130)
test_ds=test_ds.skip(130)

def get_dataset_partitions_tf(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=10000):
    
    ds_size=len(ds)
    
    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=12)
    
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)

    train_ds=ds.take(train_size)
    
    val_ds=ds.skip(train_size).take(val_size)
    test_ds=ds.skip(train_size).skip(val_size)
    
    return train_ds,val_ds,test_ds


resize_and_rescale=tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augmentation=tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    layers.experimental.preprocessing.RandomRotation(0.2)
])


def res_identity(x, filters):     
    #resnet identity block  

    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters

    #first block 
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    #second block # bottleneck (but size kept same with padding)
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # add the input 
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activations.relu)(x)

    return x



def res_conv(x, s, filters):
    # resnet convolutional building block

    x_skip = x  # this will be used for addition with the residual block 
    f1, f2 = filters

    # first block
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    # second block
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    #third block
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)

    # shortcut 
    x_skip = layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
    x_skip = layers.BatchNormalization()(x_skip)

    # add 
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activations.relu)(x)

    return x




def resnet50(shape = [227, 227, 3]):
      
    # construct resnet 50
    input_im = Input(shape=(shape[0], shape[1], shape[2])) # image size
    x = layers.ZeroPadding2D(padding=(3, 3))(input_im)

    # 1st stage
    # maxpooling
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    #2nd stage 
    # from here on only conv block and identity block
    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    # 3rd stage
    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    # 4th stage
    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    # 5th stage
    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    # Average pooling and dense connection at the end
    x = layers.AveragePooling2D((2, 2), padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x) #binary classification

    # define the model 
    model = Model(inputs=input_im, outputs=x, name='Resnet50')

    return model




shape = [227, 227, 3]
model = resnet50(shape)
print(model.summary())
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


history=model.fit(
    train_ds,
    epochs=5,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)
