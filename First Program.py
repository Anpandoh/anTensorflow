#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:27:13 2020

@author: anpandoh
"""
import tensorflow as tf
import IPython.display as display
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
AUTOTUNE = tf.data.experimental.AUTOTUNE


import pathlib
#Training Data
data_dir = tf.keras.utils.get_file(origin='http://localhost/SamplePhotos.tar.gz',
                                   fname='SamplePhotos', untar=True)
data_dir = pathlib.Path(data_dir)

print (data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

#Validation Data
val_dir = tf.keras.utils.get_file(origin='http://localhost/Validation.tar.gz',
                                   fname='Validation', untar=True)
val_dir = pathlib.Path(val_dir)
vimage_count = len(list(val_dir.glob('*/*.jpg')))
print(vimage_count)
######################################################################################################
#Training Data
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
print(CLASS_NAMES)
label_count = (len(CLASS_NAMES))       
Mountain = list(data_dir.glob('Mountain/*'))

for image_path in Mountain[:1]:
    display.display(Image.open(str(image_path)))

#Validation Data
vCLASS_NAMES = np.array([item.name for item in val_dir.glob('*')])
print(vCLASS_NAMES)
vlabel_count = (len(vCLASS_NAMES))  
Beach = list(val_dir.glob('Beach/*'))

for image_path in Beach[:1]:
    display.display(Image.open(str(image_path)))
######################################################################################################
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                  rotation_range=45,
                                                                 # width_shift_range=.15,
                                                                  #height_shift_range=.15,
                                                                  #horizontal_flip=True,
                                                                  zoom_range=0.5
                                                                  )
BATCH_SIZE = 20
IMG_HEIGHT = 32
IMG_WIDTH = 32
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))

vimage_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
vSTEPS_PER_EPOCH = np.ceil(vimage_count/BATCH_SIZE)

val_data_gen = vimage_generator.flow_from_directory(directory=str(val_dir),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    classes = list(vCLASS_NAMES))



#####################################################################################################
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(20):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')


image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)

plt.figure()
plt.imshow(image_batch[0])
plt.colorbar()
plt.grid(False)
plt.show()


print(image_batch.shape)
print(label_batch.shape)

print(label_batch[0])
#######################################################################################################
#Model Creation
#adding dropout to reduce overfitting

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Softmax, BatchNormalization


model = Sequential()

#Convolution Layer
model.add(Conv2D(32, (5,5), activation = "relu", input_shape = (IMG_HEIGHT,IMG_WIDTH,3)))
model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))

#MaxPooling Layer, reducing to 16x16
model.add(MaxPooling2D(pool_size=(2,2)))
Dropout(0.1)

#Convolution Layer
model.add(Conv2D(32, (5,5), activation = "relu"))
model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=0.001))

#MaxPooling Layer, reducing to 16x16
model.add(MaxPooling2D(pool_size=(2,2)))

#Flatten Layer to make data 1 Dimension
model.add(Flatten())

model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=0.001))
model.add(Dense(1000,activation="relu"))
model.add(Dense(label_count,activation="softmax" ))


epochs=40
#####################################################################################################
#Compile Layer

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

####################################################################################################
#Train Model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=vSTEPS_PER_EPOCH
)
####################################################################################################

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#################################################################################################
vimage_batch, vlabel_batch = next(val_data_gen)

predictions= model.predict(val_data_gen)

print('predictions:',predictions.shape)
print(predictions[1])



y=0


vimage_batch, vlabel_batch = next(val_data_gen)
show_batch(vimage_batch, vlabel_batch)
for n in range(20):
    print("Guess:",vCLASS_NAMES[np.argmax(predictions[n])], "      Actual:", vCLASS_NAMES[vlabel_batch[n]==1][0].title())
    if vCLASS_NAMES[np.argmax(predictions[n])] == vCLASS_NAMES[vlabel_batch[n]==1][0].title():
        y = (y+1)
        
print(y/20)

def show_batch(vimage_batch, vlabel_batch):
  plt.figure(figsize=(10,10))
  for n in range(20):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(vimage_batch[n])
      plt.title(vCLASS_NAMES[vlabel_batch[n]==1][0].title())
      plt.axis('off')
      
plt.figure()
plt.imshow(vimage_batch[1])
plt.colorbar()
plt.grid(False)
plt.show()



