#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:37:51 2020

@author: anpandoh
"""

#Exlamation point run directly from shell did not work, so downloaded from terminal
################################################################################################################
#Initial Imports
#Keras is API use to build and train models | Hub is a library
import numpy as np

import tensorflow as tf

#!pip install -q tensorflow-hub
#!pip install -q tfds-nightly

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

############################################################################################################

#Downloading IMDB movie review data set

# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

############################################################################################################
#looking at data set

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
print("\n")
print (train_labels_batch)

###########################################################################################################
#setting up the layers
#dense is the amount of ouputs that will occur from that layer (EXPERIMENT WITH THE AMOUNT OF NODES[outputs])
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(16))
model.add(tf.keras.layers.Dense(1))

model.summary()
#I have learned that additional nodes and layers don't nessecarily make the model more accurate/have less error but
#it increases the accuracy during training
############################################################################################################
#loss function - measure accuracy of model during training (EXPERIMENT WITH DIFFERENT LOSS FUNCTION [mean_squared_error])
#optimizer, how model will change depending on loss function
#mterics - monitor the training and testing steps 

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              #loss=tf.keras.losses.MeanSquaredError(from_logits=True)
                  #reduction=losses_utils.ReductionV2.AUTO, name = 'mean_squared_error'),
              metrics=['accuracy'])
              #metrics = [keras.metrics.MeanSquaredError()]
print("\n \n \n \n \n")
#MeanSquaredError seemed to have a higher loss but I was unable to figure out metrics and final results
#############################################################################################################
#Training the function
#epoch - iterations of of model on a sample

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)
print("\n \n \n \n \n")
#displays accuracy of model
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

#for value in results:
    #print("%s: %.3f" % (value))





