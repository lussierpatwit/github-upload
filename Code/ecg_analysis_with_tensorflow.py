# -*- coding: utf-8 -*-
"""ECG-Analysis-with-tensorflow.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1orsRwFi4JhR1A-q5g9H7z4g7XpftuCJR

# Paul Lussier 
# Created: 09/17/21 
This notebok is a copy of the one I created to learn and understand what Abhi Pote Shrestha and Chen-Hsiang Yu did in their research. In this notebook I will try and apply a convolutional neural net using keras and tensorflow to get a better predictor of heart rhythm
"""

pip install wfdb

# pip uninstall matplotlib

pip install matplotlib==3.1.3

import wfdb 
import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print(tf.__version__)

test_segments = []
test_records = ["16265", "16272", "16273", "16420", "16483", 
               "16539", "16773", "16786", "16795", "17052", 
               "17453", "18177", "18184", "19088", "19090", 
               "19093", "19140", "19830"]

testSig = wfdb.rdrecord("16265",
                        channels=[0],
                        sampfrom=100,
                        sampto=400,
                        pn_dir='nsrdb/')
wfdb.plot.plot_wfdb(testSig)

# segmentData is a method to get the ECG data from PhysioNet into the program for analysis and model
# building - consolidated versionof work done by Sherasth et. al. durign research with Prof. Jones Yu of Wentworth Institute of Technology
# Returns: List of numpy arrays with a classification number as last element of each
# Parameters: 
#   recordList: list of records as shown in PhysioNet data page stored as strings
#   recordPath: path to the PhysioNet page ie. 'nsrdb/'
#   numSamples: number of samples being read by wfdb.rdrecord()
#   classification: the number assigned to the type of heartrate being read NSR-1 CHF-2 AF-3
#   segmentSize: number of data points in each segment
def segmentData(recordList, recordPath, numSamples, classification, segmentSize):

  segments_out = [] # List of segments to be output to the variable caling the function

  print("--Start of segmentation for type", classification,"--")
  print("Len recordList:", len(recordList))

  for record in recordList: # start of loop that runs length of passed in list of records
    # curr is a temp variable holding the record being added to the final list
    curr = wfdb.rdrecord(record, channels=[0], sampto=numSamples, pn_dir=recordPath) 
    samples = curr.__dict__['p_signal']
    n=len(samples)

    # print("Record:", record, "Length:", n, "Segments:", n//segmentSize)
 
    start = 0
    while start < n: # Start while loop
      if (n-start < segmentSize):
        # if the length - the displacement of the start is less than 5000 then the loop
        # has beed through all of the segments and ends
        break
      else:
        segment = samples[start:start+segmentSize] # creates a numpy array and fills it with the next [segmentSize] values
        segment = np.append(segment,[classification]) # adds a number to the end so the type of rhythm can be ID'd
        segments_out.append(segment) # adds the finished segment to the output list with its calssification tacked on
        start += segmentSize # iterates start to the next segment
    # End while loop
  # End for loop
  print("Record:", record, "Length:", n, "Segments:", n//segmentSize)
  print("--End of segmentation for type", classification,"--")
  return segments_out

nsr_records = ["16265", "16272", "16273", "16420", "16483",
               "16539", "16773", "16786", "16795", "17052", 
               "17453", "18177", "18184", "19088", "19090", 
               "19093", "19140", "19830"]


# chf_records = ["chf01", "chf02", "chf03", "chf04", "chf05",
               # "chf06", "chf07", "chf08", "chf09", "chf10",
               # "chf11", "chf12", "chf13", "chf14", "chf15"]

# chf records with bad data removed
chf_records = ["chf01", "chf03", "chf04", "chf05", "chf07",
               "chf08", "chf09", "chf10", "chf11", "chf12",
               "chf13", "chf14", "chf15"]

# af_records = ["04015", "04043", "04048", "04126", "04746", 
              # "04908", "04936", "05091", "05121", "05261", 
              # "06426", "06453", "06995", "07162", "07859", 
              # "07879", "07910", "08215", "08219", "08378", 
              # "08405", "08434", "08455"]

# af records with bad data removed
af_records = ["04015", "04043", "04048", "04126", "04746",
              "04908", "05121", "05261", "06426", "06453",
              "06995", "07162", "07859", "07879", "07910",
              "08215", "08219", "08378", "08405", "08434",
              "08455"]

nsr_path = 'nsrdb/'
chf_path = 'chfdb/'
af_path = 'afdb/'

nsr_numSamples = 700000
chf_numSamples = 1000000
af_numSamples = 650000

segmentSize = 5000

nsr_segments = np.array(segmentData(nsr_records, nsr_path, nsr_numSamples,1,segmentSize))
chf_segments = np.array(segmentData(chf_records,chf_path,chf_numSamples,2,segmentSize))
af_segments = np.array(segmentData(af_records,af_path,af_numSamples,3,segmentSize))

# Separating labels from segments for all three rhythm types
nsrSegNp = np.array(nsr_segments)
chfSegNp = np.array(chf_segments)
afSegNp = np.array(af_segments)

nsr_labels = nsrSegNp[0:,segmentSize]
nsr_labels = nsr_labels.reshape(nsr_labels.shape[0],1)
nsrSegNp = np.delete(nsrSegNp,segmentSize,1)

chf_labels = chfSegNp[0:,segmentSize]
chf_labels = chf_labels.reshape(chf_labels.shape[0],1)
chfSegNp = np.delete(chfSegNp,segmentSize,1)

af_labels = afSegNp[0:,segmentSize]
af_labels = af_labels.reshape(af_labels.shape[0],1)
afSegNp = np.delete(afSegNp,segmentSize,1)
print("nsrSegNp shape:",nsrSegNp.shape)
print("chfSegNp shape:",chfSegNp.shape)
print("afSegNp shape:",afSegNp.shape)
print("nsr labels shape:",nsr_labels.shape)
print("chf labels shape:",chf_labels.shape)
print("af labels shape:",af_labels.shape)
# print(nsr_labels[0:10])
# print(chf_labels[0:10])
# print(af_labels[0:10])

# Resampling nsr data from 128 Hz to 250 Hz using np.interp
x250 = np.arange(0,20,0.004)
x128 = np.arange(0,39.06,0.0078125)

nsrResamp = np.zeros_like(nsrSegNp)

for i in range(0,len(nsrSegNp)):
  nsrResamp[i] = np.interp(x250,x128,nsrSegNp[i])
print("nsrSegNp shape:",nsrSegNp.shape)
print("nsrResamp shape:",nsrResamp.shape)

# nsrSegNp = np.append(nsrResamp,nsr_labels,axis = 1)

# print(type(nsr_segments))
# print(nsr_segments.shape)

plt.figure(100)
plt.plot(nsr_segments[0])
plt.figure(200)
plt.plot(nsrResamp[0])
plt.savefig('nsrResampExample.png',dpi=128)
# Visual confirmation of resampled data

# setting segments varaibels to the properly sampled arrays with labels removed
nsr_segments = nsrResamp
chf_segments = chfSegNp
af_segments = afSegNp

# Adding back lables to segemts 
nsr_segments = np.append(nsr_segments,nsr_labels,axis = 1)
chf_segments = np.append(chf_segments,chf_labels,axis = 1)
af_segments = np.append(af_segments,af_labels,axis = 1)

print(type(nsr_segments))
# print(nsr_segments)
# print(chfSegNp)
print("Total number of segments for Normal Sinus Rhythm:",len(nsr_segments))
print("Total number of segments for Congestive Heart Failure:",len(chf_segments))
print("Total number of segments for Atrial Fibrillation:",len(af_segments))
print("nsr_segments shape:",nsr_segments.shape)
print("chf_segments shape:",chf_segments.shape)
print("af_segments shape:",af_segments.shape)

# nsr_segments = nsr_segments[:2500]
# chf_segments = chf_segments[:2500]
# af_segments = af_segments[:2500]

nsr_segments = nsr_segments[:2500]
chf_segments = chf_segments[:2500]
af_segments = af_segments[:2500]

print("Total number of segments for Normal Sinus Rhythm:",nsr_segments.shape)
print("Total number of segments for Congestive Heart Failure:",chf_segments.shape)
print("Total number of segments for Atrial Fibrillation:",af_segments.shape)

# Compile all the segments into a single matrix
full_data = np.concatenate((nsr_segments, chf_segments,af_segments),axis = 0)
print("Number of total segments:",full_data.shape)

# Split the array into training and testing arrays randomly
train_data, test_data = train_test_split(full_data, test_size=0.3,random_state=42)

train_df = pd.DataFrame(data = train_data,index=np.array(range(len(train_data))),
                        columns=np.array(range(segmentSize+1)))
train_df.rename(columns={segmentSize:'Label'},inplace=True)
train_df

train_features = train_df.drop(['Label'],axis=1).values
train_labels = train_df['Label']
train_labels

test_df = pd.DataFrame(data=test_data,
          index=np.array(range(len(test_data))),
          columns=np.array(range(segmentSize+1)))
test_df.rename(columns={segmentSize:'Label'},inplace=True)
test_df

test_features = test_df.drop(['Label'], axis=1).values
test_labels = test_df['Label']
print(np.shape(train_features))
print(np.shape(test_features))
train_features = train_features.reshape(train_features.shape[0],train_features.shape[1],1)
test_features = test_features.reshape(test_features.shape[0],test_features.shape[1],1)
print(np.shape(train_features))
print(np.shape(test_features))

# One hot encoding the labels for use later in the tensorflow model
categorical_train_labels = tf.keras.utils.to_categorical(train_labels)
categorical_test_labels = tf.keras.utils.to_categorical(test_labels)

# Deleting the vestigial column 0 from the final product 
categorical_train_labels = np.delete(categorical_train_labels,0,axis=1)
categorical_test_labels = np.delete(categorical_test_labels,0,axis=1)

print(categorical_train_labels)
print(categorical_test_labels)

# Definition of each layer in the model
model_tf = tf.keras.models.Sequential()
d = 0.6
model_tf.add(layers.Conv1D(filters=32,
                           kernel_size=(1,),
                           activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                           input_shape=(5000,1,)))
model_tf.add(layers.Conv1D(32,3,activation='relu'))
model_tf.add(layers.Conv1D(32,3,activation='relu'))
model_tf.add(layers.Dropout(d))

model_tf.add(layers.GlobalAveragePooling1D())

model_tf.add(layers.Dense(64,activation='relu'))
model_tf.add(layers.Dropout(d))

model_tf.add(layers.Dense(3,activation='softmax'))

model_tf.summary()

model_tf.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

checkpoint_path = "/content/models/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,monitor='val_accuracy', mode='max', save_best_only=True)

# categorical labels
history = model_tf.fit(train_features,categorical_train_labels,epochs=10,
                       validation_data=(test_features,categorical_test_labels),
                        callbacks=[model_checkpoint])

# Testing the checkpoint stuff 
model_tf.load_weights(checkpoint_path)

# Testign accuract of the model saved by checkpoints
loss, acc = model_tf.evaluate(test_features, categorical_test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Accuracy over epochs graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss over epochs graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

loss, acc = model_tf.evaluate(test_features, categorical_test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Querying the model to get prediction 
prediction = model_tf.predict(test_rhythm)
for i in range(len(prediction)):
  prediction[i] = np.round(prediction[i],decimals=1)
print(categories[np.argmax(prediction,axis=None,out=None)])