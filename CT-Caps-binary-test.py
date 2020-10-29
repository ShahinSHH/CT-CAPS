#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-CAPS binary test code implementation.

!! Note: CT-CAPS framework is in the research stage. Use only for research ourposes at this time.
Don't use CT-CAPS as a replacement of the clinical test and radiologist review.

Created by: Shahin Heidarian, Msc. at Concordia University
E-mail: s_idari@encs.concordia.ca

** The code for the Capsule Network implementation is adopted from https://keras.io/examples/cifar10_cnn_capsule/.
"""

#%% Libraries
from __future__ import print_function
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, auc
import os
import matplotlib.pyplot as plt
import pydicom
import cv2
from lungmask import mask #lung segmentation model
import SimpleITK as sitk


K.set_image_data_format('channels_last')

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)


class Capsule(Layer):
   

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.num_capsule,
        'dim_capsule' : self.dim_capsule,
        'routings':  self.routings,
        'share_weight':self.share_weights,
        
       
           
        })
        return config

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(keras.backend.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = keras.backend.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
    
# normalization function
def normalize_image(x): #normalize image pixels between 0 and 1
        if np.max(x)-np.min(x) == 0 and np.max(x) == 0:
            return x
        elif np.max(x)-np.min(x) == 0 and np.max(x) != 0:
            return x/np.max(x)
        else:
            return (x-np.min(x))/(np.max(x)-np.min(x))


def segment_lung(mask,model,volume_path):
    
    # model = mask.get_model('unet','R231CovidWeb')
    #loop through all dcm files
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(volume_path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))
    
    dataset = pydicom.dcmread(lstFilesDCM[0]) # a sample image
    slice_numbers = len(lstFilesDCM) #number of slices
    # print('Slices:',slice_numbers)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        # print('Image size:',rows,cols)
    
    slice_z_locations = []
    for filenameDCM in lstFilesDCM:
        ds = pydicom.dcmread(filenameDCM)
        slice_z_locations.append(ds.get('SliceLocation'))
    
    #sorting slices based on z locations
    slice_locations = list(zip(lstFilesDCM,slice_z_locations))
    sorted_slice_locations = sorted(slice_locations, key = lambda x: x[1])[-1::-1]
    
    # Saving Slices in a numpy array
    ArrayDicom = np.zeros((slice_numbers,rows,cols))
    lung_mask = np.uint8(np.zeros((slice_numbers,rows,cols)))
    # loop through all the DICOM files
    i = 0
    for filenameDCM, z_location in sorted_slice_locations:
        # read the file
        ds = sitk.ReadImage(filenameDCM)
        segmentation = mask.apply(ds, model)
        lung_mask[i,:,:] = np.uint8(((segmentation>0)*1)[0])
        ArrayDicom[i, :, :] = sitk.GetArrayFromImage(ds)
        i = i+1
    
    lungs = np.zeros((ArrayDicom.shape[0],256,256,1))    
    # resizing the data
    for i in range(ArrayDicom.shape[0]):
        ct = normalize_image(ArrayDicom[i,:,:])
        mask_l = lung_mask[i,:,:]
        seg = mask_l * ct #apply mask on the image
        img = cv2.resize(seg,(256,256))
        img = normalize_image(img)
        lungs[i,:,:,:] = np.expand_dims(img,axis = -1)
    # print('Successfully segmented.')    
    return lung_mask, ArrayDicom, lungs

def max_vote(x):
    v = np.max(x,axis=0)
    return v

def test_one_dicom(model,X_test):
    # Test
    X_test_normal = np.zeros(X_test.shape)
    for i in range(X_test.shape[0]):
        X_test_normal[i,:,:,:] = normalize_image(X_test[i,:,:,:])
    # check the size    
    if X_test_normal.shape[1] != 256: 
         x_new = np.zeros((X_test_normal.shape[0],256,256,1))
         for i in range(len(X_test_normal)):
             x_new[i,:,:,0] = cv2.resize(X_test_normal[i,:,:,0],(256,256))
         X_test_normal = x_new
    
    sum_seg = np.sum(np.sum(X_test,axis=1),axis=1) 
    a = np.where(sum_seg[:,0] != 0) # to find out if lung exists or not
    X_lung = X_test_normal[a]  
    
    capsules = np.zeros((1,32,16))
    
    if len(X_test_normal)==0:
        capsules[0] = np.zeros((32,16))-1
    else:
        x_capsule = model.predict(X_lung)
        capsules[0] = max_vote(x_capsule)
      
    return capsules

def stage_two_output(x_test,model2,cutoff):
    pred = model2.predict(x_test)
    prob_one = pred[:,1]
    pred_final = (prob_one>=cutoff)*1 # cut-off probability  
    return prob_one, pred_final

#%% Model1 (Feature Extractor)

input_image = Input(shape=(None, None, 1))
x = Conv2D(64, (3, 3), activation='relu',trainable = True)(input_image)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Conv2D(64, (3, 3), activation='relu',trainable = True)(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu',trainable = True)(x)
x = Conv2D(128, (3, 3), activation='relu',trainable = True)(x)
#
x = Reshape((-1, 128))(x)
x = Capsule(32, 16, 3, True)(x)  
x = Capsule(32, 16, 3, True)(x)   
capsule = Capsule(2, 16, 3, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)

model = Model(inputs=[input_image], outputs=[output])

# adam = optimizers.Adam(lr=1e-4) 
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

model.load_weights('weights-2class-v1-71.h5') 
model.summary()
#
input_stage_fe = model.input
output_stage_fe = model.layers[-3].output

model_fe = Model(input_stage_fe,output_stage_fe)

#%% Model2 (Final Patient-Level Classifier)

input_tensor = Input(shape=(32,16))
x2 = Flatten()(input_tensor)
x2 = Dense(256,activation = 'relu')(x2)
x2 = Dense(128,activation = 'relu')(x2)
x2 = Dense(32,activation = 'relu')(x2)
out2 = Dense(2, activation = 'softmax')(x2)

model2 = Model(input_tensor,out2)

# opt = optimizers.Adam(lr=0.001)
# model2.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy']) #adadelta
model2.load_weights('binary-max-v4.h5') 
model2.summary()

#%%
data_path = r'/Users/shahinheidarian/Python Files/Corona/Annotation-SliceClassification/COVID-19 subjects/P101/'
# lstFolders = sorted(os.listdir(data_path))
model_sg = mask.get_model('unet','R231CovidWeb')
#%% Testing
print('Segmenting lung area...')
lung_mask, ArrayDicom, lung = segment_lung(mask,model_sg,data_path)
print('Segmentation is Completed.')

capsules = test_one_dicom(model_fe,lung)

# Stage 2  
cutoff = 0.5 #cut-off probability
prob_one, pred_final = stage_two_output(capsules,model2,cutoff) 
if pred_final==1 :
    prediction = 'COVID-19'
else:
    prediction = 'non-COVID'
    
print('prediction: ',prediction)
      
