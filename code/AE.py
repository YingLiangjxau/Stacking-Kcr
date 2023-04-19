"""
-- Descirption: Autoencoder
-- Authors: Yuan Zhang, Fei Ye, Dapeng Xiong, Xieping Gao
-- Institutions:
      School of Mathematics and Computational Science, Xiangtan University,411105, Xiangtan, China.
      Key Laboratory of Intelligent Computing and Information Processing of Ministry of Education, Xiangtan University, 411105, Xiangtan, China
      Department of Biological Statistics and Computational Biology, Cornell University, 14853, Ithaca, New York, USA.
      Weill Institute for Cell and Molecular Biology, Cornell University, 14853,Ithaca, New York, USA.
      College of Medical Imaging and Inspection, Xiangnan University, Chenzhou 423000, China.
-- Date:2020.5
-- If you want to use this code, please cite this article.

-- Reference:
      Zhang, Y., Ye, F., Xiong, D. et al. LDNFSGB: prediction of long non-coding rna and disease association using network feature similarity and gradient boosting. BMC Bioinformatics 21, 377 (2020). https://doi.org/10.1186/s12859-020-03721-0

"""

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # cpu
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math
import random
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import minmax_scale, scale
from keras.callbacks import EarlyStopping



def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


CKSAAPFeature = []
ReadMyCsv(CKSAAPFeature, "CKSAAP-3.csv")  #1600
EAACFeature = []
ReadMyCsv(EAACFeature, "EAAC-5.csv")      #540

SampleFeature=np.concatenate((CKSAAPFeature, EAACFeature),axis=1)


SampleFeature = np.array(SampleFeature)
x = SampleFeature
print(type(x))
print(len(x))


from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(x, test_size=0.3)


x_train = x_train.astype('float64') / 1.
x_test = x_test.astype('float64') / 1.


def data_in_one(inputdata):
    min = np.nanmin(inputdata)
    max = np.nanmax(inputdata)
    outputdata = (inputdata - min) / (max - min)
    return outputdata

x_train=data_in_one(x_train)
x_test=data_in_one(x_test)

x_train = np.array(x_train)
x_test = np.array(x_test)


print(x_train.shape)
print(x_train)
print(x_test.shape)
print(x_test)


encoding_dim1 = 128
encoding_dim2 = 32

input_img = Input(shape=(len(SampleFeature[0]),))

# 构建autoencoder
encoded_input = Input(shape=(encoding_dim2,))
encoded1 = Dense(encoding_dim1, activation='relu')(input_img)
encoded2 = Dense(encoding_dim2, activation='relu')(encoded1)

decoded1 = Dense(encoding_dim1, activation='relu')(encoded2)
decoded2 = Dense(2140, activation='sigmoid')(decoded1)

autoencoder = Model(inputs=input_img, outputs=decoded2)
decoder_layer = autoencoder.layers[-1]
encoder = Model(inputs=input_img, outputs=encoded2)

# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=100, batch_size=64, shuffle=True, validation_data=(x_test, x_test))    #128--64

# autoencoder.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='binary_crossentropy')
# autoencoder.fit(x_train, x_train, epochs=100, batch_size=128, shuffle=True, validation_data=(x_test, x_test),callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)])


# encoded_imgs = encoder.predict(x)
encoded_imgs = autoencoder.predict(x)
storFile(encoded_imgs, 'AE(out).csv')










