#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:15:40 2019

@author: user
"""
""" import package """
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import load_model
from keras.utils  import np_utils
import os
import cv2
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#TA_PATH = "C:/Users/milk3/Documents/Machine Learning/HW5/105011240 HW5"
""" load xdata """
X=glob.glob(r"C:/Users/milk3/Documents/Machine Learning/HW5/computer_assignment5/computer_assignment5/dataset/*/*.png")
y=[]
for i in range(0,10):
    for j in range(500):
        y.append(i)

train_image = []
for i in range(len(X)):
    img=image.img_to_array(cv2.imread(X[i]))
    img = img/255
    train_image.append(img)

X = np.array(train_image)    
y = np.array(y)
le=preprocessing.LabelEncoder()
le.fit(y)
y=le.transform(y)

x_train,x_valid,y_train,y_valid = train_test_split(X,y,test_size = 0.2,random_state=40)
#x_train = np.load(TA_PATH+'x_train.npy')
#x_test  = np.load(TA_PATH+'x_test.npy')
""" import your package """
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
""" define your own preprocessing """
BATCH_SIZE = 30
EPOCH = 70
LEARNINR_RATE = 0.001
NUM_CLASS = 10

y_train = np_utils.to_categorical(y_train, NUM_CLASS)
y_valid = np_utils.to_categorical(y_valid, NUM_CLASS)

def vehicle_model():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(8, (1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    return model

model=vehicle_model()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],
              optimizer=Adam(lr=LEARNINR_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
History = model.fit(x_train, y_train,
                    batch_size = BATCH_SIZE,
                    epochs = EPOCH,
                    validation_data = (x_valid, y_valid))
model.save('C:/Users/milk3/Documents/Machine Learning/HW5/model.h5')

#test
model = load_model('C:/Users/milk3/Documents/Machine Learning/HW5/model.h5')

x_test=np.load('C:/Users/milk3/Documents/Machine Learning/HW5/computer_assignment5/computer_assignment5/sample/x.npy')
y_test=np.load('C:/Users/milk3/Documents/Machine Learning/HW5/computer_assignment5/computer_assignment5/sample/y.npy')
y_test = np_utils.to_categorical(y_test, NUM_CLASS)
model.predict(x=x_test)
test_loss , test_acc = model.evaluate(x_test, y_test)
print("test acc %.2f%%" %(test_acc*100))

'''
""" Test part (cannot change) """
y_train = np.load(TA_PATH+'y_train.npy')
y_test  = np.load(TA_PATH+'y_test.npy')
y_train = np_utils.to_categorical(y_train, 10)
y_test  = np_utils.to_categorical(y_test, 10)

model = load_model("model.h5")
train_loss , train_acc = model.evaluate(x_train, y_train)
test_loss , test_acc = model.evaluate(x_test, y_test)
print("train acc %.2f%%" %(train_acc*100), ",test acc %.2f%%" %(test_acc*100))


