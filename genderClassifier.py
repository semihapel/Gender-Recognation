# -*- coding: utf-8 -*-
"""
Created on Fri May  1 00:18:33 2020

@author: Apel
"""

import pandas as pd 
import matplotlib.pyplot as plt

train_path="C:/Project/data/veriler/training_set/"
test_path="C:/Project/data/veriler/test_set/"

from keras.preprocessing.image import load_img,img_to_array
img=load_img(train_path+"/erkek/AbdA_00005_m_31_i_fr_nc_hp_2016_2_e0_nl_o.jpg")
plt.imshow(img)
x=img_to_array(img)
img_shape=x.shape
print("image shape : ",img_shape)


from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten 
from keras.preprocessing.image import ImageDataGenerator


##CNN Model 

model= Sequential()

model.add(Conv2D(16,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


## Data generator 
train_datagen=ImageDataGenerator(rescale=1/.255,shear_range=0.3
                                 ,horizontal_flip="true",zoom_range=0.30)
test_datagen=ImageDataGenerator(rescale=1/.255,shear_range=0.3
                                 ,horizontal_flip="true",zoom_range=0.30)
train_d=train_datagen.flow_from_directory(train_path,target_size=(64,64),
                                          batch_size=10,class_mode='binary',
                                          color_mode='rgb')

test_d=train_datagen.flow_from_directory(test_path,target_size=(64,64),
                                          batch_size=10,class_mode='binary',
                                          color_mode='rgb')

hist=model.fit_generator(generator=train_d,steps_per_epoch=8000//5
                         ,epochs=5,validation_data=test_d,
                         validation_steps=2000/5)

######
pred=model.predict_generator(test_d,verbose=1)
#pred = list(map(round,pred))
pred[pred > .5] = 1
pred[pred <= .5] = 0


hist_df=pd.DataFrame(hist.history)
save_path="cnn_mnist_hist.csv"
with open(save_path, 'w') as f:
    hist_df.to_csv(f)        
