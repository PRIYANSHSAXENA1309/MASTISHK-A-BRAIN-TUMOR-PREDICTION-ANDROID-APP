import cv2
import os
import numpy as np
from PIL import Image
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Dense, Normalization, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout

path = 'D:/Dataset/'

yes_tumor = os.listdir(path+"yes/")
no_tumor = os.listdir(path+"no/")

images=[]
labels=[]

for i , name in enumerate(yes_tumor):
    if (name.split(".")[1]=='jpg'):
        img=cv2.imread(path+'yes/'+name)
        img=Image.fromarray(img,'RGB')
        img=img.resize((64,64))
        images.append(img)
        labels.append(1)

for i , name in enumerate(no_tumor):
    if (name.split(".")[1]=='jpg'):
        img=cv2.imread(path+'no/'+name)
        img=Image.fromarray(img,'RGB')
        img=img.resize((64,64))
        images.append(img)
        labels.append(0)

images=np.array(images)
labels=np.array(labels)

X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=0.2,shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, shuffle = True)

normalize_layer = Normalization().adapt(X_train)

# Neural Network Model
model=Sequential(normalize_layer)
model.add(Input(shape = (64,64,3)))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train,y_train,batch_size=8,epochs=20,validation_data=(X_test,y_test), callbacks=[early_stopping])
loss, accuracy= model.evaluate(X_test, y_test)

print("Model Loss:", loss)
print("Model Accuracy:", accuracy)
model.save('BrainTumorModel1.keras')