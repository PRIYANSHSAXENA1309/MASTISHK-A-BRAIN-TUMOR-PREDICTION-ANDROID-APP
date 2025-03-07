import cv2
import os
import imutils
import numpy as np
from tensorflow import keras
from keras.regularizers import l2
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Normalization, Input, BatchNormalization

def augmented_data(file_dir, n_generated_samples, save_to_dir):
    data_gen = ImageDataGenerator(
        rotation_range = 30,
        zoom_range = 0.2, 
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.1,
        brightness_range = (0.3, 1.0),
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'nearest'
    )
    for filename in os.listdir(file_dir):
        image = cv2.imread(file_dir + '/' + filename)
        image = image.reshape((1,) + image.shape)
        save_prefix = 'aug_' + filename[:-4]
        i = 0
        for batch in data_gen.flow(x = image, batch_size = 1, save_to_dir = save_to_dir, save_prefix = save_prefix, save_format = "jpg"):
            i += 1
            if i >= n_generated_samples:
                break

def crop_brain_tumor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    thres = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thres = cv2.erode(thres, None, iterations = 2)
    thres = cv2.dilate(thres, None, iterations = 2)
    
    cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)
    extLeft = tuple(c[c[:,:,0].argmin()][0])
    extRight = tuple(c[c[:,:,0].argmax()][0])
    extTop = tuple(c[c[:,:,1].argmin()][0])
    extBot = tuple(c[c[:,:,1].argmax()][0])
    
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]] 
    return new_image

def load_data(dir_list, image_size):
    X = []
    y = []
    
    for directory in dir_list:
        for filename in os.listdir(directory):
            image = cv2.imread(directory + '/' + filename)
            image = crop_brain_tumor(image)
            image = cv2.resize(image, dsize = image_size, interpolation = cv2.INTER_CUBIC)
            image = image.astype('float32') / 255.0
            X.append(image)
            if directory[-3:] == "yes":
                y.append(1)
            else:
                y.append(0)

    X = np.array(X)
    y = np.array(y)
    X,y = shuffle(X,y)
    return X,y

dataset = "D:/Dataset/"
yes_path = dataset + 'yes' 
no_path = dataset + 'no'

if not os.path.isdir(dataset + 'augmented'):
    os.mkdir(dataset + 'augmented')
if not os.path.isdir(dataset + 'augmented/yes'):
    os.mkdir(dataset + 'augmented/yes')
if not os.path.isdir(dataset + 'augmented/no'):
    os.mkdir(dataset + 'augmented/no')

augmented_data(file_dir = yes_path, n_generated_samples = 6, save_to_dir = dataset + 'augmented/yes')
augmented_data(file_dir = no_path, n_generated_samples = 9, save_to_dir = dataset + 'augmented/no')

augmeneted_yes = dataset + 'augmented/yes'
augmeneted_no = dataset + 'augmented/no'

img_size = (64,64)
X,y = load_data([augmeneted_yes, augmeneted_no], img_size)

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, shuffle = True)

normalize_layer = Normalization()
normalize_layer.adapt(X_train)

# Neural Network Model
model=Sequential()
model.add(normalize_layer)
model.add(Conv2D(64, (3,3), kernel_regularizer=l2(0.001), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(32,(3,3), kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(16,(3,3), kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy', Precision(), Recall()])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
model.fit(X_train,y_train,batch_size=6,epochs=50,validation_data=(X_val,y_val), callbacks=[early_stopping, lr])
loss, accuracy, precision, recall = model.evaluate(X_test,y_test)

print("Model Loss:", loss)
print("Model Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)

model.save('BrainTumorModel3.keras') 