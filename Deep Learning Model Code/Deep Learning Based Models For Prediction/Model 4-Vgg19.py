import os
import cv2
import shutil
import imutils
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def augmented_data(file_dir, n_generated_samples, save_to_dir):
    data_gen = ImageDataGenerator(
        rotation_range = 10, 
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

dataset = 'D:/Dataset/'
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

img_size = (240,240)
X,y = load_data([augmeneted_yes, augmeneted_no], img_size)

base_dir = dataset + '/tumorous_and_nontumorous'

if not os.path.isdir(base_dir):
    os.mkdir(base_dir)
if not os.path.isdir(base_dir + '/train'):
    train_dir = os.path.join(base_dir , 'train')
    os.mkdir(train_dir)
if not os.path.isdir(base_dir + '/test'):
    test_dir = os.path.join(base_dir , 'test')
    os.mkdir(test_dir)
if not os.path.isdir(base_dir + '/valid'):
    valid_dir = os.path.join(base_dir , 'valid')
    os.mkdir(valid_dir)
if not os.path.isdir(train_dir + '/tumorous'):
    infected_train_dir = os.path.join(train_dir, 'tumorous')
    os.mkdir(infected_train_dir)
if not os.path.isdir(test_dir + '/tumorous'):
    infected_test_dir = os.path.join(test_dir, 'tumorous')
    os.mkdir(infected_test_dir)
if not os.path.isdir(valid_dir + '/tumorous'):
    infected_valid_dir = os.path.join(valid_dir, 'tumorous')
    os.mkdir(infected_valid_dir)
if not os.path.isdir(train_dir + '/nontumorous'):
    healthy_train_dir = os.path.join(train_dir, 'nontumorous')
    os.mkdir(healthy_train_dir)
if not os.path.isdir(test_dir + '/nontumorous'):
    healthy_test_dir = os.path.join(test_dir, 'nontumorous')
    os.mkdir(healthy_test_dir)
if not os.path.isdir(valid_dir + '/nontumorous'):
    healthy_valid_dir = os.path.join(valid_dir, 'nontumorous')
    os.mkdir(healthy_valid_dir)

yes=os.listdir(augmeneted_yes)
no=os.listdir(augmeneted_no)

i=0
while(i < int(0.7*len(yes))):
    shutil.copyfile(augmeneted_yes + '/' + yes[i],infected_train_dir + '/' + yes[i])
    i += 1
while(i < int(0.85*len(yes))):
    shutil.copyfile(augmeneted_yes + '/' + yes[i],infected_test_dir + '/' + yes[i])
    i += 1
while(i < len(yes)):
    shutil.copyfile(augmeneted_yes + '/' + yes[i],infected_valid_dir + '/' + yes[i])
    i += 1

i=0
while(i < int(0.7*len(no))):
    shutil.copyfile(augmeneted_no + '/' + no[i],healthy_train_dir + '/' + no[i])
    i += 1
while(i < int(0.85*len(no))):
    shutil.copyfile(augmeneted_no + '/' + no[i],healthy_test_dir + '/' + no[i])
    i += 1
while(i < len(no)):
    shutil.copyfile(augmeneted_no + '/' + no[i],healthy_valid_dir + '/' + no[i])
    i += 1

datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = 0.4,
    vertical_flip = 0.4,
    rotation_range = 40,
    shear_range = 0.2,
    width_shift_range = 0.4,
    height_shift_range = 0.4,
    fill_mode = 'nearest'
)

train_generator = datagen.flow_from_directory(train_dir, batch_size = 32, target_size = (240,240), class_mode = 'categorical',shuffle = True, seed = 42, color_mode = 'rgb')
test_generator = datagen.flow_from_directory(test_dir, batch_size = 32, target_size = (240,240), class_mode = 'categorical',shuffle = True, seed = 42, color_mode = 'rgb')
valid_generator = datagen.flow_from_directory(valid_dir, batch_size = 32, target_size = (240,240), class_mode = 'categorical',shuffle = True, seed = 42, color_mode = 'rgb')

class_name = {0: 'nontumorous', 1: 'tumorous'}

base_model = VGG19(input_shape = (240,240,3), include_top = False, weights = 'imagenet')
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation = 'relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation = 'relu')(drop_out)
output = Dense(2, activation = 'softmax')(class_2)
model_01 = Model(base_model.input, output)

es = EarlyStopping(monitor = 'val_loss', verbose = 1, mode = 'min',patience = 4)
cp = ModelCheckpoint('D:/Dissertation/model.keras', monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto',save_freq = 'epoch')
lrr = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.0001)
sgd = SGD(learning_rate = 0.0001, decay = 1e-6, momentum = 0.9, nesterov = True)
model_01.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

history_01 = model_01.fit(train_generator, steps_per_epoch = 10, epochs = 2, callbacks = [es,cp,lrr], validation_data = valid_generator)
if not os.path.isdir('D:/Dissertation/model_weights/'):
    os.mkdir('D:/Dissertation/model_weights/')
model_01.save_weights(filepath = "D:/Dissertation/model_weights/vgg19_model_01.weights.h5", overwrite = True)

base_model = VGG19(include_top = False, input_shape = (240,240,3))
for layer in base_model.layers:
    if layer.name in ['block5_conv4','block5_conv3']:
        layer.trainable = True
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation = 'relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation = 'relu')(drop_out)
output = Dense(2, activation = 'softmax')(class_2)
model_02 = Model(base_model.inputs, output)
model_02.load_weights("D:/Dissertation/model_weights/vgg19_model_01.weights.h5")
sgd = SGD(learning_rate = 0.0001, decay = 1e-6, momentum = 0.9, nesterov = True)
model_02.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
history_02 = model_02.fit(train_generator, steps_per_epoch = 10, epochs = 2, callbacks = [es,cp,lrr], validation_data = valid_generator)
model_02.save_weights(filepath = "D:/Dissertation/model_weights/vgg19_model_02.weights.h5", overwrite = True)

base_model = VGG19(include_top = False, input_shape = (240,240,3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation = 'relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation = 'relu')(drop_out)
output = Dense(2, activation = 'softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights('D:/Dissertation/model_weights/vgg19_model_02.weights.h5')
sgd = SGD(learning_rate = 0.0001, decay = 1e-6, momentum = 0.9, nesterov = True)
model_03.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
history_03 = model_03.fit(train_generator, steps_per_epoch = 10, epochs = 25, callbacks = [es,cp,lrr], validation_data = valid_generator)
model_03.save_weights(filepath="D:/Dissertation/model_weights/vgg19_model_03.weights.h5", overwrite = True)

model_03.load_weights("D:/Dissertation/model_weights/vgg19_model_03.weights.h5")
vgg_val_eval_03 = model_03.evaluate(valid_generator)
vgg_test_eval_03 = model_03.evaluate(test_generator)
model_03.save('BrainTumorModel4-Vgg19.keras')