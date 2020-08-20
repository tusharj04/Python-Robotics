
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np


#Preprocessing the training set

#i dont think we need all of the next 5 lines
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set_x = train_datagen.flow_from_directory(
        '/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x',
#        'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\\ArmNavigation\\arm_obstacle_navigation\\training_set_x',
        target_size=(100, 100), #need to decide whether we plan to use 100x100 data size for the images cuz its gonna take a long time to train then
        class_mode='binary') #what type of class mode is it
training_set_y = train_datagen.flow_from_directory(
        '/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y',
#        'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\ArmNavigation\\arm_obstacle_navigation\\training_set_y',
        class_mode='binary') #what type of class mode is it

#Preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set_x = test_datagen.flow_from_directory(
        '/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x',
#        'arm_obstacle_navigation/test_set_x',
        target_size=(100, 100), #also need to decide the images here
        class_mode='binary') #need to determine the proper class mode
test_set_y = test_datagen.flow_from_directory(
        '/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y',
#        'arm_obstacle_navigation/test_set_y',
        target_size=(100, 100), #also need to decide the images here
        class_mode='binary') #need to determine the proper class mode
#i am not sure if the images are already 100 by 100, but if they are, i think we do not even need most of the above stuff,

#Initializing the CNN
x = Input(shape=[100, 100, 3])

net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(x)
net = BatchNormalization()(net)
for i in range(19):
	net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(net)
	net = BatchNormalization()(net)

net = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='sigmoid')(net)
net = BatchNormalization()(net)
net = Dropout(0.10)(net)

#21 convolutional layers created, with  batch BatchNormalization


model = Model(inputs=x,outputs=net)
#creating a model based off the inputted shape and the outputted layers
model.summary()

early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
save_weights = ModelCheckpoint(filepath='weights_2d.hf5', monitor='val_acc',verbose=1, save_best_only=True)

print('Train network ...')
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.fit(training_set_x.reshape(10000, 100, 100, 3), training_set_y.reshape(10000,100,100,1), batch_size=64, validation_split=1/14, epochs=1000, verbose=1, callbacks=[early_stop, save_weights])
#we are fitting to trainng set, 10000 sets a workspace + start grid + final grid (3 total) which are all 100 by 100 each
print('Save trained model ...')
model.load_weights('weights_2d.hf5')
model.save("model_2d.hf5")

print('Test network ...')
model=load_model("model_2d.hf5")
score = model.evaluate(x_test.reshape(10000,100,100,3), y_test.reshape(10000,100,100,1), verbose=1)
print('test_acc:', score[1])
