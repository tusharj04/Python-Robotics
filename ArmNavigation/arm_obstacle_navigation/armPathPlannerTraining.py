import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
import sys


import numpy as np
np.set_printoptions(threshold=sys.maxsize)

startarmdata = np.loadtxt('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/StartArmConfigImageArray.dat')
finalarmdata = np.loadtxt('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/FinalArmConfigImageArray.dat')
workspacedata = np.loadtxt('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/WorkSpaceImageArray.dat')
routedata = np.loadtxt('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/routeGridImageArray.dat')
#Initializing the CNN
#x = Input(shape=(None, None, 3))

#net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(x)
#net = BatchNormalization()(net)
#for i in range(19):
	#net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(net)
	#net = BatchNormalization()(net)

#net = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='sigmoid')(net)
#net = BatchNormalization()(net)
#net = Dropout(0.10)(net)

#21 convolutional layers created, with  batch BatchNormalization


#model = Model(inputs=x,outputs=net)
#creating a model based off the inputted shape and the outputted layers
#model.summary()

#early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
#save_weights = ModelCheckpoint(filepath='weights_2d.hf5', monitor='val_acc',verbose=1, save_best_only=True)

#print('Train network ...')
#model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

#model.fit(training_set_x.reshape(10000, 100, 100, 3), training_set_y.reshape(10000,100,100,1), batch_size=64, validation_split=1/14, epochs=1000, verbose=1, callbacks=[early_stop, save_weights])
#model.fit_generator(generator, epochs=int, steps_per_epoch=int, validation_data=tuple, validation_steps=int)
#model.fit(trainingDataXArray.reshape(10000,100,100,3),routeGridImageArray.reshape(10000, 100, 100, 1), validation_data=(testDataXArray, routeGridImageArrayTest), epochs=1000, verbose=1, callbacks=[early_stop, save_weights])
#model.fit(x_train.reshape(n_train,n,n,3), y_train.reshape(n_train,n,n,1), batch_size=64, validation_split=1/14, epochs=1000, verbose=1, callbacks=[early_stop, save_weights])


#we are fitting to trainng set, 10000 sets a workspace + start grid + final grid (3 total) which are all 100 by 100 each
#print('Save trained model ...')
#model.load_weights('weights_2d.hf5')
#model.save("model_2d.hf5")

#print('Test network ...')
#model=load_model("model_2d.hf5")
#score = model.evaluate(testDataXArray.reshape(10000,100,100,3), routeGridImageArrayTest.reshape(10000,100,100,1), verbose=1)
#print('test_acc:', score[1])
