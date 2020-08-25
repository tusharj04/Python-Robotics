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

#Preprocessing the training set

#i dont think we need all of the next 5 lines
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set_x = tf.keras.preprocessing.image_dataset_from_directory(
    directory = '/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x',
    #'/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x',
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\\ArmNavigation\\arm_obstacle_navigation\\training_set_x',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(100, 100))
training_set_y = tf.keras.preprocessing.image_dataset_from_directory(
    directory ='/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y',
    #'/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y',
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\ArmNavigation\\arm_obstacle_navigation\\training_set_y',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(100, 100))#what type of class mode is it

#Preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set_x = tf.keras.preprocessing.image_dataset_from_directory(
    directory ='/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x',
    #'/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\ArmNavigation\\arm_obstacle_navigation\\test_set_x',
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x',
    # 'arm_obstacle_navigation/test_set_x',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(100, 100))
test_set_y = tf.keras.preprocessing.image_dataset_from_directory(
    directory ='/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y',
    #'/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\ArmNavigation\\arm_obstacle_navigation\\test_set_y',
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y',
    #'arm_obstacle_navigation/test_set_y',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(100, 100))
#i am not sure if the images are already 100 by 100, but if they are, i think we do not even need most of the above stuff,



#img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace00000.png')
#'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace00000.png')
#print("Orignal:",type(img))

print('Saving images in array format...')

# convert to numpy array
FinalArmConfigImageArray = []
for x in range (9999):
    #img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/finalarmconfig/finalarmconfig{:05d}.png'.format(x))
    img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/finalarmconfig/finalarmconfig{:05d}.png'.format(x))
    fullarray = (img_to_array(img))
    #print(fullarray)
    reshapedarray = fullarray.reshape(960,960)
    stringarray = np.array2string(reshapedarray)
    onebracketgone = stringarray.replace('[', ' ')
    shortenedarray = onebracketgone.replace(']', '')
    print(shortenedarray)
    f= open("FinalArmConfigImageArray.dat \n", "w+")
    f.write(shortenedarray)
    f.close()

    #f.write(np.array2string(img_to_array(img)))
    #FinalArmConfigImageArray.append(img_to_array(img))
#f=open("FinalArmConfigImageArray", "a+")
#np.savetxt('FinalArmConfigImageArray.dat', FinalArmConfigImageArray)

WorkSpaceImageArray = []
for x in range (9999):
    img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace{:05d}.png'.format(x))
    fullarray = (img_to_array(img))
    #print(fullarray)
    reshapedarray = fullarray.reshape(960,960)
    stringarray = np.array2string(reshapedarray)
    onebracketgone = stringarray.replace('[', ' ')
    shortenedarray = onebracketgone.replace(']', '')
    print(shortenedarray)
    f= open("WorkSpaceImageArray.dat \n", "w+")
    f.write(shortenedarray)
    f.close()


StartArmConfigImageArray = []
for x in range (9999):

    img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/startarmconfig/startarmconfig{:05d}.png'.format(x))
    #('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/startarmconfig/startarmconfig{:05d}.png'.format(x))
#('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/startarmconfig/startarmconfig{:05d}.png'.format(x))
    fullarray = (img_to_array(img))
    #print(fullarray)
    reshapedarray = fullarray.reshape(960,960)
    stringarray = np.array2string(reshapedarray)
    onebracketgone = stringarray.replace('[', ' ')
    shortenedarray = onebracketgone.replace(']', '')
    print(shortenedarray)
    f= open("StartArmConfigImageArray.dat \n", "w+")
    f.write(shortenedarray)
    f.close()

routeGridImageArray = []
for x in range (10000):

    img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y/routegrid/route{:05d}.png'.format(x))
    #('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y/routegrid/route{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y/routegrid/route{:05d}.png'.format(x))
    fullarray = (img_to_array(img))
    #print(fullarray)
    reshapedarray = fullarray.reshape(960,960)
    stringarray = np.array2string(reshapedarray)
    onebracketgone = stringarray.replace('[', ' ')
    shortenedarray = onebracketgone.replace(']', '')
    print(shortenedarray)
    f= open("routeGridImageArray.dat \n", "w+")
    f.write(shortenedarray)
    f.close()
#trainingDataXArray = []
#trainingDataXArray.append(FinalArmConfigImageArray)
#trainingDataXArray.append(WorkSpaceImageArray)
#trainingDataXArray.append(StartArmConfigImageArray)
#save('trainingDataXArray.dat', trainingDataXArray)

#FinalArmConfigImageArrayTest = []
for x in range (999):
    img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetfinalarmconfig/finalarmconfig{:05d}.png'.format(x)))
    #('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetfinalarmconfig/finalarmconfig{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetfinalarmconfig/finalarmconfig{:05d}.png'.format(x))
    #FinalArmConfigImageArrayTest.append(img_to_array(img))
    fullarray = (img_to_array(img))
    #print(fullarray)
    reshapedarray = fullarray.reshape(960,960)
    stringarray = np.array2string(reshapedarray)
    onebracketgone = stringarray.replace('[', ' ')
    shortenedarray = onebracketgone.replace(']', '')
    print(shortenedarray)
    f= open("FinalArmConfigImageArrayTest.dat \n", "w+")
    f.write(shortenedarray)
    f.close()


#FinalWorkSpaceImageArrayTest = []
for x in range (999):
    img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetworkspace/workspace{:05d}.png'.format(x))
    #FinalWorkSpaceImageArrayTest.append(img_to_array(img)))
    #('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetworkspace/workspace{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetworkspace/workspace{:05d}.png'.format(x))
    #FinalWorkSpaceImageArrayTest.append(img_to_array(img))
    fullarray = (img_to_array(img))
    #print(fullarray)
    reshapedarray = fullarray.reshape(960,960)
    stringarray = np.array2string(reshapedarray)
    onebracketgone = stringarray.replace('[', ' ')
    shortenedarray = onebracketgone.replace(']', '')
    print(shortenedarray)
    f= open("FinalWorkSpaceImageArrayTest.dat \n", "w+")
    f.write(shortenedarray)
    f.close()


StartArmConfigImageArrayTest = []
for x in range (1000):
    img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetstartarmconfig/startarmconfig{:05d}.png'.format(x))
    #('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetstartarmconfig/startarmconfig{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetstartarmconfig/startarmconfig{:05d}.png'.format(x))
    #StartArmConfigImageArrayTest.append(img_to_array(img))
    fullarray = (img_to_array(img))
    #print(fullarray)
    reshapedarray = fullarray.reshape(960,960)
    stringarray = np.array2string(reshapedarray)
    onebracketgone = stringarray.replace('[', ' ')
    shortenedarray = onebracketgone.replace(']', '')
    print(shortenedarray)
    f= open("StartArmConfigImageArrayTest.dat \n", "w+")
    f.write(shortenedarray)
    f.close()

#routeGridImageArrayTest = []
for x in range (1000):
img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y/testsetroute/route{:05d}.png'.format(x))
    #('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y/testsetroute/route{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y/testsetroute/route{:05d}.png'.format(x))
    #routeGridImageArrayTest.append(img_to_array(img))
    fullarray = (img_to_array(img))
    #print(fullarray)
    reshapedarray = fullarray.reshape(960,960)
    stringarray = np.array2string(reshapedarray)
    onebracketgone = stringarray.replace('[', ' ')
    shortenedarray = onebracketgone.replace(']', '')
    print(shortenedarray)
    f= open("routeGridImageArrayTest.dat \n", "w+")
    f.write(shortenedarray)
    f.close()

#testDataXArray = []
#testDataXArray.append(FinalArmConfigImageArrayTest)
#testDataXArray.append(WorkSpaceImageArrayTest)
#testDataXArray.append(StartArmConfigImageArrayTest)

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
