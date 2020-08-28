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
from PIL import Image

import numpy as np
np.set_printoptions(threshold=sys.maxsize)



'''
training_set_x = tf.keras.preprocessing.image_dataset_from_directory(
    directory = '/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x',
    #'/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\\ArmNavigation\\arm_obstacle_navigation\\training_set_x',
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\\ArmNavigation\\arm_obstacle_navigation\\training_set_x',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(100, 100))
training_set_y = tf.keras.preprocessing.image_dataset_from_directory(
    directory = '/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y',
    #'/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\ArmNavigation\\arm_obstacle_navigation\\training_set_y',
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\ArmNavigation\\arm_obstacle_navigation\\training_set_y',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(100, 100))
test_set_x = tf.keras.preprocessing.image_dataset_from_directory(
    directory = '/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\ArmNavigation\\arm_obstacle_navigation\\test_set_x',
    #'/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\ArmNavigation\\arm_obstacle_navigation\\test_set_x',
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x',
    # 'arm_obstacle_navigation/test_set_x',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(100, 100))
test_set_y = tf.keras.preprocessing.image_dataset_from_directory(
    directory = '/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\ArmNavigation\\arm_obstacle_navigation\\test_set_y',
    #'/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\ArmNavigation\\arm_obstacle_navigation\\test_set_y',
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y',
    #'arm_obstacle_navigation/test_set_y',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(100, 100))

#img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace00000.png')
#'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace00000.png')
#print("Orignal:",type(img))


#img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace00000.png')
#print("Orignal:" ,type(img))
#testimg = load_img('/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace{:05d}.png'.format(1))
#print(img_to_array(testimg))

'''

print('Saving images in array format...')

# convert to numpy array
FinalArmConfigImageArray = np.array([])
FinalArmConfigImageArray = np.array(FinalArmConfigImageArray, dtype=np.int8)
for x in range (9999):
    #img = load_img('/Users/palluri/GitHub/PythonRobot1ics/ArmNavigation/arm_obstacle_navigation/training_set_x/finalarmconfig/finalarmconfig{:05d}.png'.format(x))
    img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/finalarmconfig/finalarmconfig{:05d}.png'.format(x))
    appendplain = np.array(img_to_array(img))
    appendreshaped = appendplain.reshape(960,960)
    FinalArmConfigImageArray = np.append(FinalArmConfigImageArray,appendreshaped)
    print(x)
    #print(x)
    #img2 = array_to_img(appendplain)
    #img2.show()
    #FinalArmConfigImageArray = FinalArmConfigImageArray.astype(np.int32)
    #np.savetxt('FinalArmConfigImageArray.dat', FinalArmConfigImageArray)
    #fullarray = (img_to_array(img))
    #print(fullarray)
    #reshapedarray = fullarray.reshape(100,100)
#
    #onebracketgone = stringarray.replace('[', ' ')
    #shortenedarray = onebracketgone.replace(']', '')
    #print(shortenedarray)
    #reshapedarray1 = np.array2string(appendreshaped)
    #f=open("FinalArmConfigImageArray.dat", "a+")
    #f.write(reshapedarray1)
    #f.close()
    #FinalArmConfigImageArray.append(img_to_array(img))

#np.savetxt('FinalArmConfigImageArray.dat', FinalArmConfigImageArray)


WorkSpaceImageArray = np.array([])
WorkSpaceImageArray = np.array(WorkSpaceImageArray, dtype=np.int8)
for x in range (9999):
    #img = load_img('/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace{:05d}.png'.format(x))
    img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace{:05d}.png'.format(x))
    #WorkSpaceImageArray.append(img_to_array(img))
    #print(x)
    appendplain = np.array(img_to_array(img))
    appendreshaped = appendplain.reshape(960,960)
    WorkSpaceImageArray = np.append(WorkSpaceImageArray,appendreshaped)
    #fullarray = (img_to_array(img))
    #print(fullarray)
    #reshapedarray = fullarray.reshape(100,100)
    #stringarray = np.array2string(reshapedarray)
    #onebracketgone = stringarray.replace('[', ' ')
    #shortenedarray = onebracketgone.replace(']', '')
    #print(shortenedarray)
    #f= open("WorkSpaceImageArray.dat", "w+")
    #f.write(shortenedarray)
    #f.close()

StartArmConfigImageArray = np.array([])
StartArmConfigImageArray = np.array(StartArmConfigImageArray, dtype=np.int8)
for x in range (9999):
    #img = load_img('/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/startarmconfig/startarmconfig{:05d}.png'.format(x))
    ('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/startarmconfig/startarmconfig{:05d}.png'.format(x))
#('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/startarmconfig/startarmconfig{:05d}.png'.format(x))
    #StartArmConfigImageArray.append(img_to_array(img))
    appendplain = np.array(img_to_array(img))
    appendreshaped = appendplain.reshape(960,960)
    StartArmConfigImageArray = np.append(StartArmConfigImageArray,appendreshaped)
    #fullarray = (img_to_array(img))
    #print(fullarray)
    #reshapedarray = fullarray.reshape(100,100)
#    stringarray = np.array2string(reshapedarray)
    #onebracketgone = stringarray.replace('[', ' ')
    #shortenedarray = onebracketgone.replace(']', '')
    #print(shortenedarray)
    #f= open("StartArmConfigImageArray.dat", "w+")
    #f.write(shortenedarray)
    #f.close()

routeGridImageArray = np.array([])
routeGridImageArray = np.array(routeGridImageArray, dtype=np.int8)
for x in range (10000):
    #img = load_img('/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y/routegrid/route{:05d}.png'.format(x))
    img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y/routegrid/route{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y/routegrid/route{:05d}.png'.format(x))
    #routeGridImageArray.append(img_to_array(img))
    appendplain = np.array(img_to_array(img))
    appendreshaped = appendplain.reshape(960,960)
    routeGridImageArray = np.append(routeGridImageArray,appendreshaped)
    #fullarray = (img_to_array(img))
    #print(fullarray)
    #reshapedarray = fullarray.reshape(100,100)
    #stringarray = np.array2string(reshapedarray)
    #onebracketgone = stringarray.replace('[', ' ')
    #shortenedarray = onebracketgone.replace(']', '')
    #print(shortenedarray)
    #f= open("routeGridImageArray.dat", "w+")
    #f.write(shortenedarray)
    #f.close()
#trainingDataXArray = []
#trainingDataXArray.append(FinalArmConfigImageArray)
#trainingDataXArray.append(WorkSpaceImageArray)
#trainingDataXArray.append(StartArmConfigImageArray)
#save('trainingDataXArray.dat', trainingDataXArray)

trainingDataXArray = np.array([])
trainingDataXArray = np.append(trainingDataXArray, FinalArmConfigImageArray)
trainingDataXArray = np.append(trainingDataXArray, WorkSpaceImageArray)
trainingDataXArray = np.append(trainingDataXArray, StartArmConfigImageArray)

FinalArmConfigImageArrayTest = np.array([])
FinalArmConfigImageArrayTest = np.array(FinalArmConfigImageArrayTest, dtype=np.int8)
for x in range (999):
    #img = load_img('/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetfinalarmconfig/finalarmconfig{:05d}.png'.format(x))
    img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetfinalarmconfig/finalarmconfig{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetfinalarmconfig/finalarmconfig{:05d}.png'.format(x))
    #FinalArmConfigImageArrayTest.append(img_to_array(img))
    appendplain = np.array(img_to_array(img))
    appendreshaped = appendplain.reshape(960,960)
    FinalArmConfigImageArrayTest = np.append(FinalArmConfigImageArrayTest,appendreshaped)
    #FinalArmConfigImageArrayTest.append(img_to_array(img))
    #fullarray = (img_to_array(img))
    #print(fullarray)
    #reshapedarray = fullarray.reshape(100,100)
    #stringarray = np.array2string(reshapedarray)
    #onebracketgone = stringarray.replace('[', ' ')
    #shortenedarray = onebracketgone.replace(']', '')
    #print(shortenedarray)
    #f= open("FinalArmConfigImageArrayTest.dat", "w+")
    #f.write(shortenedarray)
    #f.close()

FinalWorkSpaceImageArrayTest = np.array([])
FinalWorkSpaceImageArrayTest = np.array(FinalWorkSpaceImageArrayTest, dtype=np.int8)
for x in range (999):
    #img = load_img('/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetworkspace/workspace{:05d}.png'.format(x))
    #FinalWorkSpaceImageArrayTest.append(img_to_array(img)))
    img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetworkspace/workspace{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetworkspace/workspace{:05d}.png'.format(x))
    #FinalWorkSpaceImageArrayTest.append(img_to_array(img)
    appendplain = np.array(img_to_array(img))
    appendreshaped = appendplain.reshape(960,960)
    FinalWorkSpaceImageArrayTest = np.append(FinalWorkSpaceImageArrayTest,appendreshaped)
    #FinalWorkSpaceImageArrayTest.append(img_to_array(img))
    #fullarray = (img_to_array(img))
    #print(fullarray)
    #reshapedarray = fullarray.reshape(100,100)
    #stringarray = np.array2string(reshapedarray)
    #onebracketgone = stringarray.replace('[', ' ')
    #shortenedarray = onebracketgone.replace(']', '')
    #print(shortenedarray)
    #f= open("FinalWorkSpaceImageArrayTest.dat", "w+")
    #f.write(shortenedarray)
    #f.close()


StartArmConfigImageArrayTest = np.array([])
StartArmConfigImageArrayTest = np.array(StartArmConfigImageArrayTest, dtype=np.int8)
for x in range (999):
    #img = load_img('/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetstartarmconfig/startarmconfig{:05d}.png'.format(x))
    img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetstartarmconfig/startarmconfig{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetstartarmconfig/startarmconfig{:05d}.png'.format(x))
    #StartArmConfigImageArrayTest.append(img_to_array(img))
    appendplain = np.array(img_to_array(img))
    appendreshaped = appendplain.reshape(960,960)
    StartArmConfigImageArrayTest = np.append(StartArmConfigImageArrayTest,appendreshaped)
    #StartArmConfigImageArrayTest.append(img_to_array(img))
    #fullarray = (img_to_array(img))
    #print(fullarray)
    #reshapedarray = fullarray.reshape(100,100)
    #stringarray = np.array2string(reshapedarray)
    #onebracketgone = stringarray.replace('[', ' ')
#    shortenedarray = onebracketgone.replace(']', '')
#    print(shortenedarray)
#    f= open("StartArmConfigImageArrayTest.dat", "w+")
    #f.write(shortenedarray)
    #f.close()

routeGridImageArrayTest = np.array([])
routeGridImageArrayTest = np.array(routeGridImageArrayTest, dtype=np.int8)
for x in range (1000):
    #img = load_img('/Users/palluri/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y/testsetroute/route{:05d}.png'.format(x))
    img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y/testsetroute/route{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y/testsetroute/route{:05d}.png'.format(x))
    #routeGridImageArrayTest.append(img_to_array(img))
    appendplain = np.array(img_to_array(img))
    appendreshaped = appendplain.reshape(960,960)
    routeGridImageArrayTest = np.append(routeGridImageArrayTest,appendreshaped)
    #routeGridImageArrayTest.append(img_to_array(img))
    #fullarray = (img_to_array(img))
    #print(fullarray)
    #reshapedarray = fullarray.reshape(960,960)
    #stringarray = np.array2string(reshapedarray)
    #onebracketgone = stringarray.replace('[', ' ')
    #shortenedarray = onebracketgone.replace(']', '')
    #print(shortenedarray)
    #f= open("routeGridImageArrayTest.dat \n", "w+")
    #f.write(shortenedarray)
    #f.close()

testDataXArray = np.array([])
testDataXArray = np.append(FinalArmConfigImageArrayTest, FinalArmConfigImageArray)
testDataXArray = np.append(FinalWorkSpaceImageArrayTest, WorkSpaceImageArray)
testDataXArray = np.append(StartArmConfigImageArrayTest, StartArmConfigImageArray)



#startarmdata = np.loadtxt('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/StartArmConfigImageArray.dat')
#finalarmdata = np.loadtxt('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/FinalArmConfigImageArray.dat')
#workspacedata = np.loadtxt('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/WorkSpaceImageArray.dat')
#routedata = np.loadtxt('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/routeGridImageArray.dat')
'''
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
#print('test_acc:', score[1]) '''
