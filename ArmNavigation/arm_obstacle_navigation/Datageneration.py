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
