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
class ListNode:
    """
    A node in a singly-linked list.
    """
    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next

    def __repr__(self):
        return repr(self.data)


class SinglyLinkedList:
    def __init__(self):
        """
        Create a new singly-linked list.
        Takes O(1) time.
        """
        self.head = None
        self.size = 0;

    def __repr__(self):
        """
        Return a string representation of the list.
        Takes O(n) time.
        """
        nodes = []
        curr = self.head
        while curr:
            nodes.append(repr(curr))
            curr = curr.next
        return '[' + ', '.join(nodes) + ']'

    def prepend(self, data):
        """
        Insert a new element at the beginning of the list.
        Takes O(1) time.
        """
        self.head = ListNode(data=data, next=self.head)
        self.size++

    def append(self, data):
        """
        Insert a new element at the end of the list.
        Takes O(n) time.
        """
        if not self.head:
            self.head = ListNode(data=data)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = ListNode(data=data)
        self.size++

    def find(self, key):
        """
        Search for the first element with `data` matching
        `key`. Return the element or `None` if not found.
        Takes O(n) time.
        """
        curr = self.head
        while curr and curr.data != key:
            curr = curr.next
        return curr  # Will be None if not found

    def remove(self, key):
        """
        Remove the first occurrence of `key` in the list.
        Takes O(n) time.
        """
        # Find the element and keep a
        # reference to the element preceding it
        curr = self.head
        prev = None
        while curr and curr.data != key:
            prev = curr
            curr = curr.next
        # Unlink it from the list
        if prev is None:
            self.head = curr.next
        elif curr:
            prev.next = curr.next
            curr.next = None
        self.size--
    def dequeue(self):
        """
        Dequeues first element in list
        """
        # Find the element and keep a
        # reference to the element preceding it
        curr = self.head
        self.head = curr.next
        prev = None
        self.size--
        return curr.data

    def reverse(self):
        """
        Reverse the list in-place.
        Takes O(n) time.
        """
        curr = self.head
        prev_node = None
        next_node = None
        while curr:
            next_node = curr.next
            curr.next = prev_node
            prev_node = curr
            curr = next_node
        self.head = prev_node
np.set_printoptions(threshold=sys.maxsize)

#Preprocessing the training set

#i dont think we need all of the next 5 lines

training_set_x = tf.keras.preprocessing.image_dataset_from_directory(
    directory = '/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x',
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\\ArmNavigation\\arm_obstacle_navigation\\training_set_x',
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x',
    #'C:\\Users\\nihal\\Documents\\GitHub\\PythonRobotics\\ArmNavigation\\arm_obstacle_navigation\\training_set_x',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(100, 100))
training_set_y = tf.keras.preprocessing.image_dataset_from_directory(
    directory = '/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y',
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y',
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
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x',
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
    #'/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y',
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

print('Saving images in array format...')

#img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace00000.png')
#print("Orignal:" ,type(img))

x = np.zeros((3,))
# convert to numpy array
FinalArmConfigImageArray = SinglyLinkedList()
for x in range (9999):
    #img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/finalarmconfig/finalarmconfig{:05d}.png'.format(x))
    img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/finalarmconfig/finalarmconfig{:05d}.png'.format(x))
    FinalArmConfigImageArray.append(img_to_array(img))
    print(x)
    #np.savetxt('FinalArmConfigImageArray.dat', FinalArmConfigImageArray)
    #fullarray = (img_to_array(img))
    #print(fullarray)
    #reshapedarray = fullarray.reshape(100,100)
#    stringarray = np.array2string(reshapedarray)
    #onebracketgone = stringarray.replace('[', ' ')
    #shortenedarray = onebracketgone.replace(']', '')
    #print(shortenedarray)
    #f.write(np.array2string(img_to_array(img)))
    #FinalArmConfigImageArray.append(img_to_array(img))
#f=open("FinalArmConfigImageArray", "a+")
#np.savetxt('FinalArmConfigImageArray.dat', FinalArmConfigImageArray)
holderArray = np.zeros(size)
for c in range(FinalArmConfigImageArray.size):
    holderArray[c] = FinalArmConfigImageArray.dequeue()
print(holderArray)
WorkSpaceImageArray =  SinglyLinkedList()
for x in range (9999):
    #img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace{:05d}.png'.format(x))
    img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/workspacegrid/workspace{:05d}.png'.format(x))
    WorkSpaceImageArray.append(img_to_array(img))
    print(x)

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


StartArmConfigImageArray =  SinglyLinkedList()
for x in range (9999):
    img = load_img#('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/startarmconfig/startarmconfig{:05d}.png'.format(x))
    ('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/startarmconfig/startarmconfig{:05d}.png'.format(x))
#('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_x/startarmconfig/startarmconfig{:05d}.png'.format(x))
    StartArmConfigImageArray.append(img_to_array(img))

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

routeGridImageArray = SinglyLinkedList()
for x in range (10000):
    #img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y/routegrid/route{:05d}.png'.format(x))
    img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y/routegrid/route{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/training_set_y/routegrid/route{:05d}.png'.format(x))
    routeGridImageArray.append(img_to_array(img))

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

trainingDataXArray = []
trainingDataXArray.append(FinalArmConfigImageArray)
trainingDataXArray.append(WorkSpaceImageArray)
trainingDataXArray.append(StartArmConfigImageArray)

FinalArmConfigImageArrayTest = SinglyLinkedList()
for x in range (999):
    #img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetfinalarmconfig/finalarmconfig{:05d}.png'.format(x)))
    img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetfinalarmconfig/finalarmconfig{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetfinalarmconfig/finalarmconfig{:05d}.png'.format(x))
    FinalArmConfigImageArrayTest.append(img_to_array(img))
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


FinalWorkSpaceImageArrayTest =  SinglyLinkedList()
for x in range (999):
    img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetworkspace/workspace{:05d}.png'.format(x))
    #FinalWorkSpaceImageArrayTest.append(img_to_array(img)))
    #img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetworkspace/workspace{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetworkspace/workspace{:05d}.png'.format(x))
    FinalWorkSpaceImageArrayTest.append(img_to_array(img))
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


StartArmConfigImageArrayTest = SinglyLinkedList()
for x in range (999):
    img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetstartarmconfig/startarmconfig{:05d}.png'.format(x))
    #img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetstartarmconfig/startarmconfig{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_x/testsetstartarmconfig/startarmconfig{:05d}.png'.format(x))
    StartArmConfigImageArrayTest.append(img_to_array(img))
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

routeGridImageArrayTest = SinglyLinkedList()
for x in range (1000):
    #img = load_img('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y/testsetroute/route{:05d}.png'.format(x))
    #img = load_img('/Users/prana/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y/testsetroute/route{:05d}.png'.format(x))
    #('/Users/palluri/Documents/GitHub/PythonRobotics/ArmNavigation/arm_obstacle_navigation/test_set_y/testsetroute/route{:05d}.png'.format(x))
    routeGridImageArrayTest.append(img_to_array(img))
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


FinalArmConfigImageArray = []
testDataXArray.append(FinalArmConfigImageArrayTest)
testDataXArray.append(WorkSpaceImageArrayTest)
testDataXArray.append(StartArmConfigImageArrayTest)
