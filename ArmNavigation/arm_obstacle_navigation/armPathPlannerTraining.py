
import keras
from keras.layers import keras.processing.image import ImageDataGenerator
import Input, Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np


#Preprocessing the training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'arm_obstacle_navigation/training_set'
        target_size=(100, 100), #need to decide whether we plan to use 100x100 data size for the images cuz its gonna take a long time to train then
        batch_size=32,
        class_mode='binary') #what type of class mode is it

#Preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'arm_obstacle_navigation/test_set',
        target_size=(100, 100), #also need to decide the images here
        batch_size=32,
        class_mode='binary') #need to determine the proper class mode

#Initializing the CNN
cnn = tf.keras.models.Sequential()

#Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape[100, 100, 3])) #need to update this to te size of the images

#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Second Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')) #need to update this to te size of the images
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Flattening
cnn.add(tf.keras.layers.Flatten())

#Full Connection
cnn.add(tf.keras.layer.Dense(units=128, activation='relu'))

#Output Layer
cnn.add(tf.keras.layer.Dense(units=1, activation='sigmoid')) #need to change units so that it is not binary classification

#Training the cnn
cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrcis = ['accuracy'])

#Training the cnn on the training set and evaluating it on the test set
cnn.fit(x = training_set, validation_data = test_set, epochs = )


input_path = '../data_2/maze_10x10_rnd/'
# input_path = '../data_2/maze_15x15_rnd/'
# input_path = '../data_2/maze_20x20_rnd/'
# input_path = '../data_2/maze_30x30_rnd/'

print('Load data ...')
x = np.loadtxt(input_path + 'inputs.dat')
s_map = np.loadtxt(input_path + 's_maps.dat')
g_map = np.loadtxt(input_path + 'g_maps.dat')
y = np.loadtxt(input_path + 'outputs.dat')

m = x.shape[0] #there are 30.000 samples in the dataset, shape[0] gives you the number of rows in the array
n = int(np.sqrt(x.shape[1])) #shape[1] gives you the length of the row

### Data split
n_train = 28000 #number of training samples; first n_train samples in the data set are used for training
n_test = m - n_train #the last n_test samples are used for testing

print('Transform data ...')
x = x.reshape(m,n,n)
s_map = s_map.reshape(m,n,n)
g_map = g_map.reshape(m,n,n)
y = y.reshape(m,n,n)

x3d = np.zeros((m,n,n,3))
x3d[:,:,:,0] = x
x3d[:,:,:,1] = s_map
x3d[:,:,:,2] = g_map
del x, s_map, g_map

x_train = x3d[0:n_train,:,:,:]
y_train = y[0:n_train,:,:]
x_test = x3d[n_train:m,:,:,:]
y_test = y[n_train:m,:,:]
del x3d, y


x = Input(shape=[100, 100, 3])))

net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(x)
net = BatchNormalization()(net)
for i in range(19):
	net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(net)
	net = BatchNormalization()(net)

net = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='sigmoid')(net)
net = BatchNormalization()(net)
net = Dropout(0.10)(net)

model = Model(inputs=x,outputs=net)
model.summary()

early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
save_weights = ModelCheckpoint(filepath='weights_2d.hf5', monitor='val_acc',verbose=1, save_best_only=True)

print('Train network ...')
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.fit(x_train.reshape(n_train,n,n,3), y_train.reshape(n_train,n,n,1), batch_size=64, validation_split=1/14, epochs=1000, verbose=1, callbacks=[early_stop, save_weights])

print('Save trained model ...')
model.load_weights('weights_2d.hf5')
model.save("model_2d.hf5")

print('Test network ...')
model=load_model("model_2d.hf5")
score = model.evaluate(x_test.reshape(n_test,n,n,3), y_test.reshape(n_test,n,n,1), verbose=1)
print('test_acc:', score[1])
