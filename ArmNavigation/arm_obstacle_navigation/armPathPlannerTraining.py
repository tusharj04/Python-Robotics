
import keras
from keras.layers import keras.processing.image import ImageDataGenerator
import Input, Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow
import numpy as np


#Preprocessing the training set
trainingXData = tf.keras.preprocessing.image_dataset_from_directory(
    directory='training_set_x/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
trainingYData = tf.keras.preprocessing.image_dataset_from_directory(
    directory='training_set_y/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))


########################### PAVAN PRANAY ADD DATA SHUFFLING AND BATCH normalization#############################################################
#add data shuffling and Bath normalization

x = Input()

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
model.fit(trainingXData.reshape(n_train,n,n,3),trainingYdata.reshape(n_train,n,n,1), batch_size=64, validation_split=1/14, epochs=1000, verbose=1, callbacks=[early_stop, save_weights])

print('Save trained model ...')
model.load_weights('weights_2d.hf5')
model.save("model_2d.hf5")

print('Test network ...')
model=load_model("model_2d.hf5")
score = model.evaluate(x_test.reshape(n_test,n,n,3), y_test.reshape(n_test,n,n,1), verbose=1)
print('test_acc:', score[1])
