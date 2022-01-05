# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 20:52:20 2021

@author: nehac
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train_std = (X_train/255. - .5)*2.
X_test_std = (X_test/255. - .5)*2.
X_train_std = X_train_std.reshape(-1,28,28,1)
X_test_std = X_test_std.reshape(-1,28,28,1)

onehot_encoder = OneHotEncoder(sparse=False)
Y_train_NN = onehot_encoder.fit_transform(Y_train.reshape(-1,1))
Y_test_NN = onehot_encoder.transform(Y_test.reshape(-1,1))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), 
           strides=(1,1), padding='same', name='conv_1', activation='relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), name='pool_1'))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), 
           strides=(1,1), padding='same', name='conv_2', activation='relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), name='pool_2'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=1024, name='fc_1', activation='relu'))

model.add(tf.keras.layers.Dropout(rate=0.5))

model.add(tf.keras.layers.Dense(units=10, name='fc_2', activation='softmax'))

tf.random.set_seed(1)
model.build(input_shape=(None,28,28,1))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

hist = model.fit(X_train_std,Y_train,validation_data=(X_test_std,Y_test),epochs=20,batch_size=100,verbose=1)
history = hist.history

plt.figure()
plt.plot(history['loss'],label='loss')
plt.plot(history['val_loss'],label='val_loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred = np.argmax(model.predict(X_test_std),axis=1)
check = X_test_std[y_pred != Y_test]
corr_label = Y_test[y_pred != Y_test]
pred_label = y_pred[y_pred != Y_test]
print('missclassified in test set: %d' %check.shape[0])

y_pred_train = np.argmax(model.predict(X_train_std),axis=1)
check_train = X_train_std[y_pred_train != Y_train]
print('missclassified in train set: %d' %check_train.shape[0])


fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True,figsize=(7,3))
ax = ax.flatten()
for i in range(25):
    image = check[i].reshape(28,28)
    ax[i].imshow(image,cmap='Greys')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title('%d) t: %d p: %d' %(i+1, corr_label[i], pred_label[i]))
plt.show()
