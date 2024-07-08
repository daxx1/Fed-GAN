import numpy as np
import os
from keras.utils import np_utils
from sklearn.decomposition import PCA
import tensorflow as tf
import cv2
#lis = []
#lis=results[0].tolist()
'''
votes_inx = np.loadtxt(r'F:\test/votes_inx159f.txt', delimiter=',')
clean_votes = np.ones((5000, 100000))
for i in range(5000):
    for v in range(10000):
        clean_votes[i][int(votes_inx[i][99999 - v])] = 0
result=np.sum(clean_votes,axis=0)
'''
#np.savetxt(r"F:\test/result-10000.txt", np.asarray(result), delimiter=",",fmt="%d")
'''
result = np.loadtxt(r'./votes_inx_mnist_79.27_eps10_fen3_0-1000.txt', delimiter=',')
result1 = np.loadtxt(r'./votes_inx_mnist_79.27_eps10_fen3_1000-2000.txt', delimiter=',')
result2 = np.loadtxt(r'./votes_inx_mnist_79.27_eps10_fen3_2000-3000.txt', delimiter=',')
result3 = np.loadtxt(r'./votes_inx_mnist_79.27_eps10_fen3_3000-4000.txt', delimiter=',')
result4 = np.loadtxt(r'./votes_inx_mnist_79.27_eps10_fen3_4000-5000.txt', delimiter=',')
votes_inx = np.zeros((5000, 100000))
for i in range(1000):

    votes_inx[i] = result[i]
    votes_inx[i+1000] = result1[i]
    votes_inx[i + 2000] = result2[i]
    votes_inx[i + 3000] = result3[i]
    votes_inx[i + 4000] = result4[i]

clean_votes = np.ones((5000, 100000))

for i in range(5000):
    for v in range(20000):
        clean_votes[i][int(votes_inx[i][99999 - v])] = 0
result=np.sum(clean_votes,axis=0)
np.savetxt(r"./20000.txt", np.asarray(result), delimiter=",",fmt="%d")
'''

result=np.loadtxt(r"./10000.txt",  delimiter=",")
'''
clean_votes = np.ones((5000, 100000))

for i in range(5000):
    for v in range(10000):
        clean_votes[i][int(votes_inx[i][99999 - v])] = 0
result=np.sum(clean_votes,axis=0)
np.savetxt(r"./10000.txt", np.asarray(result), delimiter=",",fmt="%d")
'''
con=0
for i in range(100000):
    if result[i]>4750:
        con =con+1
print(con)
result = np.argsort(result)[::-1]
'''
#lis = np.argsort(lis)#范数专用
print('pppppppppppppppppp')
import joblib
data = joblib.load(r"F:\eps-3.0074m.data")


'''
dim = 0
data = np.zeros((100000, 794))
import joblib
'''
from tqdm import tqdm
for i in tqdm(range(10)):
    x =  joblib.load(r"eps-9.95.data" + f'-{i}.pkl')
    data[dim: dim+len(x)] = x
    dim += len(x)
'''
data = joblib.load('eps-3.00.data52')
x, label = np.hsplit(data, [-10])
nb_classes = 10
label = label.reshape((label.shape[0], nb_classes), order='F')

x = x.reshape(x.shape[0], 28, 28,1)

new_im = np.zeros((100000,784))

#new_im = new_im.reshape(new_im.shape[0], 28, 28, 1)
print(new_im.shape)
print(x[result[0]].flatten().shape)
new_im[0] = x[result[0]].flatten()

#a = np.concatenate([new_im,x[lis[1]].flatten()],axis=1)
for i in range(1,100000):
    if i%100==0:
        print(i)
    #new_im = np.vstack((new_im,x[result[i]].flatten()))
    new_im[i] = x[result[i]].flatten()
print('?o')
print(new_im.shape)
print(x[result[1]].flatten().shape)
label_list = np.zeros((100000,10))
label_list[0] = label[result[0]]
for i in range(1,100000):
    #label_list = np.vstack((label_list,label[result[i]]))
    label_list[i] = label[result[i]]
print('?o')
print(label_list)
new_im = new_im.reshape(new_im.shape[0], 28, 28, 1)
'''
import cv2
for i in range(10000):
    print(i)
    y=new_im[i]
    print(label_list[i])
    cv2.imshow('img',y)
    cv2.waitKey(0)
'''


import joblib
#joblib.dump(new_im, r"F:\eps-1.00f59-100000x.data")
#joblib.dump(label_list, r"F:\eps-1.00f59-100000y.data")
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse



import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.keras.backend.set_session(tf.Session(config=config));




parser = argparse.ArgumentParser(description='Train classifier and evaluate their accuracy')
parser.add_argument('--data', type=str, help='datafile name')

args = parser.parse_args()
'''
import joblib
data = joblib.load(args.data)
print(args.data)
x, label = np.hsplit(data, [-10])
nb_classes = 10
label = label.reshape((label.shape[0], nb_classes), order='F')
x = x.reshape(x.shape[0], 28, 28, 1)
'''
from keras.datasets import fashion_mnist
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255.

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.optimizers import Adam
from keras import optimizers

model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1), name='Conv2D-1'))
model.add(MaxPooling2D(pool_size=2, name='MaxPool'))
model.add(Dropout(0.2, name='Dropout-1'))
model.add(Conv2D(64, kernel_size=3, activation='relu', name='Conv2D-2'))
model.add(Dropout(0.25, name='Dropout-2'))
model.add(Flatten(name='flatten'))
model.add(Dense(64, activation='relu', name='Dense'))
model.add(Dense(nb_classes, activation='softmax', name='Output'))
sgd = optimizers.sgd(lr=2e-3)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint
filepath = 'weights.best50.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]

print(x.shape)
print(label.shape)
print(x_test.shape)
print(y_test.shape)
train_accs = []
eval_accs = []
#     for i in range(70):
new_im = new_im[:con]
label_list = label_list[:con]
history = model.fit(new_im, label_list, batch_size=512, epochs=500, validation_data=(x_test, y_test), shuffle=True,callbacks=callback_list)
train_accs = history.history['accuracy']
eval_accs = history.history['val_accuracy']

