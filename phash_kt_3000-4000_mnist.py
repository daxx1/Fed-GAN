import numpy as np
import os
from keras.utils import np_utils
from sklearn.decomposition import PCA
import tensorflow as tf
import cv2
num_teacher = 5000
section = 1000




def pHash(img,leng=28,wid=28):

    #img = cv2.resize(img, (leng, wid))

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(img))
    dct_roi = dct[0:8, 0:8]
    avreage = np.mean(dct_roi)

    phash_01 = (dct_roi>avreage)+0

    phash_list = phash_01.reshape(1,-1)[0].tolist()

    hash = ''.join([str(x) for x in phash_list])

    return hash
def Hamming_distance(hash1,hash2):

    num = 0

    for index in range(len(hash1)):

        if hash1[index] != hash2[index]:

            num += 1

    return num


def pca(teacher):
  pca = PCA(n_components=10)
  pca.fit(teacher)
  max_component = pca.components_.T
  teacher = np.dot(teacher, max_component)
  #student = np.dot(student, max_component)
  return teacher
def evenly_partition_dataset(data, labels, nb_teachers):
    """
    Simple partitioning algorithm that returns the right portion of the data
    needed by a given teacher out of a certain nb of teachers
    :param data: input data to be partitioned
    :param labels: output data to be partitioned
    :param nb_teachers: number of teachers in the ensemble (affects size of each
                       partition)
    :param teacher_id: id of partition to retrieve
    :return:
    """

    # This will floor the possible number of batches
    batch_len = int(len(data) / nb_teachers)

    nclasses = len(labels[0])
    print("Start Index Selection")
    data_sel = [data[labels[:, j] == 1] for j in range(nclasses)]
    print("End Index Selection")
    i = 0
    data_sel_id = [0] * len(labels[0])
    partition_data = []
    partition_labels = []

    while True:
        partition_data.append(data_sel[i][data_sel_id[i]])
        partition_labels.append(np_utils.to_categorical(i, nclasses))

        if len(partition_data) == batch_len:
            partition_data = np.asarray(partition_data)
            partition_labels = np.asarray(partition_labels)
            yield partition_data, partition_labels
            partition_data = []
            partition_labels = []

        data_sel_id[i] += 1
        if data_sel_id[i] == len(data_sel[i]):
            data_sel_id[i] = 0
        i = (i + 1) % nclasses

def load_mnist():
    #data_dir = os.path.join(self.data_dir, self.dataset_name)
    data_dir = r'C:\G-PATE\data\mnist'
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
    print(trX)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)
    # fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    # loaded = np.fromfile(file=fd,dtype=np.uint8)
    # teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    # fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    # loaded = np.fromfile(file=fd,dtype=np.uint8)
    # teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    # teY = np.asarray(teY)

    # X = np.concatenate((trX, teX), axis=0)
    # y = np.concatenate((trY, teY), axis=0).astype(np.int)
    X = trX
    y = trY.astype(np.int)

    seed = 307
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec
def load_fashion_mnist():
    #data_dir = os.path.join(self.data_dir, self.dataset_name)
    data_dir = r'C:\G-PATE\data\fashion_mnist'
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    # fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    # loaded = np.fromfile(file=fd,dtype=np.uint8)
    # teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    # fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    # loaded = np.fromfile(file=fd,dtype=np.uint8)
    # teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    # teY = np.asarray(teY)

    # X = np.concatenate((trX, teX), axis=0)
    # y = np.concatenate((trY, teY), axis=0).astype(np.int)
    X = trX
    y = trY.astype(np.int)

    seed = 307
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec
#data_X, data_y = load_mnist()
from keras.datasets import mnist
def job1(data_list):
	"""
	:param z:
	:return:
	"""
	return hamm(data_list[0], data_list[1],data_list[2],data_list[3],data_list[4])
data_X, data_y = load_mnist()
#data_X, data_y = load_fashion_mnist()
import numpy as np
conp = 0
#clean_votes = np.ones((1666, 100000))
votes_inx = np.zeros((section, 100000))



train_data_list = []
train_label_list = []
from sklearn.utils import shuffle

data_X, data_y = shuffle(data_X, data_y)

gen = evenly_partition_dataset(data_X, data_y, num_teacher)
from tqdm import tqdm
for i in tqdm(range(num_teacher)):
    partition_data, partition_labels = next(gen)
    train_data_list.append(partition_data)
    train_label_list.append(partition_labels)

train_size = len(train_data_list[0])

import joblib
#data = joblib.load(r"F:\eps-2.98.data142")

data = np.zeros((100000, 794))
dim = 0
import joblib

from tqdm import tqdm
for i in tqdm(range(10)):
    x =  joblib.load(r"F:\eps-9.94.data" + f'-{i}.pkl')
    data[dim: dim+len(x)] = x
    dim += len(x)

x, label = np.hsplit(data, [-10])
nb_classes = 10
label = label.reshape((label.shape[0], nb_classes), order='F')

x = x.reshape(x.shape[0], 28, 28,1)
print('i')

print(x.shape)
#lis = []
#print(train_data_list[0])#.flatten().reshape(12,784))


import time
start2 = time.time()
li=[]
import torch
label3 = [np.argmax(i) for i in label]
print('llll')
#print(label3)
'''
for i in range(1000):
    p_dist = Hamming_distance(pHash(train_data_list[0][label3[i]]),pHash(image2))
    li.append(1 - p_dist * 1.0 / 64)
    #np.mean(li)
'''


end2 = time.time()
#print(np.mean(li))
print(train_label_list[0][0])
#print('p_dist is '+'%d' % p_dist + ', similarity is ' +'%f' % (1 - p_dist * 1.0 / 64) + ', time is ' +'%f' % (end2-start2))
import cv2
#list_inx = []
list_inx =[[[]for z in range(10)] for p in range(section)]
for n in range(section*3,section*4):

    #if label[i].tolist() == train_label_list[n][l].tolist():
    inx = [np.argmax(i) for i in train_label_list[n]]
    for j in range(len(inx)):
        list_inx[n-section*3][inx[j]].append(j)
for n in range(section*3,section*4):
    lis = []

    print(n)
    for i in range(100000):
        query_data = x[i]
        p = []
        #print(list_inx[n][label3[i]])
        #print(label3[i])
        #print(list_inx[n])
        for z in range(len(list_inx[n-section*3][label3[i]])):
            nimi_inx = list_inx[n-section*3][label3[i]][z]
            p_dist = Hamming_distance(pHash(train_data_list[n][nimi_inx]), pHash(query_data))
            p.append(1 - p_dist * 1.0 / 64)
        lis.append(np.mean(p))
    lis = np.argsort(lis)[::-1]
    votes_inx[n-section*3] = lis
np.savetxt(r"F:\test/votes_inx_mnist_79.05_eps10_fen2_3000-4000.txt", np.asarray(votes_inx), delimiter=",",fmt="%d")
import sys
sys.exit()


for i in range(num_teacher):
    for v in range(10000):
        clean_votes[i][int(votes_inx[i][99999 - v])] = 0
result=np.sum(clean_votes,axis=0)
np.savetxt(r"F:\test/result-10000.txt", np.asarray(result), delimiter=",",fmt="%d")




    #dis = np.linalg.norm(train_data_list[0][label3[i]] - query_data)
    #lis.append(dis)


    #findmax.append(list(similarity.eval()))

    #lis.append(np.mean(findmax))
'''
import cv2
for i in range(10):
    y=train_data_list[0][i]
    print(train_label_list[i])
    cv2.imshow('img',y)
    cv2.waitKey(0)
'''
#lis = []
#lis=results[0].tolist()
result = np.argsort(result)[::-1]
#lis = np.argsort(lis)#范数专用
print('pppppppppppppppppp')

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
joblib.dump(new_im, r"F:\test\eps-3.00m74-100000x.data")
joblib.dump(label_list, r"F:\test\eps-3.00m74-100000y.data")
#!/usr/bin/env python
# coding: utf-8

import sys
sys.exit()
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

print(x.shape)
print(label.shape)
print(x_test.shape)
print(y_test.shape)
train_accs = []
eval_accs = []
#     for i in range(70):
history = model.fit(new_im, label_list, batch_size=512, epochs=500, validation_data=(x_test, y_test), shuffle=True)
train_accs = history.history['accuracy']
eval_accs = history.history['val_accuracy']





