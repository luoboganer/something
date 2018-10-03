#!/usr/bin/env python 
# -*- coding:utf-8 -*- 
'''
 * @Author: shifaqiang(石发强)--[14061115@buaa.edu.cn] 
 * @Date: 2018-09-28 19:45:16 
 * @Last Modified by:   shifaqiang 
 * @Last Modified time: 2018-09-28 19:45:16 
 * @Desc: data augmentation and rebuild model structure for Fashion-MNIST
'''

import numpy as np 
import os
import gzip
import mxnet as mx
import struct
import logging
import matplotlib.pyplot as plt

# setting up debug inforamtion output
logging.getLogger().setLevel(logging.DEBUG)

# read MNIST foramt data
def read_data(label_file,image_file):
    with gzip.open(label_file) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.frombuffer(flbl.read(), dtype=np.int8)
    with gzip.open(image_file) as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        images = np.frombuffer(fimg.read(), dtype=np.uint8)
        images = images.reshape(len(labels), 1, rows, cols)
        images = images.astype(np.float32) / 255.0  # normalization pixel value to  (0,1)
    return (labels, images)

# load data
(train_labels, train_images) = read_data("./data/fashion-minist/train-labels-idx1-ubyte.gz", "./data/fashion-minist/train-images-idx3-ubyte.gz")
(validation_labels, validation_images) = read_data("./data/fashion-minist/t10k-labels-idx1-ubyte.gz", "./data/fashion-minist/t10k-images-idx3-ubyte.gz")
batch_size = 32
# train_iter = mx.io.NDArrayIter(train_images, label=train_labels, batch_size=batch_size, shuffle=True)
validation_iter = mx.io.NDArrayIter(validation_images, label=validation_labels, batch_size=batch_size, shuffle=False)

# defination of model
def CBA(src, suffix, num_filter, kernel, pad):
    conv = mx.symbol.Convolution(data=src, kernel=kernel, num_filter=num_filter, pad=pad, name="conv{}".format(suffix))
    bn = mx.symbol.BatchNorm(data=conv, fix_gamma=False, name="bn{}".format(suffix))
    act = mx.symbol.Activation(data=bn, act_type="relu", name="act{}".format(suffix))
    return act
def LAYER(src, layer, num_filter, pad):
    conv1 = CBA(src, suffix=layer + "1", num_filter=num_filter, kernel=(3, 3), pad=pad)
    conv2 = CBA(conv1, suffix=layer + "2", num_filter=num_filter, kernel=(3, 3), pad=pad)
    pool = mx.symbol.Pooling(data=conv2, name="pooling" + layer, pool_type="max", kernel=(2, 2), stride=(2, 2))
    return pool

data = mx.symbol.Variable("data") # input data
shape = {"data": (batch_size, 1, 28, 28)} # input data shape
net = LAYER(data, "1", num_filter=32, pad=(1, 1)) # change image from 1*28*28 to 32*14*14
net = LAYER(net, "2", num_filter=64, pad=(1, 1)) # change image from 32*14*14 to 64*7*7 
net = LAYER(net, "3", num_filter=128, pad=(1, 1)) # change image from 64*7*7 to 64*3*3
net = CBA(net, "4", num_filter=256, kernel=(3, 3), pad=(0, 0))  # change image from 64*3*3 to 128*1*1
net = mx.symbol.Convolution(net, name="final", num_filter=10, kernel=(1, 1)) # change image from 128*1*1 to 10*1*1
net = mx.symbol.flatten(net, name="flatten") # change image from 10*1*1 to 10
net = mx.symbol.SoftmaxOutput(data=net, name="softmax")
ctx=[mx.gpu(i) for i in [0,1,2,3,]]
model = mx.mod.Module(symbol=net, context=ctx)

# print network parameters
mx.viz.print_summary(symbol=net, shape=shape)

# manually to runn 40 epoches
for epoch in range(40):
    # data augmentation, in fact this task can be done in another process
    aug_img = train_images.copy()
    for i in range(len(aug_img)):
        # flip around with a probability of 0.5
        if np.random.random() < 0.5:
            # channel 0 of image i
            aug_img[i][0] = np.fliplr(aug_img[i][0])
        # shift up to 2 pixel (left or right)
        amt = np.random.randint(0, 3)
        if amt > 0:
            if np.random.random() < 0.5:
                aug_img[i][0] = np.pad(aug_img[i][0], ((0, 0), (0, amt)), mode="constant")[:,: - amt]
            else:
                aug_img[i][0] = np.pad(aug_img[i][0], ((0, 0), (amt, 0)), mode="constant")[:, amt:]
        # shift up to 2 pixel (up or down)
        amt = np.random.randint(0, 3)
        if amt > 0:
            if np.random.random() < 0.5:
                aug_img[i][0] = np.pad(aug_img[i][0], ((amt, 0), (0, 0)), mode="constant")[: - amt,:]
            else:
                aug_img[i][0] = np.pad(aug_img[i][0], ((0, amt), (0, 0)), mode="constant")[amt:,:]
        # randomly clear 5*5 data
        x_size = np.random.randint(1, 6)
        y_size = np.random.randint(1, 6)
        x_begin = np.random.randint(0, 28 - x_size + 1)
        y_begin = np.random.randint(0, 29 - y_size + 1)
        aug_img[i][0][x_begin:x_begin + x_size][y_begin:y_begin + y_size] = 0
    
    # reset data iteration for each epoch
    train_iter = mx.io.NDArrayIter(data=aug_img, label=train_labels, shuffle=True, batch_size=batch_size)
    # adjust learning_rate for each epoch
    lr = 0.06 * pow(0.95, epoch)
    # print current epoch information
    print("epoch:{}\tlearning_rate:{}".format(epoch, lr))
    model.fit(train_iter, eval_data=validation_iter, eval_metric="acc", optimizer="sgd", optimizer_params={"learning_rate": lr}, num_epoch=1, batch_end_callback=mx.callback.Speedometer(batch_size, 60000 / batch_size))
    