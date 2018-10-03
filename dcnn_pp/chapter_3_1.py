#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
 * @Author: shifaqiang(石发强)--[14061115@buaa.edu.cn] 
 * @Date: 2018-09-22 20:14:44 
 * @Last Modified by:   shifaqiang 
 * @Last Modified time: 2018-09-22 20:14:44 
 * @Desc: a implementation of MLP based on MINIST dataset with MXNET framework
'''

import mxnet as mx
import os
import gzip
import numpy as np
import struct
import logging
import matplotlib.pyplot as plt

# images format is [NCWH] in MXNET

# setting up debug information
logging.getLogger().setLevel(logging.DEBUG)

# read MINIST format data


def read_data(lable_url, image_url):
    # open label file
    with gzip.open(lable_url) as flbl:
        # read file header
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    # open image file
    with gzip.open(image_url) as fimg:
        # read image file header
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8)
        image = image.reshape(len(label), 1, rows, cols)
        # data normalization: from (0,255) to (0,1)
        image = image.astype(np.float)/255.0
    return (label, image)


# read data
prefix = "./data/minist/"
fashion_prefix = "./data/fashion-minist/"
train_lbl, train_img = read_data(
    fashion_prefix+"train-labels-idx1-ubyte.gz", fashion_prefix+"train-images-idx3-ubyte.gz")
test_lbl, test_img = read_data(
    fashion_prefix+"t10k-labels-idx1-ubyte.gz", fashion_prefix+"t10k-images-idx3-ubyte.gz")
batch_size = 32  # setting up hyper-parameters
train_iter = mx.io.NDArrayIter(
    data=train_img, label=train_lbl, batch_size=batch_size, shuffle=True)
test_iter = mx.io.NDArrayIter(data=test_img, label=test_lbl, batch_size=batch_size)

# show some data examples
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(train_img[i].reshape(28, 28), cmap="Greys_r")
    plt.axis("off")
plt.show()
print("labels: {}".format(train_lbl[:10]))

# defination of model
data=mx.symbol.Variable("data")
flatten=mx.symbol.flatten(data)
fc1=mx.symbol.FullyConnected(data=flatten, num_hidden=128, name="fc1")
act1 = mx.symbol.Activation(data=fc1, act_type="relu", name="act1")
fc2 = mx.symbol.FullyConnected(data=act1, num_hidden=64, name="fc2")
act2 = mx.symbol.Activation(data=fc2, act_type="relu", name="act2")
fc3 = mx.symbol.FullyConnected(data=act2, num_hidden=10, name="fc3")
net = mx.symbol.SoftmaxOutput(data=fc3, name="softmax")
# model = mx.mod.Module(net, context=mx.cpu())
model = mx.mod.Module(net, context=mx.gpu())

# print some neural network paprameters for observing
shape = {"data": (batch_size, 1, 28, 28,)}
mx.viz.print_summary(symbol=net, shape=shape)
# mx.viz.plot_network(symbol=net, shape=shape).view()

# training
model.fit(train_data=train_iter, eval_data=test_iter,
          optimizer="sgd",
          optimizer_params={"learning_rate": 0.2,
                            "lr_scheduler":mx.lr_scheduler.FactorScheduler(step=60000/batch_size, factor=0.9),}, 
          num_epoch=20,
          batch_end_callback=mx.callback.Speedometer(batch_size, frequent=60000/batch_size))
