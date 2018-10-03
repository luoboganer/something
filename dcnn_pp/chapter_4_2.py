#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 * @Author: shifaqiang(石发强)--[14061115@buaa.edu.cn] 
 * @Date: 2018-09-23 11:27:43 
 * @Last Modified by:   shifaqiang 
 * @Last Modified time: 2018-09-23 11:27:43 
 * @Desc: A simpler implementation of basic CNN with MINIST dataset
"""

import mxnet as mx
import os
import gzip
import numpy as np
import struct
import logging
import matplotlib.pyplot as plt

# using functions to define multi-layers so that we can quickly define the network structures


def layer(src, layer_id, kernel, num_filter):
    conv = mx.symbol.Convolution(data=src, name="conv{}".format(
        layer_id), num_filter=num_filter, kernel=kernel)
    bn = mx.symbol.BatchNorm(
        data=conv, name="bn{}".format(layer_id), fix_gamma=False)
    act = mx.symbol.Activation(
        data=bn, name="act{}".format(layer_id), act_type="relu")
    return act

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
        image = image.astype(np.float) / 255.0
    return (label, image)


# read data
prefix = "./data/minist/"
fashion_prefix = "./data/fashion-minist/"
train_lbl, train_img = read_data(
    prefix + "train-labels-idx1-ubyte.gz", prefix + "train-images-idx3-ubyte.gz"
)
test_lbl, test_img = read_data(
    prefix + "t10k-labels-idx1-ubyte.gz", prefix + "t10k-images-idx3-ubyte.gz"
)
batch_size = 32  # setting up hyper-parameters
train_iter = mx.io.NDArrayIter(
    data=train_img, label=train_lbl, batch_size=batch_size, shuffle=True
)
test_iter = mx.io.NDArrayIter(
    data=test_img, label=test_lbl, batch_size=batch_size)

# show some data examples
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(train_img[i].reshape(28, 28), cmap="Greys_r")
    plt.axis("off")
plt.show()
print("labels: {}".format(train_lbl[:10]))

# defination of model
data = mx.symbol.Variable("data")
pool1 = mx.symbol.Pooling(data=layer(src=data, layer_id=1, kernel=(
    5, 5), num_filter=32), kernel=(3, 3), stride=(2, 2), name="pool1", pool_type="max")
pool2 = mx.symbol.Pooling(data=layer(src=pool1, layer_id=2, kernel=(
    5, 5), num_filter=64), pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool2")
conv3 = mx.symbol.Convolution(
    data=pool2, name="conv3", kernel=(1, 1), num_filter=10)
pool3 = mx.symbol.Pooling(data=conv3, name="pool3",
                          global_pool=True, pool_type="avg", kernel=(1, 1))
flatten = mx.symbol.Flatten(data=pool3, name="flatten")
net = mx.symbol.SoftmaxOutput(data=flatten, name="softmax")

# model = mx.mod.Module(net, context=mx.cpu())
model = mx.mod.Module(net, context=mx.gpu())

# print some neural network paprameters for observing
shape = {"data": (batch_size, 1, 28, 28)}
mx.viz.print_summary(symbol=net, shape=shape)
# mx.viz.plot_network(symbol=net, shape=shape, save_format="pdf").view()

# training
model.fit(
    train_data=train_iter,
    eval_data=test_iter,
    optimizer="sgd",
    optimizer_params={
        "learning_rate": 0.3,
        "lr_scheduler": mx.lr_scheduler.FactorScheduler(
            step=60000 / batch_size, factor=0.9
        ),
    },
    num_epoch=20,
    batch_end_callback=mx.callback.Speedometer(
        batch_size, frequent=60000 / batch_size),
)

# make forward with trained model to observe its outputs
test_iter.reset()
model.forward(test_iter.next(), is_train=False)
out = model.get_outputs()[0].asnumpy()
result = zip(out.argmax(axis=1), test_iter.getlabel()
             [0].asnumpy(), out.max(axis=1))
for item in result:
    print(item)

# fine a error example to observe
test_iter.reset()
while True:
    flag = False
    model.forward(test_iter.next(), is_train=False)
    out = model.get_outputs()[0].asnumpy().argmax(axis=1)
    for i in range(len(out)):
        if out[i] != test_iter.getlabel()[0].asnumpy()[i]:
            print("predicted:{},label:{}".format(
                out[i], test_iter.getlabel()[0].asnumpy()[i]))
            plt.imshow(test_iter.getdata()[0].asnumpy()[
                       i].reshape(28, 28), cmap="Greys_r")
            plt.axis("off")
            plt.show()
            flag = True
            break
    if flag:
        break

# observe the model result for one single sample

from collections import namedtuple
my_batch = namedtuple("my_batch", ["data", "label"])
new_batch_size = 1
model.bind(data_shapes=[("data", (new_batch_size, 1, 28, 28))],
           label_shapes=[("softmax_label", (new_batch_size,))],
           force_rebind=True, for_training=False)
my_batchData = my_batch(
    [mx.nd.array(test_img[0].reshape(1, 1, 28, 28).astype(np.float32))], None)
model.forward(my_batchData)
np.set_printoptions(suppress=True)
print("predicted probality:", model.get_outputs()[0].asnumpy())
print("predicted_result:{}".format(
    model.get_outputs()[0].asnumpy().argmax(axis=1)))
print("label:{}".format(test_lbl[0]))

# save and load model
prefix="./data/model/"
if not os.path.exists(prefix):
    os.mkdir(prefix)
# model.save_params("{}minist.params".format(prefix))
# model._symbol.save("{}minist-symbol.json".format(prefix))
model.save_checkpoint("{}minist".format(prefix),epoch=20)

# symbol = mx.symbol.load("{}minist-symbol.json".format(prefix))
# model = mx.mod.Module(symbol, context=mx.gpu())
# model.init_params(initializer=mx.init.load("{}minist-0020.params".format(prefix)))