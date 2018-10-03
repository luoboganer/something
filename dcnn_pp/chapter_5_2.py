#!/usr/bin/env python 
# -*- coding:utf-8 -*- 
'''
 * @Author: shifaqiang(石发强)--[14061115@buaa.edu.cn] 
 * @Date: 2018-09-29 22:20:40 
 * @Last Modified by:   shifaqiang 
 * @Last Modified time: 2018-09-29 22:20:40 
 * @Desc: a deeper model and RecordIO data for cifar-10
'''

import mxnet as mx
import logging
import numpy as np

logging.getLogger().setLevel(logging.DEBUG)

batch_size = 50

# "Conv-BN-Act" model block
def CBA(src,suffix,num_filter,kernel,pad,stride=(1,1)):
    conv = mx.symbol.Convolution(data=src, kernel=kernel, num_filter=num_filter, pad=pad, stride=stride, name="conv{}".format(suffix))
    bn = mx.symbol.BatchNorm(data=conv, fix_gamma=False, name="bn{}".format(suffix))
    act = mx.symbol.Activation(data=bn, act_type="relu", name="act{}".format(suffix))
    return act

# full convolutional neural network and zoom out image by convolution stride
net = mx.symbol.Variable("data") # 3*28*28
net = CBA(net, suffix=1, num_filter=96, kernel=(3, 3), pad=(1, 1), stride=(1, 1))  # 96*28*28
net = CBA(net, suffix=2, num_filter=96, kernel=(3, 3), pad=(1, 1), stride=(1, 1))  # 96*28*28
net = CBA(net, suffix=3, num_filter=96, kernel=(3, 3), pad=(1, 1), stride=(2, 2))  # 96*14*14
net = CBA(net, suffix=4, num_filter=192, kernel=(3, 3), pad=(1, 1), stride=(1, 1))  # 192*14*14
net = CBA(net, suffix=5, num_filter=192, kernel=(3, 3), pad=(1, 1), stride=(1, 1))  # 192*14*14
net = CBA(net, suffix=6, num_filter=192, kernel=(3, 3), pad=(0, 0), stride=(2, 2))  # 192*7*7
net = CBA(net, suffix=7, num_filter=192, kernel=(3, 3), pad=(0, 0), stride=(1, 1))  # 192*5*5
net = CBA(net, suffix=8, num_filter=192, kernel=(1, 1), pad=(0, 0), stride=(1, 1))  # 192*5*5
net = CBA(net, suffix=9, num_filter=10, kernel=(1, 1), pad=(0, 0), stride=(1, 1))  # 10*5*5
net = mx.symbol.Pooling(net, global_pool=True, pool_type="avg", name="pooling", kernel=(1, 1))  # 10*1*1
net = mx.symbol.flatten(net, name="flatten")  # 10
net = mx.symbol.SoftmaxOutput(data=net, name="softmax")
shape = {"data": (batch_size, 3, 28, 28)}
mx.viz.print_summary(symbol=net, shape=shape)
ctx = [mx.gpu(i) for i in [0, 1, 2, 3]]
model = mx.mod.Module(symbol=net, context=ctx)

# load data
train_iter = mx.io.ImageRecordIter(
    path_imgrec="./data/cifar/train.rec",
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    rand_crop=True,
    rand_mirro=True,
    random_h=10,
    random_s=20,
    random_l=25,
    max_random_scale=1.2,
    min_random_scale=0.88,
    max_rotate_angle=20,
    max_aspect_ratio=0.15,
    max_shear_ratio=0.10,
    shuffle=True,
    fill_value=0,
)
val_iter = mx.io.ImageRecordIter(
    path_imgrec="./data/cifar/test.rec",
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    rand_crop=False,
    rand_mirro=False,
    shuffle=False,
)
# training process
import os
prefix = "./data/model/"
if not os.path.exists(prefix):
    os.mkdir(prefix)
    
model.fit(
    train_data=train_iter,
    eval_data=val_iter,
    eval_metric="acc",
    initializer=mx.init.MSRAPrelu(slope=0.0),
    optimizer="sgd",
    optimizer_params={"learning_rate": 0.5, "lr_scheduler": mx.lr_scheduler.FactorScheduler(step=50000 / batch_size, factor=0.98)},
    num_epoch=200,
    batch_end_callback=mx.callback.Speedometer(batch_size, 50000 / batch_size),
    epoch_end_callback=mx.callback.do_checkpoint("./data/model/cifar"),
)