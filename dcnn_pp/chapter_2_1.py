#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
 * @Author: shifaqiang(石发强)--[14061115@buaa.edu.cn] 
 * @Date: 2018-09-21 16:11:31 
 * @Last Modified by:   shifaqiang 
 * @Last Modified time: 2018-09-21 16:11:31 
 * @Desc: 用mxnet框架实现一个基本的神经网络，训练模型将一组随机数分为两类
'''
import logging
import math
import numpy as np
import mxnet as mx

# setting debug information
logging.getLogger().setLevel(logging.DEBUG)

# setting hyper-parameters
n_sample = 10**4
batch_size = 10
learning_rate = 0.1
n_epoch = 10

# create random dataset，a sample has two random number a,b within (0,1), and the laber is the max(a,b)
train_x = [[np.random.uniform(0, 1), np.random.uniform(0, 1)]
           for n in range(n_sample)]
train_y = [max(item) for item in train_x]
test_x = [[np.random.uniform(0, 1), np.random.uniform(0, 1)]
           for n in range(n_sample)]
test_y = [max(item) for item in train_x]
train_iter = mx.io.NDArrayIter(data=np.array(train_x), label={"reg_label": train_y}, batch_size=batch_size, shuffle=True)
test_iter = mx.io.NDArrayIter(data=np.array(test_x), label={"reg_label": test_y}, batch_size=batch_size,)

# model defination
data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data=data, num_hidden=10, name="fc1")
act1 = mx.symbol.Activation(data=fc1, act_type="relu", name="act1")
fc2 = mx.symbol.FullyConnected(data=act1, num_hidden=10, name="fc2")
act2 = mx.symbol.Activation(data=fc2, act_type="relu", name="fc2")
fc3 = mx.symbol.FullyConnected(data=act2, num_hidden=1, name="fc3")
net = mx.symbol.LinearRegressionOutput(data=fc3, name="reg")
model = mx.mod.Module(net, label_names=(["reg_label"]))

# training
model.fit(train_iter, eval_data=test_iter, eval_metric="mse",
          initializer=mx.initializer.Uniform(0.5),
          optimizer="sgd",
          optimizer_params={"learning_rate": learning_rate,},
          num_epoch=n_epoch)

# print some parameters for observing
for para in model.get_params():
    print(para)