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
n_sample = 2
batch_size = 1
learning_rate = 0.1
n_epoch = 1

# create random dataset，a sample has two random number a,b within (0,1), and the laber is the max(a,b)
train_x = [[0.5,1],[0.2,0.6]]
train_y = [1,0.6]

train_iter = mx.io.NDArrayIter(data=np.array(train_x), label={"reg_label": train_y}, batch_size=batch_size, shuffle=False)

# model defination
data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data=data, num_hidden=2, name="fc1")
act1 = mx.symbol.Activation(data=fc1, act_type="relu", name="act1")
fc2 = mx.symbol.FullyConnected(data=act1, num_hidden=1, name="fc2")
net = mx.symbol.LinearRegressionOutput(data=fc2, name="reg")
model = mx.mod.Module(net, label_names=(["reg_label"]))

# manully bind data and label to model
model.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)
# manully initialization for testing
model.init_params(arg_params={
    "fc1_weight":mx.nd.array([[0.5,0],[0.5,1]]),
    "fc1_bias":mx.nd.array([0,0]),
    "fc2_weight":mx.nd.array([[0.5,0.5]]),
    "fc2_bias":mx.nd.array([0]),
})
# normal initialization method with MXNET method
# model.init_params(initializer=mx.init.Uniform(scale=0.1))
model.init_optimizer(optimizer="sgd",optimizer_params=({"learning_rate":0.1}))
# setting up metric index
metric=mx.metric.create("mse")




# manully training
# 手动训练每一个epoch
for epoch in range(n_epoch):
    train_iter.reset() # 每个epoch都要将数据迭代器和评价指标迭代器重置
    # metric.reset()
    for batch in train_iter:
        print("================input====================")
        print(batch.data)
        # 模型送入数据前馈传播
        model.forward(batch,is_train=True)
        print("===============output====================")
        # 得到前馈输出
        print(model.get_outputs())
        metric.reset()
        # 根据模型输出更新评价指标
        model.update_metric(metric,batch.label)
        print("===============metric====================")
        print(metric.get())
        # 计算梯度反向传播
        model.backward()
        print("================grads====================")
        print(model._exec_group.grad_arrays)
        # 更新模型参数
        model.update()
        print("===============params====================")
        print(model.get_params())
        # 完成一个epoch的训练