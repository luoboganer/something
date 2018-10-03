#!/usr/bin/env python 
# -*- coding:utf-8 -*- 
'''
 * @Author: shifaqiang(石发强)--[14061115@buaa.edu.cn] 
 * @Date: 2018-10-01 09:36:01 
 * @Last Modified by:   shifaqiang 
 * @Last Modified time: 2018-10-01 09:36:01 
 * @Desc: a pre-cat ResNet model for cifar-10
'''
import mxnet as mx
import logging
import numpy as np

logging.getLogger().setLevel(logging.DEBUG)

batch_size=128

# defination of residual block
def residualBlock(net,suffix,n_block,n_filter,stride=(1,1)):
    for i in range(n_block):
        if i==0:
            net=mx.symbol.BatchNorm(net,name="bn{}A{}".format(suffix,i),fix_gamma=False)
            net=mx.symbol.Activation(net,name="act{}A{}".format(suffix,i),act_type="relu")
            # the branch of first residual block from here, which is different from subsequent ResNet block
            """
            这里第一个残差块与其它块不一样是因为第一个块涉及到用stride改变通道图大小，而后面的残差块均不改编通道图大小
            """
            pathway=mx.symbol.Convolution(net,name="adj{}".format(suffix),kernel=(1,1),stride=stride,num_filter=n_filter)
            # return to main branch
            net=mx.symbol.Convolution(net,name="conv{}A{}".format(suffix,i),kernel=(3,3),pad=(1,1),num_filter=n_filter,stride=stride)
            net=mx.symbol.BatchNorm(net,name="bn{}B{}".format(suffix,i),fix_gamma=False)
            net=mx.symbol.Activation(net,name="act{}B{}".format(suffix,i),act_type="relu")
            net=mx.symbol.Convolution(net,name="conv{}B{}".format(suffix,i),kernel=(3,3),pad=(1,1),num_filter=n_filter)
            net=net+pathway
        else:
            # the branch of other residual block from here
            pathway=net
            # return to main branch
            # 这里的其它残差块均没有stride，即默认为（1,1）
            net=mx.symbol.BatchNorm(net,name="bn{}A{}".format(suffix,i),fix_gamma=False)
            net=mx.symbol.Activation(net,name="act{}A{}".format(suffix,i),act_type="relu")
            net=mx.symbol.Convolution(net,name="conv{}A{}".format(suffix,i),kernel=(3,3),pad=(1,1),num_filter=n_filter)
            net=mx.symbol.BatchNorm(net,name="bn{}B{}".format(suffix,i),fix_gamma=False)
            net=mx.symbol.Activation(net,name="act{}B{}".format(suffix,i),act_type="relu")
            net=mx.symbol.Convolution(net,name="conv{}B{}".format(suffix,i),kernel=(3,3),pad=(1,1),num_filter=n_filter)
            net=net+pathway
    return net

def symbol():
    # defination of model computional graph
    net=mx.symbol.Variable("data")
    net=mx.symbol.BatchNorm(net,name="bnPre",fix_gamma=False)
    net=mx.symbol.Convolution(net,name="convPre",kernel=(3,3),pad=(1,1),num_filter=32) # from 3*32*32 to 32*32*32
    # 64*32*32
    net=residualBlock(net,suffix=1,n_block=3,n_filter=64)
    # 64*16*16
    net=residualBlock(net,suffix=2,n_block=3,n_filter=64,stride=(2,2))
    # 128*8*8
    net=residualBlock(net,suffix=3,n_block=3,n_filter=128,stride=(2,2))
    # output of residual block is convolutional layer, so we should continue to add BN and relu layer
    net=mx.symbol.BatchNorm(net,name="bnFinal",fix_gamma=False)
    net=mx.symbol.Activation(net,name="actFinal",act_type="relu")
    # change 128*8*8 to 128*1*1
    net=mx.symbol.Pooling(net,name="pooling",global_pool=True,pool_type="avg",kernel=(1,1))
    net=mx.symbol.flatten(net,name="flatten")
    net=mx.symbol.FullyConnected(net,num_hidden=10,name="fc")
    net=mx.symbol.SoftmaxOutput(net,name="softmax")
    return net

def load_data():
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
    return train_iter,val_iter

def main():
    net=symbol()
    # defination of data shape
    shape={"data":(batch_size,3,28,28)}
    mx.viz.print_summary(net,shape=shape)
    # mx.viz.plot_network(symbol=net,title="pre-act residual network model for cifar-10",save_format="pdf",shape=shape).view()
    num_gpus=1
    ctx=[mx.gpu(i) for i in range(num_gpus)]
    module=mx.mod.Module(net,context=ctx)

    # load data and define data iter
    train_iter,val_iter=load_data()

    # training process
    import os
    prefix = "./data/model/"
    if not os.path.exists(prefix):
        os.mkdir(prefix)
        
    module.fit(
        train_data=train_iter,
        eval_data=val_iter,
        eval_metric="acc",
        initializer=mx.init.MSRAPrelu(slope=0.0),
        optimizer="sgd",
        optimizer_params={"learning_rate": 0.5, "lr_scheduler": mx.lr_scheduler.FactorScheduler(step=50000 / batch_size, factor=0.984)},
        num_epoch=200,
        batch_end_callback=mx.callback.Speedometer(batch_size, 50000 / batch_size),
        epoch_end_callback=mx.callback.do_checkpoint(prefix+"cifarPre-act"),
    )

if __name__ == '__main__':
    main()