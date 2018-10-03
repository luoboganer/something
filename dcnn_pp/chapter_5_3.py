#!/usr/bin/env python 
# -*- coding:utf-8 -*- 
'''
 * @Author: shifaqiang(石发强)--[14061115@buaa.edu.cn] 
 * @Date: 2018-09-30 19:00:05 
 * @Last Modified by:   shifaqiang 
 * @Last Modified time: 2018-09-30 19:00:05 
 * @Desc: defination of residual neural network with pre-act structure
'''

import mxnet as mx

def symbol(num_filter):
    net=mx.symbol.Variable("data")
    # pre-process
    net=mx.symbol.Convolution(data=net,name="convPRE",kernel=(3,3),pad=(1,1),num_filter=num_filter)
    # residual structure
    for i in range(6):
        # a residual block 
        identity=net
        net=mx.symbol.BatchNorm(net,name="bnA{}".format(i),fix_gamma=False)
        net=mx.symbol.Activation(net,name="actA{}".format(i),act_type="relu")
        net=mx.symbol.Convolution(net,name="convA{}".format(i),kernel=(3,3),pad=(1,1),num_filter=num_filter)
        net=mx.symbol.BatchNorm(net,name="bnB{}".format(i),fix_gamma=False)
        net=mx.symbol.Activation(net,name="actB{}".format(i),act_type="relu")
        net=mx.symbol.Convolution(net,name="convB{}".format(i),kernel=(3,3),pad=(1,1),num_filter=num_filter)
        net=net+identity

    # tail of network
    net=mx.symbol.BatchNorm(net,name="bnFinal",fix_gamma=False)
    net=mx.symbol.Activation(net,name="actFinal",act_type="relu")
    # merge to one channel
    net=mx.symbol.Convolution(net,name="convFinal",kernel=(1,1),num_filter=1)
    net=mx.symbol.flatten(net)
    net=mx.symbol.SoftmaxOutput(net,name="softmax")

    return net

def main():
    # check parameters
    batch_size=32
    net=symbol(num_filter=128)
    shape={"data":(batch_size,8,19,19)}

    mx.viz.print_summary(net,shape=shape)
    mx.viz.plot_network(symbol=net,shape=shape).view()

if __name__ == '__main__':
    main()


