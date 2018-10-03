#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
 * @Author: shifaqiang(石发强)--[14061115@buaa.edu.cn] 
 * @Date: 2018-10-02 21:02:59 
 * @Last Modified by:   shifaqiang 
 * @Last Modified time: 2018-10-02 21:02:59 
 * @Desc: a implementation of DCGAN(deep convolutional generative adverisial netowrk) with MXNET framwork
       
        data file [celeba.rec] link(baiduNetdisk):
           链接: https://pan.baidu.com/s/1H2hHVPz4tIPkZt7BPqMS8w 提取码: p5nh
        please downlaod all file celeba.rec and place it to ./data/celebA/
        the file tree should be:
            ./data
                |-----celebA
                        |-----celeba.rec
                        |-----celeba.idx
                        |-----celeba.lst

'''
import mxnet as mx
import numpy as np
import datetime
import cv2
import os
import time

# 定义网络需要的一些辅助函数
#----------------------------------------------------------------------
def leaky(s, name):
    leaky_slope = 0.2
    return mx.symbol.LeakyReLU(data=s, act_type="prelu", slope=leaky_slope, name=name)
def batchNorm(s, name):
    return mx.symbol.BatchNorm(data=s, name=name, fix_gamma=True, eps=1e-5+1e-12)
def upsize2x(s, name, num_filter):
    return mx.symbol.Deconvolution(data=s, name=name, num_filter=num_filter, kernel=(4, 4), stride=(2, 2), pad=(1, 1))
def downsize2x(s, name, num_filter):
    return mx.symbol.Convolution(data=s, name=name, num_filter=num_filter, kernel=(4, 4), stride=(2, 2), pad=(1, 1))

def upsize2x_BN_ACT(s, name, num_filter):
    s = upsize2x(s, name=name, num_filter=num_filter)
    s = batchNorm(s, name="{}_bn".format(name))
    s = leaky(s, name="{}_act".format(name))
    return s
def downsize2x_BN_ACT(s, name, num_filter):
    s = downsize2x(s, name=name, num_filter=num_filter)
    s = batchNorm(s, name="{}_bn".format(name))
    s = leaky(s, name="{}_act".format(name))
    return s

#----------------------------------------------------------------------
def get_symbol(ngf, ndf, nc):
    """defination of D and G netowrk symbol"""
    g_net = mx.symbol.Variable("rand")
    # upsize to 4*4
    g_net = mx.symbol.Deconvolution(data=g_net, name="g1", kernel=(4, 4), num_filter=8*ngf)
    g_net = batchNorm(g_net, name="g1_bn")
    g_net = leaky(g_net, name="g1_act")
    
    g_net = upsize2x_BN_ACT(g_net, name="g2", num_filter=ngf*4)  # 8*8
    g_net = upsize2x_BN_ACT(g_net, name="g3", num_filter=ngf*2)  # 16*16
    g_net = upsize2x_BN_ACT(g_net, name="g4", num_filter=ngf)  # 32*32
    
    g_net = upsize2x(g_net, name="g5", num_filter=nc)  # 64*64
    # after tanh function, return g_net which are in (-1,1)    
    g_net = mx.symbol.Activation(g_net, act_type="tanh", name="gout")
    
    d_net = mx.symbol.Variable("data")  # 64*64
    d_net = downsize2x(d_net, name="d1", num_filter=ndf)  # downsize to 32*32
    d_net = leaky(d_net, name="g1_act")
    
    d_net = downsize2x_BN_ACT(d_net, name="d2", num_filter=ndf*2)  # 16*16
    d_net = downsize2x_BN_ACT(d_net, name="d3", num_filter=ndf*4)  # 8*8
    d_net = downsize2x_BN_ACT(d_net, name="d4", num_filter=ndf*8)  # 4*4
    
    d_net = mx.symbol.Convolution(d_net, name="d5", kernel=(4, 4), num_filter=1)
    d_net = mx.symbol.flatten(d_net, name="d_flatten")
    d_net = mx.symbol.LinearRegressionOutput(d_net, label=mx.symbol.Variable("label"), name="d_loss")
    
    return g_net, d_net

# #######################################################################
class RandIter(mx.io.DataIter):
    """z data iteration"""
    def __init__(self, batch_size, ndim):
        """Constructor"""
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [("rand", (batch_size, ndim, 1, 1))]
        self.provide_label = []
    #----------------------------------------------------------------------
    def iter_next(self):
        return True
    #----------------------------------------------------------------------
    def getdata(self):
        """get z from random sample"""
        return [mx.nd.random_normal(loc=0, scale=1, shape=(self.batch_size, self.ndim, 1, 1))]
    
########################################################################
class ImageIter(mx.io.DataIter):
    """x data iteration"""
    #----------------------------------------------------------------------
    def __init__(self, path, batch_size, data_shape, img_size):
        """Constructor"""
        self.internal = mx.io.ImageRecordIter(path_imgrec=path,
                                              data_shape=data_shape,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              resize=img_size,
                                              min_crop_size=img_size,
                                              max_crop_size=img_size,
                                              min_img_size=img_size,
                                              max_img_size=img_size,
                                              rand_crop=False,
                                              rand_mirror=True)
        self.provide_data = [("data", (batch_size, ) + data_shape)]
        self.provide_label = []
    #----------------------------------------------------------------------
    def reset(self):
        self.internal.reset()
    #----------------------------------------------------------------------
    def iter_next(self):
        return self.internal.iter_next()
    #----------------------------------------------------------------------
    def getdata(self):
        data = self.internal.getdata()
        data = data * (2.0 / 255.0) - 1.0  # normalize data to (-1,1)
        return [data]
    
# Auxiliary function for print generated image
def fill_buf(buf, i, img, shape):
    n = buf.shape[0] / shape[1]
    m = buf.shape[1] / shape[0]
    sx = (i % m) * shape[0]
    sy = (i / m) * shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img
def visual(title, x):
    x = x.transpose((0, 2, 3, 1))
    x = np.clip((x+1) * 255.0/2.0, 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(x.shape[0]))
    buff = np.zeors((int(n*x.shape[1]), int(n*x.shape[2]), int(n*x.shape[3])), dtype=np.uint8)
    for i, img in enumerate(x):
        fill_buf(buff, i, img, x.shape[1:3])
    cv2.imshow("tmp", buff)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
#----------------------------------------------------------------------
def facc(label, pred):
    """defination of metrics"""
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()

def main():
    
    print("initialization of hyper-parameteres...")
    # #########################################################################
    # setting up some related parameters
    ctx = [mx.gpu(i) for i in range(4)]
    stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    img_path = "./data/celebA/celeba.rec"
    n_channels = 3  # generally, 3 channels is RGB color space
    batch_size = 6
    img_size = 64
    Z = 100  # dimension of encoding
    
    ndf = 64  # the size of D network, disciminator
    ngf = 64  # the size of G network, generator
    
    # for the case which need to load old model to continue to train
    loadG = None
    loadD = None
    
    check_point_period = 1
    
    lr_G = 0.0002
    lr_D = 0.0002
    beta1_G = 0.5
    beta1_D = 0.5
    wd_G = 0
    wd_D = 0
    leaky_slope = 0.2  # leaky relu 的负轴泄漏程度
    # #########################################################################
    
    print("loading...")
    sym_g, sym_d = get_symbol(ngf, ndf, n_channels)
    train_iter = ImageIter(path=img_path, batch_size=batch_size, data_shape=(n_channels, img_size, img_size), img_size=img_size)
    rand_iter = RandIter(batch_size, ndim=Z)
    label = mx.nd.zeros((batch_size))
    
    # ########################## G model ############################################
    mod_g = mx.mod.Module(sym_g, context=ctx, data_names=("rand", ), label_names=None)
    mod_g.bind(data_shapes=rand_iter.provide_data, inputs_need_grad=True)
    if loadG:
        mod_g.init_params(initializer=mx.init.load(loadG))
    else:
        mod_g.init_params(initializer=mx.init.MSRAPrelu(slope=leaky_slope))
    mod_g.init_optimizer(optimizer="adam",
                         optimizer_params={"learning_rate": lr_G,
                                           "wd": wd_G,
                                           "beta1_G": beta1_G,})

    # ########################## D model ############################################
    mod_d = mx.mod.Module(sym_d, context=ctx, data_names=("data", ), label_names=("label", ))
    mod_d.bind(data_shapes=train_iter.provide_data, label_shapes=[('label', (batch_size))], inputs_need_grad=True)
    if loadD:
        mod_d.init_params(initializer=mx.init.load(loadD))
    else:
        mod_d.init_params(initializer=mx.init.MSRAPrelu(slope=leaky_slope))
    mod_g.init_optimizer(optimizer="adam",
                         optimizer_params={"learning_rate": lr_D,
                                           "wd": wd_D,
                                           "beta1_G": beta1_D,})
    #  ######################### training ###########################################
    print("Training...")
    mFake=mx.metric.CustomMetric(facc)
    mReal=mx.metric.CustomMetric(facc)

    prefix="./data/model/"
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    
    start_time = time.time()
    for epoch in range(500):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            # 将batch中的x送入D，要求D输出1
            label[:] = 1
            mod_d.label = [label]
            mod_d.forward(batch, is_train=True)
            mod_d.update_metric(mReal, [label])
            mod_d.backward()
            # 保存梯度待用
            grad_d1 = [[grad.copyto(grad.context) for grad in grads] for grads in mod_d._exec_group.grad_arrays]
            
            # 将rbatch中的z送入G训练
            rbatch = rand_iter.next()
            mod_g.forward(rbatch, is_train=True)
            out_g = mod_g.get_outputs()
            
            # 将G生成的x送入D，要求D输出0
            label[:] = 0
            mod_d.forward(mx.io.DataBatch(out_g, [label]), is_train=True)
            mod_d.update_metric(mFake, [label])
            mod_d.backward()
            
            # 将G在两种情况下的梯度相加
            for i in range(len(grad_d1)):
                for j in range(len(grad_d1[i])):
                    mod_d._exec_group.grad_arrays[i][j] += grad_d1[i][j]
            # 更新D的参数
            mod_d.update()
            
            # 调整G的参数，是的D(G(z))更接近1
            label[:] = 1
            mod_d.forward(mx.io.DataBatch(out_g, [label]), is_train=False)
            mod_d.backward()  # 反向传播，获得D的梯度
            mod_g.backward(mod_d.get_input_grads())  # 将D对于输入的梯度送入G反向传播
            mod_g.update()
        
        # 完成一个epoch后，输出训练信息和生成的图像
        print("epoch:{}\treal:{}\tfake:{}\ttime:{}".format(epoch, mReal.get()[1], mFake.get()[1], time.time() - start_time))
        start_time = time.time()
        mReal.reset()
        mFake.reset()
        # 输出epoch的图像
        visual("gout", out_g[0].asnumpy())
        if check_point_period > 0 and epoch % check_point_period == (check_point_period - 1):
            # save network parameteres
            mod_g.save_params("./data/model/G_%s-%04d.params" % (stamp, epoch))
            mod_d.save_params("./data/model/D_%s-%04d.params" % (stamp, epoch))
            
if __name__ == '__main__':
    main()