#!/usr/bin/env python 
# -*- coding:utf-8 -*- 
'''
 * @Author: shifaqiang(石发强)--[14061115@buaa.edu.cn] 
 * @Date: 2018-09-29 10:00:26 
 * @Last Modified by:   shifaqiang 
 * @Last Modified time: 2018-09-29 10:00:26 
 * @Desc: sort out cifar-10 dataset for MNXET RecordIO format

    * you can download cifar-10 mirro from cifar offical website or other mirro website
    * example
        wget https://pjreddie.com/media/files/cifar.tgz
        tar xzf cifar.tgz

'''

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="help to sort out cifar-10 dataset for MXNET RecordIO format")
    parser.add_argument("--path", type=str, default="./cifar", help="original cifar-10 files path")
    return parser.parse_args()


def move_images(target,labels):
    
    for i in range(len(labels)):
        directory =target + str(i)
        if not os.path.exists(directory):
            os.mkdir(directory)
    
    for filename in os.listdir(target):
        if not filename.endswith(".png"):
            continue
        print(filename)
        tmp = filename.split("_")
        index = labels.index(tmp[1][:-4])
        os.rename(target+filename,target+str(index)+"/"+filename)

def main():
    
    args = get_args()
    file = open(args.path + "/labels.txt", "r")
    labels = file.readlines()
    file.close()
    for i in range(len(labels)):
        labels[i] = labels[i].strip()
    
    move_images(target=args.path+"/train/",labels=labels)
    move_images(target=args.path+"/test/",labels=labels)

if __name__ == '__main__':
    main()