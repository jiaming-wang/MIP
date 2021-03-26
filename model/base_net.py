#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-22 09:46:19
@LastEditTime: 2020-07-11 19:50:07
@Description: file content
'''
import torch
import math
import torch.optim as optim
import torch.nn as nn
from importlib import import_module
import torch.nn.functional as F
import numpy as np
from model.utils import *

######################################
#            common model
######################################
class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, activation='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        if scale == 3:
            modules.append(ConvBlock(n_feat, 9 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(3))
            if bn: 
                modules.append(torch.nn.BatchNorm2d(n_feat))
        else:
            for _ in range(int(math.log(scale, 2))):
                modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
                modules.append(torch.nn.PixelShuffle(2))
                if bn: 
                    modules.append(torch.nn.BatchNorm2d(n_feat))
        
        self.up = torch.nn.Sequential(*modules)
        
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None, pad_model=None):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        if self.norm =='batch':
            self.norm = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.norm = torch.nn.InstanceNorm2d(output_size)
        else:
            self.norm = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None
        
        if self.pad_model == None:   
            self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
            self.padding = None
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None, [self.padding, self.conv, self.norm, self.act])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)

######################################
#           loss function
######################################
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight = 1):
        super(TVLoss, self).__init__()

        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class CycleLoss(nn.Module):
    def __init__(self, scale = 1/2, loss_type = 'MSE'):
        super(CycleLoss, self).__init__()

        self.scale = scale

        if loss_type == "MSE":
            self.loss = nn.MSELoss()
        elif loss_type == "L1":
            self.loss = nn.L1Loss()
        else:
            raise ValueError

    def forward(self, x_sr, x_lr):
        downsampler = Downsampler(n_planes=3, factor=4, phase=0.5, preserve_size=True).cuda(0)
        down_x = downsampler(x_sr)

        # down_x = F.interpolate(x_hr, scale_factor=self.scale, mode='bicubic')
        return self.loss(down_x, x_lr), down_x

######################################
#           resnet_block
###################################### 
class ResnetBlock(torch.nn.Module):
    def __init__(self, input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu', norm='batch', pad_model=None):
        super().__init__()

        self.norm = norm
        self.pad_model = pad_model
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.scale = scale
        
        if self.norm =='batch':
            self.normlayer = torch.nn.BatchNorm2d(input_size)
        elif self.norm == 'instance':
            self.normlayer = torch.nn.InstanceNorm2d(input_size)
        else:
            self.normlayer = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        if self.pad_model == None:   
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None, [self.pad, self.conv1, self.normlayer, self.act, self.pad, self.conv2, self.normlayer, self.act])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out
        
class ResnetBlock_triple(ResnetBlock):
    def __init__(self, *args, middle_size, output_size, **kwargs):
        ResnetBlock.__init__(self, *args, **kwargs)

        if self.norm =='batch':
            self.normlayer1 = torch.nn.BatchNorm2d(middle_size)
            self.normlayer2 = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.normlayer1 = torch.nn.InstanceNorm2d(middle_size)
            self.normlayer2 = torch.nn.BatchNorm2d(output_size)
        else:
            self.normlayer1 = None
            self.normlayer2 = None
            
        if self.pad_model == None:   
            self.conv1 = torch.nn.Conv2d(self.input_size, middle_size, self.kernel_size, self.stride, self.padding, bias=self.bias)
            self.conv2 = torch.nn.Conv2d(middle_size, output_size, self.kernel_size, self.stride, self.padding, bias=self.bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad= nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv1 = torch.nn.Conv2d(self.input_size, middle_size, self.kernel_size, self.stride, 0, bias=self.bias)
            self.conv2 = torch.nn.Conv2d(middle_size, output_size, self.kernel_size, self.stride, 0, bias=self.bias)

        layers = filter(lambda x: x is not None, [self.pad, self.conv1, self.normlayer1, self.act, self.pad, self.conv2, self.normlayer2, self.act])
        self.layers = nn.Sequential(*layers) 

    def forward(self, x):

        residual = x
        out = x
        out= self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out