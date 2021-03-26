import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
from .utils import *
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, scale_factor, args):
        super(Net, self).__init__()

        self.skip = skip()
        self.unet = UNet()

    def forward(self, x, ref_hr):

        x_input = x
        x = self.skip(x)
        x_ref = self.unet(ref_hr)
        x_out = x_ref + x_input  
        x_out = torch.sigmoid(x_out)

        return x, x_out   

class STN_1(nn.Module):
    def __init__(self):
        super(STN_1, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size = 7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size = 5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 44 * 44, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 44 * 44)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

class STN_2(nn.Module):
    def __init__(self):
        super(STN_2, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size = 7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size = 5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 20 * 20, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 20 * 20)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

class STN_3(nn.Module):
    def __init__(self):
        super(STN_3, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(256, 8, kernel_size = 7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size = 5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 8 * 8, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 8 * 8)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        n_channels = 3
        bilinear = True
        factor = 2 

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)

        body = [
            ResnetBlock(256, 3, 1, 1, True, 1, 'prelu', norm=None, pad_model='reflection') for _ in range(3)
        ]
        self.body = nn.Sequential(*body)

        self.up2 = Up(384, 256, bilinear)
        self.up3 = Up(320, 128, bilinear)

        self.outc = OutConv(128, 32)

        self.stn_3 = STN_3()
        self.stn_1 = STN_1()
        self.stn_2 = STN_2()

    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.body(x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = torch.sigmoid(self.outc(x))
        return logits

class skip(nn.Module):
    def __init__(self):
        super(skip, self).__init__()

        num_input_channels=32
        num_output_channels=3
        num_channels_down=[128, 128, 128, 128, 128]
        num_channels_up=[128, 128, 128, 128, 128]
        num_channels_skip=[4, 4, 4, 4, 4]
        filter_size_down=3
        filter_size_up=3
        filter_skip_size=1 
        need_sigmoid=True
        need_bias=True
        pad='reflection'
        upsample_mode='bilinear'
        downsample_mode='stride'
        act_fun='LeakyReLU'
        need1x1_up=True

        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        n_scales = len(num_channels_down) 

        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
            upsample_mode   = [upsample_mode]*n_scales

        if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
            downsample_mode   = [downsample_mode]*n_scales
    
        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
            filter_size_down   = [filter_size_down]*n_scales

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
            filter_size_up   = [filter_size_up]*n_scales

        last_scale = n_scales - 1 

        cur_depth = None

        self.model = nn.Sequential()
        self.model_tmp = self.model

        input_depth = num_input_channels
        for i in range(len(num_channels_down)):

            self.deeper = nn.Sequential()
            self.skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                self.model_tmp.add(Concat(1, self.skip, self.deeper))
            else:
                self.model_tmp.add(self.deeper)
        
            self.model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

            if num_channels_skip[i] != 0:
                self.skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
                self.skip.add(bn(num_channels_skip[i]))
                self.skip.add(act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

            self.deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
            self.deeper.add(bn(num_channels_down[i]))
            self.deeper.add(act(act_fun))

            self.deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
            self.deeper.add(bn(num_channels_down[i]))
            self.deeper.add(act(act_fun))

            self.deeper_main = nn.Sequential()

            if i == len(num_channels_down) - 1:
            # The deepest
                k = num_channels_down[i]
            else:
                self.deeper.add(self.deeper_main)
                k = num_channels_up[i + 1]

            self.deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

            self.model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
            self.model_tmp.add(bn(num_channels_up[i]))
            self.model_tmp.add(act(act_fun))


            if need1x1_up:
                self.model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
                self.model_tmp.add(bn(num_channels_up[i]))
                self.model_tmp.add(act(act_fun))

            input_depth = num_channels_down[i]
            self.model_tmp = self.deeper_main

        self.model.add(conv(num_channels_up[0], 3, 1, bias=need_bias, pad=pad))
        if need_sigmoid:
            self.model.add(nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)

        return x    
