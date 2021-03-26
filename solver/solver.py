#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:04:48
@LastEditTime: 2020-07-13 17:03:29
@Description: file content
'''
import os, importlib, torch, shutil, cv2
from solver.basesolver import BaseSolver
from utils.utils import maek_optimizer, make_loss, calculate_psnr, calculate_ssim, save_config, save_net_config
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from importlib import import_module
from torch.autograd import Variable
from data.data import DatasetFromHdf5
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.config import save_yml
from model.base_net import CycleLoss
from model.utils import *
from PIL import Image
from pylab import *
from data.data import get_data

class Solver(BaseSolver):
    def __init__(self, cfg, name):
        super(Solver, self).__init__(cfg)
        self.init_epoch = self.cfg['schedule']
        
        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net

        self.model = net(
            num_channels=self.cfg['data']['n_colors'], 
            base_filter=64,  
            scale_factor=self.cfg['data']['upsacle'], 
            args = self.cfg
        )
        
        self.train_dataset = get_data(self.cfg, str(self.cfg['train_dataset'])+'/'+str(name)+'.png', str(self.cfg['train_dataset'])+'/'+str(name)+'.png', self.cfg['data']['upsacle'])
        self.train_loader = DataLoader(self.train_dataset, self.cfg['data']['batch_size'], shuffle=False,
            num_workers=self.num_workers)

        for iteration, batch in enumerate(self.train_loader, 1):
            lr, hr, bic, hr_ref, bic_ref, file_name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[4]), Variable(batch[5]), (batch[6])
        self.hr_ref = hr_ref
        self.lr = lr
        self.file_name = file_name

        self.noise_init = get_noise(32, 'noise', (48*4, 48*4))
        self.noise =  self.noise_init.detach().clone()

        self.optimizer = maek_optimizer(self.cfg['schedule']['optimizer'], cfg, self.model.parameters())
        self.loss = CycleLoss(scale=1/4, loss_type = 'MSE')

        self.log_name = self.cfg['algorithm'] + '_' + str(self.cfg['data']['upsacle']) + '_' + str(self.timestamp)
        # save log
        self.writer = SummaryWriter('log/' + str(self.log_name))
        save_net_config(self.log_name, self.model)
        save_yml(cfg, os.path.join('log/' + str(self.log_name), 'config.yml'))

    def train(self): 
        
        epoch_loss = 0
    
        if self.cuda:
            self.noise = self.noise_init.cuda(self.gpu_ids[0]) + (self.noise.normal_() * 0.03).cuda(self.gpu_ids[0])
        self.optimizer.zero_grad()               
        self.model.train()
        self.sr, out = self.model(self.noise, self.hr_ref)
        self.noise = out.detach()
        loss, _ = self.loss(self.sr, self.lr)

        epoch_loss = epoch_loss + loss.data

        loss.backward()

        self.optimizer.step()
        self.records['Loss'].append(epoch_loss / len(self.train_loader))
        self.writer.add_scalar('loss',self.records['Loss'][-1], self.epoch)
        print(str(self.epoch) + '/'+str(self.nEpochs), self.file_name, self.records['Loss'][-1])

    def save_img(self, img, img_name):
        save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
        # save img
        save_dir=os.path.join('results/',self.cfg['test']['type'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_fn = save_dir +'/'+ img_namezhengru 
        cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def check_gpu(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
              
            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >=0:
                    self.gpu_ids.append(gid)

            torch.cuda.set_device(self.gpu_ids[0]) 
            self.loss = self.loss.cuda(self.gpu_ids[0])

            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
      
            self.hr_ref = self.hr_ref.cuda(self.gpu_ids[0])
            self.lr = self.lr.cuda(self.gpu_ids[0])

    def check_pretrained(self):
        checkpoint = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['pretrain']['pre_sr'])
        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['net'])
            self.epoch = torch.load(checkpoint, map_location=lambda storage, loc: storage)['epoch']
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must less than the max epoch!")
        else:
            raise Exception("Pretrain path error!")

    def save_checkpoint(self):
        super(Solver, self).save_checkpoint()
        if self.records['Loss'] != [] and self.records['Loss'][-1] == np.array(self.records['Loss']).min():
            self.save_img(self.sr[0].cpu().data, self.file_name[0])

    def run(self):
        self.check_gpu()
        while self.epoch <= self.nEpochs:
            self.train()
            self.epoch += 1
        self.save_img(self.sr[0].cpu().data, self.file_name[0])
        # save_config(self.log_name, 'Training done.')
