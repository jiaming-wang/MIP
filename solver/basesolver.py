#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:07:03
@LastEditTime: 2020-07-13 16:45:30
@Description: file content
'''
import os, torch, time
from utils.utils import  draw_curve_and_save, save_config
from data.dataset import data
from data.data import get_data
from torch.utils.data import DataLoader

class BaseSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nEpochs = cfg['nEpochs']
        self.checkpoint_dir = cfg['checkpoint']
        self.epoch = 1

        self.timestamp = int(time.time())

        if cfg['gpu_mode']:
            self.num_workers = cfg['threads']
        else:
            self.num_workers = 0

        self.records = {'Epoch': [], 'PSNR': [], 'SSIM': [], 'Loss': []}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            self.epoch = ckpt['epoch']
            self.records = ckpt['records']
        else:
            raise FileNotFoundError

    def save_checkpoint(self):
        self.ckp = {
            'epoch': self.epoch,
            'records': self.records,
        }

    def train(self):
        raise NotImplementedError
    
    def eval(self):
        raise NotImplementedError
    
    def run(self):
        while self.epoch <= self.nEpochs:
            self.train()
            self.eval()
            self.save_checkpoint()
            self.save_records()
            self.epoch += 1
        #self.logger.log('Training done.')
