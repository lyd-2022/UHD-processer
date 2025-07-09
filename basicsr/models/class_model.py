import torch
from torch import nn as nn
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch.nn import functional as F
import math
from copy import deepcopy
import os

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import torch
import numpy as np



@MODEL_REGISTRY.register()
class ClassModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(ClassModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.celoss=nn.CrossEntropyLoss()

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq_path = data['lq_path']
        if self.is_train:
            self.label = data['label'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.celoss(self.output, self.label)
            l_total += l_pix
            loss_dict['l_ce'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.label)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):

        print(self.lq_path)

        # if 'snow' in self.lq_path:
        #     print(ddd)
        # elif '_r.' in self.lq_path[0] or 'rain' in self.lq_path[0]:
        #     self.label='r'
        # elif '_h.' in self.lq_path[0] or 'haze' in self.lq_path[0]:
        #     self.label='h'
        # elif '_b.' in self.lq_path[0] or 'blur' in self.lq_path[0]:
        #     self.label='b'
        # elif '_n.' in self.lq_path[0] or 'noise' in self.lq_path[0]:
        #     self.label='n'
        # elif '_j.' in self.lq_path[0] or 'jpeg' in self.lq_path[0]:
        #     self.label='j'
        # elif '_d.' in self.lq_path[0] or 'dark' in self.lq_path[0]:
        #     self.label='d'
        # elif '_s.' in self.lq_path[0] or 'sr' in self.lq_path[0]:
        #     self.label='s'
        # else:
        #     self.label='None'
        
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img,save_as_dir=None,dwt_levels=None,bs_j=None,wt_function=None,iwt_functio=None):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img,save_as_dir=None,dwt_levels=None,bs_j=None,wt_function=None,iwt_functio=None):

        self.is_train=False

        dataset_name = dataloader.dataset.opt['name']
        pbar = tqdm(total=len(dataloader), unit='image')

        data_num=0
        acc_num=0

        for idx, val_data in enumerate(dataloader):
            data_num+=1
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            result = visuals['result']
            
            label = val_data['label'].float().cpu()
            # del self.label

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            mask = (result == result.max(dim=1,keepdim=True)[0]).to(dtype=torch.float32)
            # result=torch.mul(mask,result)
            assert mask.size()==label.size()
            if not torch.equal(mask,label):
                print('wrong')
                print(result)
                print(label)
            else:
                acc_num+=1
            
            log_str = f'img {img_name}'
        
            log_str += f'\t # result: {result}'
            logger = get_root_logger()
            logger.info(log_str)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()
        acc_rate=acc_num/data_num
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger,acc_rate)

        self.is_train=True

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger,acc_rate):
        log_str = f'Validation {dataset_name}\n'
        
        log_str += f'\t # acc: {acc_rate:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)

        if tb_logger is not None:
            tb_logger.add_scalar('acc', acc_rate, current_iter)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        
        # out_dict['label'] = self.label

        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)