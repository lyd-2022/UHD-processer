import datetime
import logging
import math
import time
import torch
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path as osp
import sys
sys.path.append("/code/UHD-allinone")
from basicsr.data import create_dataloader, create_dataset
# from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
from basicsr.archs.wtconv.util import wavelet
import numpy as np
import random
import cv2
from basicsr.archs.wtconv.waveT import DWT_2D
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        new_name = new_name.replace('tb_logger', 'tb_logger_archived')
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        shutil.move(path, new_name)
    os.makedirs(path, exist_ok=True)


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('/output', 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
            
        elif phase.split('_')[0] == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters



# def create_train_val_dataloader(opt, logger):
#     # create train and val dataloaders
#     train_loader, val_loaders = None, []
#     for phase, dataset_opt in opt['datasets'].items():
#         if phase == 'train':
#             dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
#             train_set = build_dataset(dataset_opt)
#             train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
#             train_loader = build_dataloader(
#                 train_set,
#                 dataset_opt,
#                 num_gpu=opt['num_gpu'],
#                 dist=opt['dist'],
#                 sampler=train_sampler,
#                 seed=opt['manual_seed'])
#
#             num_iter_per_epoch = math.ceil(
#                 len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
#             total_iters = int(opt['train']['total_iter'])
#             total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
#             logger.info('Training statistics:'
#                         f'\n\tNumber of train images: {len(train_set)}'
#                         f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
#                         f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
#                         f'\n\tWorld size (gpu number): {opt["world_size"]}'
#                         f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
#                         f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
#         elif phase.split('_')[0] == 'val':
#             val_set = build_dataset(dataset_opt)
#             val_loader = build_dataloader(
#                 val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
#             logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
#             val_loaders.append(val_loader)
#         else:
#             raise ValueError(f'Dataset phase {phase} is not recognized.')
#
#     return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('/model/liuyidi/VAE/UHD-allinone/experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        # resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        resume_state = torch.load(resume_state_path, map_location='cpu')
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            os.makedirs(osp.join(opt['root_path'], 'tb_logger_archived'), exist_ok=True)
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    para_num = sum([param.nelement() for param in model.net_g.parameters() if param.requires_grad])
    logger.info(f"model parameters number:{para_num}")
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        print('####################################'
              '####################################'
              '#################')
    else:
        start_epoch = 0
        current_iter = 0
        # model.resume_training(resume_state)  # handle optimizers and schedulers
        # logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        # start_epoch = resume_state['epoch']
        # current_iter = resume_state['iter']

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()


    wt_filter, iwt_filter = wavelet.create_wavelet_filter('db1', 3, 3, torch.float)
    wt_filter = nn.Parameter(wt_filter, requires_grad=False)
    iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

    wt_function = wavelet.wavelet_transform_init(wt_filter)
    iwt_function = wavelet.inverse_wavelet_transform_init(iwt_filter)

    
    # wt_function = DWT_2D("haar")

    wt_function = wavelet.wavelet_transform_init(wt_filter)
    
    
    
    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
       
    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    logger_j = [True] * len(groups)
    dwt_levels = len(groups)

    # print('total_epochs',total_epochs)
    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            train_data = prefetcher.next()
            data_timer.record()

            current_iter += 1
            if current_iter < 10:
                continue

            if current_iter > total_iters:
                break
            # print('total_iters', total_iters)
            # print('current_iter', current_iter)
            # update learning rate
            # model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            
            
            ## -----frequency progessive learning-----
            j = ((current_iter>groups) !=True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

           

            mini_batch_size = mini_batch_sizes[bs_j]
            
            lq = train_data['lq']
            gt = train_data['gt']
            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]
            # for i in range(lq.shape[0]):
            #         lq_i = lq[i].cpu().numpy()
            #         #保存numpy数据
            #         np.save(f'/model/liuyidi/VAE/UHD-allinone/vis/ori/ft_00{i}.npy', lq_i)
            #         gt_i = gt[i].cpu().numpy()
            #         np.save(f'/model/liuyidi/VAE/UHD-allinone/vis/ori/ft_00{i}_gt.npy', gt_i)
            #         lq_i = np.transpose(lq_i, (1, 2, 0))
            #         gt_i = np.transpose(gt_i, (1, 2, 0))
            #         lq_i = (lq_i * 255).astype(np.uint8)
            #         gt_i = (gt_i * 255).astype(np.uint8)
            #         folder = '/model/liuyidi/VAE/UHD-allinone/vis/ori'
            #         if not os.path.exists(folder):
            #             os.makedirs(folder)
            #         cv2.imwrite(f'/model/liuyidi/VAE/UHD-allinone/vis/ori/ft_00{i}.png', lq_i)
            #         cv2.imwrite(f'/model/liuyidi/VAE/UHD-allinone/vis/ori/ft_00{i}_gt.png', gt_i)
            #         print('lq', lq.shape)
            if bs_j != len(groups) - 1:
                for j in range(dwt_levels-bs_j-1):
                    with torch.no_grad():
                        lq = wt_function(lq)[:,:,0,:,:]
                        gt = wt_function(gt)[:,:,0,:,:]
                        # #将lq和gt保存为图片
                        # for i in range(lq.shape[0]):
                        #     lq_i = lq[i].cpu().numpy()
                        #     np.save(f'/model/liuyidi/VAE/UHD-allinone/vis/level_{j}/ft_{i}_{j}.npy', lq_i)

                        #     gt_i = gt[i].cpu().numpy()
                        #     np.save(f'/model/liuyidi/VAE/UHD-allinone/vis/level_{j}/ft_{i}_{j}_gt.npy', gt_i)
                        #     lq_i = np.transpose(lq_i, (1, 2, 0))
                        #     gt_i = np.transpose(gt_i, (1, 2, 0))
                        #     lq_i = (lq_i * 255).astype(np.uint8)
                        #     gt_i = (gt_i * 255).astype(np.uint8)
                        #     folder = f'/model/liuyidi/VAE/UHD-allinone/vis/level_{j}'
                        #     if not os.path.exists(folder):
                        #         os.makedirs(folder)
                        #     cv2.imwrite(f'/model/liuyidi/VAE/UHD-allinone/vis/level_{j}/ft_{i}_{j}.png', lq_i*255)
                        #     cv2.imwrite(f'/model/liuyidi/VAE/UHD-allinone/vis/level_{j}/ft_{i}_{j}_gt.png', gt_i*255)
                        #     print('lq', lq.shape)

            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(lq.shape[-1], lq.shape[0]*torch.cuda.device_count())) 
                logger_j[bs_j] = False
             
            break
           


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)

