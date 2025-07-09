from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch
import torchvision.utils as tvu

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import copy

import pyiqa
from .cal_ssim import SSIM
from torch import nn


# @MODEL_REGISTRY.register()
class dwtModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.ema_decay = opt['train'].get('ema_decay', 0)
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

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.ssim = SSIM().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.filter = None

        # define metric functions 
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items():
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # load pre-trained HQ ckpt, frozen decoder and codebook 
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False)
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

            # hq_opt = self.opt['network_g'].copy()
            # hq_opt['LQ_stage'] = False
            # self.net_hq = build_network(hq_opt)
            # self.net_hq = self.model_to_device(self.net_hq)
            # self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])

            self.load_network(self.net_g, load_path, False)
            # frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None)
            # if frozen_module_keywords is not None:
            #     for name, module in self.net_g.named_modules():
            #         for fkw in frozen_module_keywords:
            #             if fkw in name:
            #                 for p in module.parameters():
            #                     p.requires_grad = False
            #                 break

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        # print('#########################################################################',load_path)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'], param_key = self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
            # self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0)
            # self.net_d_best = copy.deepcopy(self.net_d)

        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        # self.net_d = build_network(self.opt['network_d'])
        # self.net_d = self.model_to_device(self.net_d)
        # load pretrained d models
        # load_path = self.opt['path'].get('pretrain_network_d', None)
        # # print(load_path)
        # if load_path is not None:
        #     logger.info(f'Loading net_d from {load_path}')
        #     self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        # self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        # if train_opt.get('perceptual_opt'):
        #     self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        #     self.model_to_device(self.cri_perceptual)
        # else:
        #     self.cri_perceptual = None

        # if train_opt.get('gan_opt'):
        #     self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # self.net_d_iters = train_opt.get('net_d_iters', 1)
        # self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        # optim_type = train_opt['optim_d'].pop('type')
        # optim_class = getattr(torch.optim, optim_type)
        # self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        # self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        # self.lq_equalize = data['lq_equalize'].to(self.device)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def feed_data2(self, data):
        self.lq = data['lq']
        # self.lq_equalize = data['lq_equalize'].to(self.device)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.filter = None

        # if self.mixing_flag:
        #     self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)
    
    
    
    def feed_train_data2(self, data):
        self.lq = data['lq'].to(self.device)
        # self.lq.requires_grad = True
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.gt_list = data['gt_list']
        self.filter = data['filter']
        # if self.mixing_flag:
        #     self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        # for p in self.net_d.parameters():
        #     p.requires_grad = False
        self.optimizer_g.zero_grad()
        if self.filter:
            
            output = self.net_g(self.lq)
            # xll, xlh, xhl, xhh = self.gt_list[-1]
            # xll = self.filter(output, xlh, xhl, xhh)
            # self.gt_list[-1][0] = output
            
            for i in range(len(self.gt_list))[::-1]:
                _, xlh, xhl, xhh = self.gt_list[i]
                output = self.filter(output, xlh, xhl, xhh)
                # print(xll.shape)
            self.output = output
            # self.output.requires_grad = True

        else:
            self.output = self.net_g(self.lq)

        # if current_iter==0:

        l_g_total = 0
        loss_dict = OrderedDict()

        # ===================================================
        # codebook loss
        # if train_opt.get('codebook_opt', None):
        #     l_codebook *= train_opt['codebook_opt']['loss_weight']
        #     l_g_total += l_codebook.mean()
        #     loss_dict['l_codebook'] = l_codebook.mean()

        # semantic cluster loss, only for LQ stage!
        # if train_opt.get('semantic_opt', None) and isinstance(l_semantic, torch.Tensor):
        #     l_semantic *= train_opt['semantic_opt']['loss_weight']
        #     l_semantic = l_semantic.mean()
        #     l_g_total += l_semantic
        #     loss_dict['l_semantic'] = l_semantic

        # pixel loss
        # if self.cri_pix:

        l_pix = self.l1(self.output, self.gt)
        # l_pix = 1 - self.ssim(self.output, self.gt)
        l_g_total += l_pix
        loss_dict['l_pix'] = l_pix

        if train_opt.get('fft_opt', None):
            l_fft = self.cri_fft(self.output, self.gt)
            l_g_total += l_fft
            loss_dict['l_freq'] = l_fft

        l_g_total.mean().backward()

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 1024 * 1024  # use smaller min_size with limited GPU memory
        lq_input = self.lq
        
        # restoration = self.net_g(self.lq)
        _, _, h, w = lq_input.shape
        if h * w > min_size:
            self.output = net_g.test_tile(lq_input)
        else:
            self.output = net_g(lq_input)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, save_as_dir=False,dwt_levels=0,bs_j=0 ,wt_function= None,iwt_function=None):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            wavelet_list = []
            wavelet_list_gt = []
            if dwt_levels-bs_j > 0:
                lq = val_data['lq']
                gt = val_data['gt']
                gt0 = gt
                for i in range(dwt_levels-bs_j):
                    with torch.no_grad():
                        lq, xlh, xhl, xhh =  wt_function(lq.cuda())
                        wavelet_list.append([lq, xlh, xhl, xhh])

                        gt, xlh, xhl, xhh =  wt_function(gt.cuda())
                        wavelet_list_gt.append([gt, xlh, xhl, xhh])

                
                self.feed_data2({'lq':wavelet_list[-1][0], 'gt':gt0, 'gt_list':wavelet_list_gt})
            else:
                lq = val_data['lq']
                self.feed_data(val_data)
            self.test()
            if dwt_levels-bs_j > 0:
                wavelet_list_gt[-1][0] = self.output
                for i in range(len(wavelet_list_gt))[::-1]:
                    xll, xlh, xhl, xhh = wavelet_list_gt[i]
                    xll = iwt_function(xll, xlh, xhl, xhh)
                    # print(xll.shape)
                self.output = xll

            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]
            # metric_data = [self.lq, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}',
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_result = self.metric_funcs[name](*metric_data)
                    self.metric_results[name] += tmp_result.item()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()

        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)

                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    # self.copy_model(self.net_d, self.net_d_best)
                    # self.save_network(self.net_g, 'net_g_best', current_iter, epoch)
                    # self.save_network(self.net_d, 'net_d_best', current_iter, epoch)
            else:
                # update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated
                if sum(updated):
                    self.copy_model(self.net_g, self.net_g_best)
                    # self.copy_model(self.net_d, self.net_d_best)
                    # self.save_network(self.net_g, 'net_g_best', '')
                    # self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            print('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(lq.shape[-1], lq.shape[0]*torch.cuda.device_count()))

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx)
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
