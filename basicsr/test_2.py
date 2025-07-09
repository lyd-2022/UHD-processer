import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import torch
from os import path as osp
# import sys
# sys.path.append('/home/zhuqi/lyd/UHDformer-main')
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
# 其他必要的导入...

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Test epochs and log to TensorBoard.')
parser.add_argument('--tensorboard_dir', type=str, default='runs/test_epochs',
                    help='Directory for TensorBoard logs')
parser.add_argument('--last_step', type=int, default=None,
                    help='The last step to test (optional)')

# 解析命令行参数
args = parser.parse_args()


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    # print(opt)
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], epoch=0,tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
    # 根据命令行参数初始化TensorBoard的SummaryWriter
    writer = SummaryWriter(args.tensorboard_dir)

    # 其他代码...

    # 遍历文件夹中的所有文件并测试
    for filename in os.listdir(checkpoints_dir):
        if filename.endswith('.pth'):
            epoch_number = int(filename.split('_')[-1].split('.')[0])
            # 如果指定了last_step，并且当前epoch小于last_step，则跳过
            if args.last_step is not None and epoch_number < args.last_step:
                continue

            # 测试模型的代码...

            # 将测试指标写入TensorBoard
            test_metrics = test_model(model)  # 假设test_model是已定义的测试函数
            for key, value in test_metrics.items():
                writer.add_scalar(f'Test/{key}', value, epoch_number)

    # 关闭TensorBoard的SummaryWriter
    writer.close()

    # 打印完成信息
    print(f"Testing complete and results have been logged to TensorBoard at {args.tensorboard_dir}.")