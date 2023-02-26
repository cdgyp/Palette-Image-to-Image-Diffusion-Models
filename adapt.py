from .core.base_model import BaseModel
from .core import util as Util
from .data.dataset import MaskShiftingUncroppingDataset
import os, warnings
import torch

from .core.logger import VisualWriter, InfoLogger
from .core import praser as Praser
from .data import define_dataloader
from .models import create_model, define_network, define_loss, define_metric
from .models.model import Palette

def _main_worker(gpu, ngpus_per_node, opt):
    opt['local_rank'] = opt['global_rank'] = gpu

    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader, total_dataset = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )


    model: Palette
    total_dataset: MaskShiftingUncroppingDataset
    return model, total_dataset



def prepare(class_number: int, base_config: str, batch_size: int, epoch_per_train: int, iter_per_train: int, gpu_ids:list, phase='train', debug=False):
    """准备 Palette 模型和数据集

    :param int class_number: 物品种类数
    :param str base_config: 基础配置，见 palette/config 下
    :param int batch_size: Batch 大小
    :param int epoch_per_train: 每次调用 train() 时运行的 epoch 数量
    :param int iter_per_train: 每次调用 train() 时运行的迭代次数
    :param list[int | str] gpu_ids: 可用的 GPU 列表
    :param str phase: 'train' 或 'test', defaults to 'train'
    :param bool debug: bool, defaults to True
    :return tuple[Palette, MaskShiftingUncroppingDataset]: Palette 模型、数据集

    ### Palette:

    - .train(): 进行一些训练
    - .__call__(y_with_mask): 对输入的不完整地图进行补全
      - y_with_mask: BatchSize x (ClassNumber + 1) x H x W. 通道中多出的一维对应 mask, mask[y, x]=1 表示当前位置是预测的
    
    ### MaskShiftingUncroppingDataset

    - .add_mask(ground_truth_id, mask): 向一个已经存在的真相中添加一个新的 mask。`mask` 可以包含多个遮罩，形状为 BatchSize x H x W
    - .add_ground_truth(ground_truth_id, ground_truth): 增加一个新的真相
    - .is_ground_truth(ground_truth_id)：检测一个真相是否已经存在，返回 `True` 表示该真相已经存在
    """
    assert class_number > 0
    torch.multiprocessing.set_start_method('spawn')

    args_dict = {
        'config': base_config,
        'phase': phase,
        'batch': batch_size,
        'gpu_ids': ','.join([str(id) for id in gpu_ids]),
        'debug': debug,
        'port': 114514
    }

    class Args:
        def __init__(self, args_dict: dict) -> None:
            for k, v in args_dict.items():
                setattr(self, k, v)

    opt = Praser.parse(Args(args_dict))

    opt['model']['which_networks'][0]['args']['unet']['in_channel'] = class_number * 2
    opt['model']['which_networks'][0]['args']['unet']['out_channel'] = class_number
    opt['model']['which_networks'][0]['args']['unet']['inner_channel'] = max(class_number * 2, 64)
    opt['train']['n_epoch'] = epoch_per_train
    opt['train']['n_iter'] = iter_per_train


    gpu_str = ','.join(str(x) for x in gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    opt['world_size'] = 1
    
    return _main_worker(gpu_ids[0], 1, opt)

