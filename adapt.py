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
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
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
    phase_dataset: MaskShiftingUncroppingDataset = phase_loader.dataset
    val_dataset: MaskShiftingUncroppingDataset = val_loader.dataset
    return model, phase_dataset, val_dataset



def prepare(class_number: int, base_config: str, batch_size: int, gpu_ids:list, phase='train', debug=True):
    """准备 Palette 模型和数据集

    :param int class_number: 物品种类数
    :param str base_config: 基础配置，见 palette/config 下
    :param int batch_size: Batch 大小
    :param list[int | str] gpu_ids: 可用的 GPU 列表
    :param str phase: 'train' 或 'test', defaults to 'train'
    :param bool debug: bool, defaults to True
    :return tuple[Palette, MaskShiftingUncroppingDataset, MaskShiftingUncroppingDataset]: Palette 模型、训练数据集、验证数据集

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

    args = {
        'config': base_config,
        'phase': phase,
        'batch': batch_size,
        'gpu_ids': gpu_ids,
        'debug': debug,
        'port': 114514
    }

    opt = Praser.parse(args)

    opt['model']['which_networks']['args']['unet']['in_channel'] = class_number
    opt['model']['which_networks']['args']['unet']['inner_channel'] = max(class_number * 5, 64)


    gpu_str = ','.join(str(x) for x in gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    opt['world_size'] = 1
    
    return _main_worker(0, 1, opt)

