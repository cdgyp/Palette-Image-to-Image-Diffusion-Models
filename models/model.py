import torch
import tqdm
from ..core.base_model import BaseModel
from ..core.logger import LogTracker
from .network import ChannelAdaptor
from ....utils import fold_channel, display_channels, gibson_visualize
import copy
from .loss import set_presence_map, nonempty_points, set_mask
class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        self.adaptor:ChannelAdaptor = networks[1]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        self.adaptor = self.set_device(self.adaptor, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, list(self.netG.parameters()) + list(self.adaptor.parameters()))), **optimizers[0])
        self.optimizers.append(self.optG)
        self.resume_training() 

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        
    def set_input(self, data):
        self.path = data['path']
        self.batch_size = len(data['path'])

        ''' must use set_device in tensor '''

        self.uncompressed_gt_image = self.set_device(data.get('gt_image'))
        set_presence_map(self.uncompressed_gt_image) # 部分 loss 需要未压缩的图片
        set_mask(self.set_device(data.get('mask')))

        self.adaptor.clear_loss()
        self.gt_image = self.adaptor(self.set_device(data.get('gt_image'),), data.get('channel'), self.set_device(data.get('class_manager')))
        self.adaptor_loss = self.adaptor.get_loss().mean()
        self.mask = self.set_device(data.get('mask'))
        img, mask = self.gt_image, self.mask
        self.cond_image = self.set_device(img*(1. - mask) + mask * self.set_device(torch.randn_like(img)))
        self.mask_image = img * (1. - mask) + mask

    
    def get_current_visuals(self, phase='train'):
        dict = {
            # 'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            # 'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
            'gt_image': self.gt_image.detach()[:].float().cpu(),
            'cond_image': self.cond_image.detach()[:].float().cpu(),
        }
        if self.task in ['inpainting','uncropping']:
            print(self.mask_image.shape, self.mask.shape)
            channel_mask = self.set_device(torch.ones([1, self.mask_image.shape[1], 1, 1]))
            d = self.mask.expand(-1, channel_mask.shape[1], -1, -1) * channel_mask
            channel_mask[:, -2] = 0
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                # 'mask_image': (self.mask_image+1)/2,
                'mask_image': self.mask_image - d
            })
        if phase != 'train':
            dict.update({
                # 'output': (self.output.detach()[:].float().cpu()+1)/2
                'output': self.output.detach()[:].float().cpu()
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
        
        if self.task in ['inpainting','uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.optG.zero_grad()
            self.set_input(train_data)
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask) 
            loss_informative = self.adaptor_loss
            loss_sum = loss + self.adaptor.weight * loss_informative
            loss_sum.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            self.train_metrics.update('informative_loss', loss_informative.item())
            if self.iter // self.opt['train']['log_iter'] > (self.iter - self.batch_size) // self.opt['train']['log_iter']:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
            if self.iter // self.opt['train']['img_log_iter'] > (self.iter - self.batch_size) // self.opt['train']['img_log_iter']:
                for key, value in self.get_current_visuals().items():
                    assert isinstance(value, torch.Tensor)
                    if value.shape[1] > 3:
                        with torch.no_grad():
                            first_image = [None] * len(value)
                            if key == 'gt_image':
                                first_image = nonempty_points(self.uncompressed_gt_image)
                                first_image = first_image.expand(-1, 3, -1, -1).float()
                            value = [gibson_visualize(v, f) for v, f in zip(value, first_image)]
                            value = torch.stack(value, dim=0)
                    assert (~((0<=value) & (value<256))).sum() == 0, (value.min(), value.max())
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()

        print("iteration count:", self.iter)
        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration((self.cond_image), y_t=(self.cond_image), 
                            y_0=(self.gt_image), mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration((self.cond_image), y_t=(self.cond_image), 
                            y_0=(self.gt_image), mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                    
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met((self.gt_image), self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    assert isinstance(value, torch.Tensor)
                    if value.shape[1] > 3:
                        with torch.no_grad():
                            first_image = [None] * len(value)
                            if key == 'gt_image' or key == 'output':
                                first_image = nonempty_points(self.uncompressed_gt_image)
                                if 'output' in key:
                                    first_image = first_image * self.mask
                                first_image = first_image.expand(-1, 3, -1, -1).float()
                            value = [gibson_visualize(v, f) for v, f in zip(value, first_image)]
                            value = torch.stack(value, dim=0)
                    self.writer.add_images(key, value)
                # self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()
    def __call__(self, y_with_mask):
        """forward

        :param torch.Tensor y_with_mask: BatchSize x (ClassNumber + 1) x H x W. 通道中多出的一维对应 mask, mask[y, x]=1 表示当前位置是预测的
        :raises NotImplemented: 没有实现的场景

        这将自动切换到 evaluation mode
        """

        self.netG.eval()
        with torch.no_grad():
            mask, y = y_with_mask[:, -1], y_with_mask[:, :-1]
            y = self.adaptor(y)
            mask = mask.unsqueeze(dim=1)
            y = (1. - mask) * y + mask * torch.randn_like(y).to(mask.device)
        
            if self.opt['distributed']:
                raise NotImplemented()
            else:
                if self.task in ['uncropping']:
                    output, visuals = self.netG.restoration(y, y_t=y, y_0=y, mask=mask, sample_num=self.sample_num)
                else:
                    raise NotImplemented()
        return output

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration((self.cond_image), y_t=(self.cond_image), 
                            y_0=(self.gt_image), mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration((self.cond_image), y_t=(self.cond_image), 
                            y_0=(self.gt_image), mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                        
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met((self.gt_image), self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    assert isinstance(value, torch.Tensor)
                    if value.shape[1] > 3:
                        with torch.no_grad():
                            value = [gibson_visualize(v) for v in value]
                            value = torch.stack(value, dim=0)
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
            adaptor_label = self.adaptor.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
            adaptor_label = self.adaptor.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        self.load_network(network=self.adaptor, network_label=adaptor_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
            adaptor_label = self.adaptor.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
            adaptor_label = self.adaptor.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        self.save_network(network=self.adaptor, network_label=adaptor_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()
