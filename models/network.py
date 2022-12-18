import math
import os
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm.auto import tqdm
from ..core.base_network import BaseNetwork

from ....utils import ClassManager

class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if mask is not None:
                y_t = y_0*(1.-mask) + mask*y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
            loss = self.loss_fn(mask*noise, mask*noise_hat)
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

class ChannelAdaptor(BaseNetwork):
    """降低通道数以降低计算开销
    """
    def __init__(self, in_channels: int, out_channels: int, path_to_classes_json: str, path_to_groundtruths: str=None, **kwargs) -> None:
        from .....common import goal_categories
        import json
        super().__init__(**kwargs)

        self.classes_json = path_to_classes_json
        assert os.path.isfile(self.classes_json)
        with open(self.classes_json, 'r') as fp:
            classes_json = json.load(fp)
        target_classes = [classes_json['class_id'][c] for c in goal_categories]
        print(goal_categories, target_classes)
        if not isinstance(target_classes, torch.Tensor):
            target_classes = torch.tensor(target_classes)
        self.target_classes = target_classes.flatten().type(torch.long)
        assert out_channels > len(self.target_classes)
        
        if path_to_groundtruths is None:
            path_to_groundtruths = os.path.dirname(path_to_classes_json)
        self.path_to_groundtruths = path_to_groundtruths
        self.cm: ClassManager = None
        self.kernel = torch.nn.Parameter(torch.rand((out_channels - len(self.target_classes), in_channels, 1, 1)), requires_grad=True)
        assert self.kernel.requires_grad

    def set_path(self, paths: list):
        self.cm = []
        for path in paths:
            assert isinstance(path, str)
            if path[-4:] == '.gsm':
                path = path[:-4] + '.cm'
            assert path[-3:] == '.cm', path
            try:
                cm = ClassManager.load(path, self.classes_json, device=self.kernel.device)
            except FileNotFoundError:
                path = os.path.join(self.path_to_groundtruths, path)
                cm = ClassManager.load(path, self.classes_json, device=self.kernel.device)
            self.cm.append(cm)
    
    def conv(self, x: torch.Tensor, cm: ClassManager):
        if cm is None:
            return torch.nn.functional.conv2d(x.unsqueeze(dim=0), self.kernel).squeeze(dim=0)
        x = x.to(self.kernel.device)
        local_kernel = cm.reduce_channel(self.kernel, dim=1)
        assert local_kernel.requires_grad == True or x.requires_grad == False
        assert local_kernel.shape[1] == x.shape[0], (x.shape, local_kernel.shape)
        return torch.nn.functional.conv2d(x.unsqueeze(dim=0), local_kernel).squeeze(dim=0)
    def select(self, x: torch.Tensor, cm:ClassManager):
        if cm is None:
            return x[self.target_classes]
        indices = cm.reduce(self.target_classes).clamp(min=-1)
        if (indices < 0).sum() > 0:
            x = torch.cat([x, torch.zeros_like(x)], dim=0)
        return x[indices]
    def forward_sample(self, x: torch.Tensor, channel: int, i: int):
        assert len(x.shape) == 3, len(x.shape)
        assert x[channel:].abs().sum() == 0
        x = x[:channel]
        cm = self.cm[i] if i is not None else None
        return torch.cat([self.select(x, cm), self.conv(x, cm)], dim=0)

    def forward(self, xs: torch.Tensor, channels:list=None):
        """
        `channel=None` 表示没有经过压缩和 padding
        """
        if channels is None:
            if len(xs.shape) == 4:
                xs = xs.squeeze(dim=0)
            res = self.forward_sample(xs, xs.shape[-3], None)
        else:
            assert len(xs) == len(channels)
            res = [self.forward_sample(xs[i], channels[i], i) for i in range(len(channels))]
            res = torch.stack(res, dim=0)
        self.cm = None
        return res
