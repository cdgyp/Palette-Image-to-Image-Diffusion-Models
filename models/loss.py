import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .....common import goal_categories

# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)

def nonempty_points(x: torch.Tensor):
    mean = x.mean(dim=(-1, -2), keepdim=True)
    return ((x >= mean * 0.1) & (x > 0)).sum(dim=-3, keepdim=True) > 0

_presence_map:torch.Tensor = None
_mask:torch.Tensor = None
def set_mask(mask: torch.Tensor):
    global _mask
    _mask = mask.unsqueeze(dim=-3)
def set_presence_map(gt_map: torch.Tensor):
    global _presence_map
    _presence_map = nonempty_points(gt_map)
def _get_mask():
    global _mask
    assert _mask is not None
    res = _mask
    _mask = None
    return res.clamp(min=0, max=1).round()
def _get_presence_map():
    global _presence_map
    assert _presence_map is not None
    res = _presence_map
    _presence_map = None
    return res

_t_range: 'tuple[int, int]' = None
_t: int = None
def set_t_range(t_range: 'tuple[int, int]'):
    global _t_range
    _t_range = t_range
def set_t(t: int):
    global _t
    _t = t
def _get_t():
    global _t
    assert _t is not None
    res = _t
    _t = None
    return res



def focused_mse_loss(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
    len_goal = len(goal_categories)
    weight_goal = torch.full([len_goal], 0.5 / len_goal).to(output.device)
    len_non_goal = target.shape[-3] - len_goal
    weight_non_goal = torch.full([len_non_goal], 0.5 / len_non_goal).to(output.device)
    channel_weight = torch.cat([weight_goal, weight_non_goal])
    assert channel_weight.requires_grad == False and (channel_weight.sum() - 1).abs().item() < 1e-5

    if mask is None: mask = _get_mask()
    nonempty_predicted = (_get_presence_map() * mask).to(channel_weight.device) # 非空且被预测的区域
    channelwise_loss = (((output - target)**2) * nonempty_predicted).sum(dim=(-1, -2)) / (nonempty_predicted.sum(dim=(-1, -2)) + 1e-9)
    return (channel_weight * channelwise_loss).sum(dim=-1)

def _dice_coefficient(output: torch.Tensor, target: torch.Tensor):
    channel_wise = (output * target).sum(dim=[-1, -2]) / (output ** 2 + target ** 2 + 1e-9).sum(dim=[-1, -2])
    return 2 * channel_wise.mean(dim=-1)

def _dice_loss(output: torch.Tensor, target: torch.Tensor):
    return 1 - _dice_coefficient(output, target)

def focused_dice_loss(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
    if mask is None: mask = _get_mask()
    return _dice_loss(output * mask, target * mask)

def _linear_mix_ratio(t):
    return (t - _t_range[0]) / (_t_range[1] - _t_range[0] - 1)

def relay_loss(output: torch.Tensor, target: torch.Tensor):
    t = _get_t()
    r = _linear_mix_ratio(t)
    mask = _get_mask()
    return r * focused_dice_loss(output, target, mask) + (1-r) * focused_mse_loss(output, target, mask)


def mse_loss(output: torch.Tensor, target: torch.Tensor):
    return F.mse_loss(output, target)

def informative_loss():
    pass
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

