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

def _nonempty_points(x: torch.Tensor):
    mean = x.mean(dim=(-1, -2), keepdim=True)
    return ((x >= mean * 0.1) & (x > 0)).sum(dim=-3, keepdim=True) > 0

_presence_map:torch.Tensor = None
def set_presence_map(gt_map: torch.Tensor):
    global _presence_map
    _presence_map = _nonempty_points(gt_map)
def _get_presence_map():
    global _presence_map
    assert _presence_map is not None
    res = _presence_map
    _presence_map = None
    return res


def focused_mse_loss(output: torch.Tensor, target: torch.Tensor):
    len_goal = len(goal_categories)
    weight_goal = torch.full([len_goal], 0.5 / len_goal).to(output.device)
    len_non_goal = target.shape[-3] - len_goal
    weight_non_goal = torch.full([len_non_goal], 0.5 / len_non_goal).to(output.device)
    channel_weight = torch.cat([weight_goal, weight_non_goal])
    assert channel_weight.requires_grad == False and (channel_weight.sum() - 1).abs().item() < 1e-5
    nonempty = _get_presence_map().to(channel_weight.device)
    # print(nonempty.float().sum(dim=(-1, -2)).mean())
    channelwise_loss = (((output - target)**2) * nonempty).sum(dim=(-1, -2)) / (nonempty.sum(dim=(-1, -2)) + 1e-9)
    return (channel_weight * channelwise_loss).sum(dim=-1).mean()

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

