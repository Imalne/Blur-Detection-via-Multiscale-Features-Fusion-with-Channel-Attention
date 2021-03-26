import torch
from torch.nn import CrossEntropyLoss
from loss.CE import *


def getLoss(opt):
    if opt.loss_type == "MS_CE":
        return Multi_MultiScaleCrossEntropyLoss(opt)
    elif opt.loss_type == "MSP_CE":
        return MultiProgressive_MultiScaleCrossEntropyLoss(opt)
    else:
        pass


class MLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MLoss, self).__init__()
        self.main_loss = getLoss(opt)

    def forward(self, pred, tar, mask, model):
        main_loss = self.main_loss(pred, tar, mask)
        return main_loss, [("main_loss", main_loss)]
