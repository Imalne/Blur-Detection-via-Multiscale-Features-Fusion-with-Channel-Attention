from torch.nn import CrossEntropyLoss
from torch import nn
import torch


def one_hot(inp, class_num):
    ones = torch.sparse.torch.eye(class_num)
    b, h, w = inp.shape
    flat = inp.flatten()
    flat_onehot = ones.cpu().index_select(0, flat.cpu())
    one_hot = flat_onehot.view((b, h, w, class_num))
    one_hot = one_hot.cuda() if inp.is_cuda else one_hot

    one_hot = one_hot.permute(0, 3, 1, 2)
    return one_hot


class CEwithThreshold(nn.Module):
    def __init__(self, threshold=0):
        super(CEwithThreshold, self).__init__()
        self.threshold = threshold

    def forward(self, inp, target, mask=None):
        tar_onehot = one_hot(target, 2)
        soft_int = nn.functional.softmax(input=inp, dim=1)
        delta = torch.abs(tar_onehot.float() - soft_int)[:,0,:,:]
        mask[delta <= self.threshold] = 0
        mask = mask.float() * mask.shape[0]* mask.shape[1]* mask.shape[2]/torch.sum(mask)
        loss = nn.functional.nll_loss(input=torch.log(soft_int), target=target, reduce=False)
        if mask is not None:
            loss = loss * mask.float()
        return torch.mean(loss)


class MultiScaleCrossEntropyLoss(nn.Module):
    def __init__(self, opt, threshold=0):
        super(MultiScaleCrossEntropyLoss, self).__init__()
        self.single_CE = CEwithThreshold(threshold)

    def forward(self, output, target, mask=None):
        assert len(output) <= len(target)
        length = len(output)
        loss = 0
        for i in range(length):
            loss = loss + self.single_CE(output[i], target[i], mask[i]) * (4**i)
        return loss


class Multi_MultiScaleCrossEntropyLoss(nn.Module):
    def __init__(self, opt):
        super(Multi_MultiScaleCrossEntropyLoss, self).__init__()
        assert len(opt.CE_Thre) == len(opt.loss_weights) and len(opt.edge_mask_initial) == len(opt.loss_weights)
        self.thresholds = opt.CE_Thre
        self.single_CEs = [MultiScaleCrossEntropyLoss(opt, i) for i in opt.CE_Thre]
        self.weights = opt.loss_weights
        self.using_edge = opt.edge_mask_initial

    def remove_edge(self, masks, i):
        if masks is None:
            return masks
        new_masks = [torch.clone(mask) for mask in masks]
        for j in range(len(new_masks)):
            old_mask = torch.clone(new_masks[j])
            old_mask[old_mask == 0] = 2
            old_min = torch.min(old_mask)
            if old_min == 1:
                return masks
            old_mask[new_masks[j] == 2] = 0
            crop_mask = torch.ones_like(old_mask)
            crop_mask[old_mask == 0] = 0
            new_masks[j] = ((old_mask - old_min) / (1 - old_min) * (1 - self.using_edge[i]) + self.using_edge[
                i]) * crop_mask
        return new_masks

    def forward(self, output, target, mask=None):
        assert len(output) <= len(self.weights)
        total_loss = 0
        for i in range(len(output)):
            nmask = self.remove_edge(mask, i)
            single_loss = self.single_CEs[i](output[i], target, nmask)
            total_loss = total_loss + self.weights[i] * single_loss
        return total_loss


class MultiProgressive_MultiScaleCrossEntropyLoss(nn.Module):
    def __init__(self, opt):
        super(MultiProgressive_MultiScaleCrossEntropyLoss, self).__init__()
        assert len(opt.CE_Thre) == len(opt.loss_weights)
        self.thresholds = opt.CE_Thre
        self.single_CEs = [MultiScaleCrossEntropyLoss(opt, i) for i in opt.CE_Thre]
        self.weights = opt.loss_weights
        self.using_edge = opt.edge_mask_initial

    def remove_edge(self, masks, i):
        if masks is None:
            return masks
        new_masks = [torch.clone(mask) for mask in masks]
        for j in range(len(new_masks)):
            old_mask = torch.clone(new_masks[j])
            old_mask[old_mask == 0] = 2
            old_min = torch.min(old_mask)
            old_mask[new_masks[j] == 2] = 0
            crop_mask = torch.ones_like(old_mask)
            crop_mask[old_mask == 0] = 0
            new_masks[j] = ((old_mask-old_min)/(1-old_min) *(1-self.using_edge[i]) + self.using_edge[i])*crop_mask
        return new_masks

    def sample(self, ind, output, target, mask):
        if ind == len(output)-1:
            return output[ind], target, mask

        s = min((len(output)- ind - 2) * 1, len(target)-2)
        sampled_target = target[s:]
        sampled_mask = mask[s:]
        sampled_output = [nn.functional.interpolate(j, scale_factor=1/2**s) for j in output[ind]]
        return sampled_output, sampled_target, sampled_mask

    def forward(self, output, target, mask=None):
        assert len(output) <= len(self.weights)
        total_loss = 0
        for i in range(len(output)):
            nmask = self.remove_edge(mask, i)
            sampled_output, sampled_target, sampled_mask = self.sample(i, output, target, nmask)
            single_loss = self.single_CEs[i](sampled_output, sampled_target, sampled_mask)
            total_loss = total_loss + self.weights[i] * single_loss
        return total_loss


if __name__ == '__main__':
    # input = torch.rand((1,2,224,224)).cuda()
    # target = torch.randint(0,2,(1,224,224)).cuda()
    # input_2 = torch.nn.functional.interpolate(input,scale_factor=0.5)
    # target_2 = torch.nn.functional.interpolate(target.unsqueeze(dim=1).float(),scale_factor=0.5)
    # target_2 =target_2[0].long()
    # loss = CEwithThreshold(0)
    # lo =loss(input, target)
    # lo_2 =loss(input_2, target_2)
    # print(lo)
    # print(lo_2)
    exit(0)