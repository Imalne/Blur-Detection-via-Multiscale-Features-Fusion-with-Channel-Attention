import torch.nn as nn
import torch
import numpy as np


class ConvDownSample(nn.Module):
    def __init__(self, in_channel, out_channel, factor_log=1):
        super(ConvDownSample, self).__init__()
        self.conv = nn.Sequential()
        for i in range(factor_log - 1):
            self.conv.add_module(name=str(i + 1), module=nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(in_channel),
                nn.ReLU()))
        self.conv.add_module(name=str(factor_log), module=nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel)))

    def forward(self, x):
        return self.conv(x)


class DeConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor=1):
        super(DeConvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,
                                         kernel_size=3 * scale_factor, stride=2 * scale_factor,
                                         padding=1 * scale_factor, output_padding=1 * scale_factor)

    def forward(self, x):
        return self.deconv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, conv_num=2):
        super(ConvBlock, self).__init__()
        assert conv_num > 0
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        for i in range(conv_num - 1):
            self.conv.add_module("conv" + str(i), nn.Sequential(
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()))

    def forward(self, x):
        return self.conv(x)


from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def dropout(length, drop_prob, training=True):
    index = []
    for i in range(length):
        if np.random.choice([0, 1], size=1, replace=False, p=[1 - drop_prob, drop_prob])[0] != 1 or not training:
            index.append(i)
        else:
            pass
    return index


class ARM(nn.Module):
    def __init__(self, channel):
        super(ARM, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x, attention_single=False):
        if attention_single:
            return x * self.attention(x), self.attention(x)
        else:
            return x * self.attention(x)

class ARM_bn(nn.Module):
    def __init__(self, channel):
        super(ARM_bn, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

    def forward(self, x, attention_single=False):
        if attention_single:
            return x * self.attention(x), self.attention(x)
        else:
            return x * self.attention(x)


class FFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FFM, self).__init__()
        # self.fusion_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU()
        # )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x, attention_single=False):
        atte = self.attention(x)
        if attention_single:
            return x * atte, atte
        else:
            return x * atte


class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x[0].shape[0]

        feats = x
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V

class MultiARM(nn.Module):
    def __init__(self, split_num, channel):
        super(MultiARM, self).__init__()
        self.split_num = split_num
        self.attention = torch.nn.ModuleList([nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        ) for i in range(split_num**2)])

    def forward(self, x, attention_single=False):
        b,c,h,w = x.shape
        h1 = h//self.split_num
        w1 = w//self.split_num

        attentions=[]
        for i in range(self.split_num):
            for j in range(self.split_num):
                attention = self.attention[i*self.split_num+j](x[:,:,i*h1:i*h1+h1,j*w1:j*w1+w1])
                attentions.append(attention)

        blocksList=[]
        for i in range(self.split_num):
            blocks=[]
            for j in range(self.split_num):
                block = x[:,:,i*h1:i*h1+h1,j*w1:j*w1+w1] * attentions[i*self.split_num+j]
                blocks.append(block)
            blocksList.append(torch.cat(blocks, 3))
        return torch.cat(blocksList, 2)

class MultiARM_Sync(nn.Module):
    def __init__(self, split_num, channel):
        super(MultiARM_Sync, self).__init__()
        self.split_num = split_num
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x, attention_single=False):
        b,c,h,w = x.shape
        h1 = h//self.split_num
        w1 = w//self.split_num

        attentions=[]
        for i in range(self.split_num):
            for j in range(self.split_num):
                attention = self.attention(x[:,:,i*h1:i*h1+h1,j*w1:j*w1+w1])
                attentions.append(attention)

        blocksList=[]
        for i in range(self.split_num):
            blocks=[]
            for j in range(self.split_num):
                block = x[:,:,i*h1:i*h1+h1,j*w1:j*w1+w1] * attentions[i*self.split_num+j]
                blocks.append(block)
            blocksList.append(torch.cat(blocks, 3))
        return torch.cat(blocksList, 2)

