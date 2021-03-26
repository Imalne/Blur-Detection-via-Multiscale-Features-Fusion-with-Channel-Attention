import torch.nn as nn
from model.modules import *
import numpy as np
from model.encoder import *


class IDEARe_MScale(nn.Module):
    def __init__(self, encoder_type, reunit_num, class_num, reunit, outlayer, dropout_possibility=[1]):
        super(IDEARe_MScale, self).__init__()
        self.encoder, channel_list = getencoder(cfg=encoder_type)
        self.drop_possibility = dropout_possibility[0]
        self.max_drop_num = len(dropout_possibility) - 1
        # assert self.max_drop_num < reunit_num
        reUnits = []
        for i in range(reunit_num):
            if len(reunit) == reunit_num:
                unit = reunit[i](channel_list)
            else:
                unit = reunit[0](channel_list)
            reUnits.append(unit)
        self.reUnits = nn.ModuleList(reUnits)
        if reunit_num == 0:
            self.reUnits = None
        self.outLayer = outlayer(channel_list, class_num)

    def dropout(self, length, drop_prob, training=True):
        index = []
        for i in range(length):
            if np.random.choice([0, 1], size=1, replace=False, p=[1 - drop_prob, drop_prob])[0] != 1 or not training:
                index.append(i)
            else:
                pass
        return index

    def forward(self, x, show_attention=False):
        if not show_attention:
            current = self.encoder(x)
            outs = [self.outLayer(current)]
            mean_out = outs[0]
            if self.reUnits is not None:
                index = self.dropout(len(self.reUnits), drop_prob=self.drop_possibility)
                for i in index:
                    current = self.reUnits[i](current)
                    outs.append(self.outLayer(current))
                    mean_out = [mean_out[i] + outs[-1][i] for i in range(len(mean_out))]
            outs.append([i/len(outs) for i in mean_out])
            return outs
        else:
            current = self.encoder(x)
            out, weight = self.outLayer(current, show_attention)
            outs = [out]
            weights = [weight]
            mean_out = outs[0]
            if self.reUnits is not None:
                index = self.dropout(len(self.reUnits), drop_prob=self.drop_possibility)
                for i in index:
                    current = self.reUnits[i](current)
                    out,weight = self.outLayer(current)
                    outs.append(out)
                    weights.append(weight)
                    mean_out = [mean_out[i] + outs[-1][i] for i in range(len(mean_out))]
            outs.append([i/len(outs) for i in mean_out])
            return outs, weights



class IDEA_MScale_SK(nn.Module):
    def __init__(self, encoder_type, class_num, outlayer):
        super(IDEA_MScale_SK, self).__init__()
        self.encoder, channel_list = getencoder(cfg=encoder_type)
        self.outLayer = outlayer(channel_list[0], class_num)
        self.sk_conv = SKConv(channel_list[0], M=len(channel_list))
        up_samples=[]
        for i in range(1, len(channel_list)):
            up_samples.append(nn.Sequential(
                nn.Conv2d(
                    channel_list[i],
                    channel_list[0],
                    1, 1, 0, bias=False
                ),
                nn.BatchNorm2d(channel_list[0]),
                nn.UpsamplingBilinear2d(scale_factor=2 ** (i))
            )
            )
        self.up_samples = nn.ModuleList(up_samples)

    def forward(self, x):
        current = self.encoder(x)
        feas = [current[0]]
        for i in range(len(self.up_samples)):
            feas.append(self.up_samples[i](current[i+1]))
        fuse = self.sk_conv(feas)

        outs = []
        for i in range(len(feas)):
            outs.append(self.outLayer(feas[i]))
        outs.append(self.outLayer(fuse))
        return outs


## SK with multi outlayers and thicker SKUnit
class IDEA_MScale_SK_MOL(nn.Module):
    def __init__(self, encoder_type, class_num, outlayer, pred_channel=128):
        super(IDEA_MScale_SK_MOL, self).__init__()
        self.encoder, channel_list = getencoder(cfg=encoder_type)
        self.outLayers = nn.ModuleList([outlayer(pred_channel, class_num) for i in range(len(channel_list) + 1)])
        self.sk_conv = SKConv(pred_channel, M=len(channel_list), L=64)
        up_samples=[]
        for i in range(0, len(channel_list)):
            up_samples.append(nn.Sequential(
                nn.Conv2d(
                    channel_list[i],
                    pred_channel,
                    1, 1, 0, bias=False
                ),
                nn.BatchNorm2d(pred_channel),
                nn.UpsamplingBilinear2d(scale_factor=2 ** (i))
            )
            )
        self.up_samples = nn.ModuleList(up_samples)

    def forward(self, x):
        current = self.encoder(x)
        feas = []
        for i in range(len(self.up_samples)):
            feas.append(self.up_samples[i](current[i]))
        fuse = self.sk_conv(feas)

        outs = []
        for i in range(len(feas)):
            outs.append(self.outLayers[i](feas[i]))
        outs.append(self.outLayers[-1](fuse))
        return outs


## SK with thicker SKUnit and fusestep
class IDEA_MScale_SK_MS(nn.Module):
    def __init__(self, encoder_type, class_num, outlayer, pred_channel=128):
        super(IDEA_MScale_SK_MS, self).__init__()
        self.encoder, channel_list = getencoder(cfg=encoder_type)
        self.outLayer = outlayer(pred_channel, class_num)
        # self.sk_conv = SKConv(channel_list[0], M=len(channel_list))
        fuse_conv = []
        for i in range(len(channel_list)-1, 0, -1):
            fuse = []
            for j in range(i):
                fuse.append(SKConv(pred_channel, M=2, L=64))
            fuse_conv.append(nn.ModuleList(fuse))
        self.fuse_conv = nn.ModuleList(fuse_conv)


        up_samples=[]
        for i in range(0, len(channel_list)):
            up_samples.append(nn.Sequential(
                nn.Conv2d(
                    channel_list[i],
                    pred_channel,
                    1, 1, 0, bias=False
                ),
                nn.BatchNorm2d(pred_channel),
                nn.UpsamplingBilinear2d(scale_factor=2 ** (i))
            )
            )
        self.up_samples = nn.ModuleList(up_samples)

    def forward(self, x):
        current = self.encoder(x)
        feas = []
        for i in range(len(self.up_samples)):
            feas.append(self.up_samples[i](current[i]))

        outs=[]
        new_feas = []

        for j in range(len(current)):
            outs.append(self.outLayer(feas[0]))
            for i in range(len(feas)-1):
                outs.append(self.outLayer(feas[i+1]))
                new_feas.append(self.fuse_conv[j][i]([feas[i],feas[i+1]]))
            feas = new_feas
            new_feas=[]
        return outs


## SK with thicker SKUnit and fusestep(few)
# class IDEA_MScale_SK_MSL(nn.Module):
#     def __init__(self, encoder_type, class_num, outlayer, pred_channel=128):
#         super(IDEA_MScale_SK_MSL, self).__init__()
#         self.encoder, channel_list = getencoder(cfg=encoder_type)
#         self.outLayer = outlayer(pred_channel, class_num)
#         # self.sk_conv = SKConv(channel_list[0], M=len(channel_list))
#         fuse_conv = []
#         for i in range((len(channel_list)+1)//2, 0, -2):
#             fuse = []
#             for j in range(i):
#                 if (j == 0 or j == i - 1) and i != 1:
#                     fuse.append(SKConv(pred_channel, M=2, L=64))
#                 else:
#                     fuse.append(SKConv(pred_channel, M=3, L=64))
#             fuse_conv.append(nn.ModuleList(fuse))
#         self.fuse_conv = nn.ModuleList(fuse_conv)
#
#
#         up_samples=[]
#         for i in range(0, len(channel_list)):
#             up_samples.append(nn.Sequential(
#                 nn.Conv2d(
#                     channel_list[i],
#                     pred_channel,
#                     1, 1, 0, bias=False
#                 ),
#                 nn.BatchNorm2d(pred_channel),
#                 nn.UpsamplingBilinear2d(scale_factor=2 ** (i))
#             )
#             )
#         self.up_samples = nn.ModuleList(up_samples)
#
#     def forward(self, x):
#         current = self.encoder(x)
#         feas = []
#         for i in range(len(self.up_samples)):
#             feas.append(self.up_samples[i](current[i]))
#
#         outs=[]
#         new_feas = []
#
#         for j in range((len(current))//2):
#             for fea in feas:
#                 outs.append(self.outLayer(fea))
#             for i in range(len(self.fuse_conv[j])):
#                 if len(self.fuse_conv[j]) == 1:
#                     new_feas.append(self.fuse_conv[j][i]([feas[0], feas[1], feas[2]]))
#                     break
#                 elif i == 0:
#                     new_feas.append(self.fuse_conv[j][i]([feas[2 * i], feas[2 * i + 1]]))
#                 elif i == len(self.fuse_conv[j]) - 1:
#                     new_feas.append(self.fuse_conv[j][i]([feas[2 * i - 1], feas[2 * i]]))
#                 else:
#                     new_feas.append(self.fuse_conv[j][i]([feas[2 * i - 1], feas[2 * i], feas[2 * i + 1]]))
#             feas = new_feas
#             new_feas=[]
#         outs.append(self.outLayer(feas[0]))
#         return outs

## SK with thicker SKUnit and fusestep(few) (multi outlayer)
class IDEA_MScale_SK_MSL(nn.Module):
    def __init__(self, encoder_type, class_num, outlayer, pred_channel=128):
        super(IDEA_MScale_SK_MSL, self).__init__()
        self.encoder, channel_list = getencoder(cfg=encoder_type)
        # self.sk_conv = SKConv(channel_list[0], M=len(channel_list))
        fuse_conv = []
        outlayers=[nn.ModuleList(outlayer(pred_channel, class_num) for j in range(len(channel_list)))]
        for i in range((len(channel_list)+1)//2, 0, -2):
            fuse = []
            outlayers.append(nn.ModuleList(outlayer(pred_channel, class_num) for j in range(i)))
            for j in range(i):
                if (j == 0 or j == i - 1) and i != 1:
                    fuse.append(SKConv(pred_channel, M=2, L=64))
                else:
                    fuse.append(SKConv(pred_channel, M=3, L=64))
            fuse_conv.append(nn.ModuleList(fuse))
        self.fuse_conv = nn.ModuleList(fuse_conv)
        self.outLayers = nn.ModuleList(outlayers)


        up_samples=[]
        for i in range(0, len(channel_list)):
            up_samples.append(nn.Sequential(
                nn.Conv2d(
                    channel_list[i],
                    pred_channel,
                    1, 1, 0, bias=False
                ),
                nn.BatchNorm2d(pred_channel),
                nn.UpsamplingBilinear2d(scale_factor=2 ** (i))
            )
            )
        self.up_samples = nn.ModuleList(up_samples)

    def forward(self, x):
        current = self.encoder(x)
        feas = []
        for i in range(len(self.up_samples)):
            feas.append(self.up_samples[i](current[i]))

        outs=[]
        new_feas = []

        for j in range((len(current))//2):
            for i in range(len(feas)):
                outs.append(self.outLayers[j][i](feas[i]))
            for i in range(len(self.fuse_conv[j])):
                if len(self.fuse_conv[j]) == 1:
                    new_feas.append(self.fuse_conv[j][i]([feas[0], feas[1], feas[2]]))
                    break
                elif i == 0:
                    new_feas.append(self.fuse_conv[j][i]([feas[2 * i], feas[2 * i + 1]]))
                elif i == len(self.fuse_conv[j]) - 1:
                    new_feas.append(self.fuse_conv[j][i]([feas[2 * i - 1], feas[2 * i]]))
                else:
                    new_feas.append(self.fuse_conv[j][i]([feas[2 * i - 1], feas[2 * i], feas[2 * i + 1]]))
            feas = new_feas
            new_feas=[]
        outs.append(self.outLayers[-1][-1](feas[0]))
        return outs