import torch
import torch.nn as nn
from model.modules import ARM, DeConvBlock, ConvBlock,SKConv, ConvDownSample, FFM, ARM_bn, MultiARM, MultiARM_Sync
from model.CBAM import CBAM


class OutLayer_4(nn.Module):
    def __init__(self, channel_list, class_num, show_attnetion=False):
        super(OutLayer_4, self).__init__()
        up_samples = [None]
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
        self.self_attention = ARM(len(channel_list) * channel_list[0])
        self.up = None
        self.up = nn.Sequential(DeConvBlock(len(channel_list) * channel_list[0], channel_list[0]),
                                ConvBlock(channel_list[0], channel_list[0]))
        self.final_up = nn.Conv2d(in_channels=channel_list[0], out_channels=class_num, kernel_size=1, stride=1)
        self.final = nn.Conv2d(in_channels=len(channel_list) * channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs, show_attention=False):
        concats = inputs[0]
        for i in range(1, len(self.up_samples)):
            concats = torch.cat((concats, self.up_samples[i](inputs[i])), dim=1)

        if not show_attention:
            fea = self.self_attention(concats)
        else:
            fea, weight = self.self_attention(concats, show_attention)
        out = self.final(fea)
        up_out = self.final_up(self.up(fea))

        if not show_attention:
            return [up_out, out]
        else:
            return [up_out, out], weight


class OutLayer_4_(nn.Module):
    def __init__(self, channel_list, class_num):
        super(OutLayer_4_, self).__init__()
        up_samples = [None]
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
        self.self_attention = ARM(len(channel_list) * channel_list[0])
        self.up = None
        self.up = nn.Sequential(DeConvBlock(len(channel_list) * channel_list[0], channel_list[0]),
                                ConvBlock(channel_list[0], channel_list[0]))
        self.final_up = nn.Conv2d(in_channels=channel_list[0], out_channels=class_num, kernel_size=1, stride=1)
        self.final = nn.Conv2d(in_channels=len(channel_list) * channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs):
        concats = inputs[0]
        for i in range(1, len(self.up_samples)):
            concats = torch.cat((concats, self.up_samples[i](inputs[i])), dim=1)

        fea = self.self_attention(concats)
        out = self.final(fea)
        up_out = nn.functional.interpolate(out, scale_factor=2, mode='bilinear')
        return [up_out, out]



class OutLayer_5(nn.Module):
    def __init__(self, channel_list, class_num):
        super(OutLayer_5, self).__init__()
        up_samples = [None]
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
        self.self_attention = SKConv(channel_list[0],M=len(channel_list))
        self.up = None
        self.up = nn.Sequential(DeConvBlock(channel_list[0], channel_list[0]))
        self.final_up = nn.Conv2d(in_channels=channel_list[0], out_channels=class_num, kernel_size=1, stride=1)
        self.final = nn.Conv2d(in_channels=channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs):
        concats = [inputs[0]]
        for i in range(1, len(self.up_samples)):
            concats.append(self.up_samples[i](inputs[i]))

        fea = self.self_attention(concats)
        out = self.final(fea)
        up_out = self.final_up(self.up(fea))
        return [up_out, out]

class OutLayer_6(nn.Module):
    def __init__(self, channel, class_num):
        super(OutLayer_6, self).__init__()
        self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2))
        self.final = nn.Conv2d(in_channels=channel, out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, fea):
        out = self.final(fea)
        up_out = self.up(out)
        return [up_out, out]

class OutLayer_7(nn.Module):
    def __init__(self, channel_list, class_num):
        super(OutLayer_7, self).__init__()
        up_samples = [None]
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
        self.self_attention = ARM(len(channel_list) * channel_list[0])
        self.final_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.final = nn.Conv2d(in_channels=len(channel_list) * channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs):
        concats = inputs[0]
        for i in range(1, len(self.up_samples)):
            concats = torch.cat((concats, self.up_samples[i](inputs[i])), dim=1)

        fea = self.self_attention(concats)
        out = self.final(fea)
        up_out = self.final_up(out)
        return [up_out, out]


class OutLayer_8(nn.Module):
    def __init__(self, channel_list, class_num, show_attnetion=False):
        super(OutLayer_8, self).__init__()
        up_samples = [None]
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
        self.self_attention = FFM(len(channel_list) * channel_list[0], len(channel_list) * channel_list[0])
        self.up = None
        self.up = nn.Sequential(DeConvBlock(len(channel_list) * channel_list[0], channel_list[0]),
                                ConvBlock(channel_list[0], channel_list[0]))
        self.final_up = nn.Conv2d(in_channels=channel_list[0], out_channels=class_num, kernel_size=1, stride=1)
        self.final = nn.Conv2d(in_channels=len(channel_list) * channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs, show_attention=False):
        concats = inputs[0]
        for i in range(1, len(self.up_samples)):
            concats = torch.cat((concats, self.up_samples[i](inputs[i])), dim=1)

        if not show_attention:
            fea = self.self_attention(concats)
        else:
            fea, weight = self.self_attention(concats, show_attention)
        out = self.final(fea)
        up_out = self.final_up(self.up(fea))

        if not show_attention:
            return [up_out, out]
        else:
            return [up_out, out], weight

class OutLayer_9(nn.Module):
    def __init__(self, channel_list, class_num, show_attnetion=False):
        super(OutLayer_9, self).__init__()
        up_samples = [None]
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
        self.self_attention = ARM_bn(len(channel_list) * channel_list[0])
        self.up = None
        self.up = nn.Sequential(DeConvBlock(len(channel_list) * channel_list[0], channel_list[0]),
                                ConvBlock(channel_list[0], channel_list[0]))
        self.final_up = nn.Conv2d(in_channels=channel_list[0], out_channels=class_num, kernel_size=1, stride=1)
        self.final = nn.Conv2d(in_channels=len(channel_list) * channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs, show_attention=False):
        concats = inputs[0]
        for i in range(1, len(self.up_samples)):
            concats = torch.cat((concats, self.up_samples[i](inputs[i])), dim=1)

        if not show_attention:
            fea = self.self_attention(concats)
        else:
            fea, weight = self.self_attention(concats, show_attention)
        out = self.final(fea)
        up_out = self.final_up(self.up(fea))

        if not show_attention:
            return [up_out, out]
        else:
            return [up_out, out], weight

class OutLayer_10(nn.Module):
    def __init__(self, channel_list, class_num, show_attnetion=False):
        super(OutLayer_10, self).__init__()
        up_samples = [None]
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
        # self.self_attention = FFM(len(channel_list) * channel_list[0], len(channel_list) * channel_list[0])
        self.up = None
        self.up = nn.Sequential(DeConvBlock(len(channel_list) * channel_list[0], channel_list[0]),
                                ConvBlock(channel_list[0], channel_list[0]))
        self.final_up = nn.Conv2d(in_channels=channel_list[0], out_channels=class_num, kernel_size=1, stride=1)
        self.final = nn.Conv2d(in_channels=len(channel_list) * channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs, show_attention=False):
        concats = inputs[0]
        for i in range(1, len(self.up_samples)):
            concats = torch.cat((concats, self.up_samples[i](inputs[i])), dim=1)

        
        out = self.final(concats)
        up_out = self.final_up(self.up(concats))

        
        return [up_out, out]
        
class OutLayer_11(nn.Module):
    def __init__(self, channel_list, class_num, show_attnetion=False):
        super(OutLayer_11, self).__init__()
        up_samples = [None]
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
        # self.self_attention = FFM(len(channel_list) * channel_list[0], len(channel_list) * channel_list[0])
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.final = nn.Conv2d(in_channels=len(channel_list) * channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs, show_attention=False):
        concats = inputs[0]
        for i in range(1, len(self.up_samples)):
            concats = torch.cat((concats, self.up_samples[i](inputs[i])), dim=1)

        
        out = self.final(concats)
        up_out = self.up(out)

        
        return [up_out, out]


class OutLayer_12(nn.Module):
    def __init__(self, channel_list, class_num, show_attnetion=False):
        super(OutLayer_12, self).__init__()
        up_samples = [None]
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
        self.self_attention = CBAM(len(channel_list) * channel_list[0])
        self.up = None
        self.up = nn.Sequential(DeConvBlock(len(channel_list) * channel_list[0], channel_list[0]),
                                ConvBlock(channel_list[0], channel_list[0]))
        self.final_up = nn.Conv2d(in_channels=channel_list[0], out_channels=class_num, kernel_size=1, stride=1)
        self.final = nn.Conv2d(in_channels=len(channel_list) * channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs, show_attention=False):
        concats = inputs[0]
        for i in range(1, len(self.up_samples)):
            concats = torch.cat((concats, self.up_samples[i](inputs[i])), dim=1)

        if not show_attention:
            fea = self.self_attention(concats)
        else:
            fea, weight = self.self_attention(concats, show_attention)
        out = self.final(fea)
        up_out = self.final_up(self.up(fea))

        if not show_attention:
            return [up_out, out]
        else:
            return [up_out, out], weight

class OutLayer_13(nn.Module):
    def __init__(self, channel_list, class_num, show_attnetion=False):
        super(OutLayer_13, self).__init__()
        up_samples = [None]
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
        self.self_attention = MultiARM(4,len(channel_list) * channel_list[0])
        self.up = None
        self.up = nn.Sequential(DeConvBlock(len(channel_list) * channel_list[0], channel_list[0]),
                                ConvBlock(channel_list[0], channel_list[0]))
        self.final_up = nn.Conv2d(in_channels=channel_list[0], out_channels=class_num, kernel_size=1, stride=1)
        self.final = nn.Conv2d(in_channels=len(channel_list) * channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs, show_attention=False):
        concats = inputs[0]
        for i in range(1, len(self.up_samples)):
            concats = torch.cat((concats, self.up_samples[i](inputs[i])), dim=1)

        if not show_attention:
            fea = self.self_attention(concats)
        else:
            fea, weight = self.self_attention(concats, show_attention)
        out = self.final(fea)
        up_out = self.final_up(self.up(fea))

        if not show_attention:
            return [up_out, out]
        else:
            return [up_out, out], weight

class OutLayer_14(nn.Module):
    def __init__(self, channel_list, class_num, show_attnetion=False):
        super(OutLayer_14, self).__init__()
        up_samples = [None]
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
        self.self_attention = MultiARM_Sync(4,len(channel_list) * channel_list[0])
        self.up = None
        self.up = nn.Sequential(DeConvBlock(len(channel_list) * channel_list[0], channel_list[0]),
                                ConvBlock(channel_list[0], channel_list[0]))
        self.final_up = nn.Conv2d(in_channels=channel_list[0], out_channels=class_num, kernel_size=1, stride=1)
        self.final = nn.Conv2d(in_channels=len(channel_list) * channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs, show_attention=False):
        concats = inputs[0]
        for i in range(1, len(self.up_samples)):
            concats = torch.cat((concats, self.up_samples[i](inputs[i])), dim=1)

        if not show_attention:
            fea = self.self_attention(concats)
        else:
            fea, weight = self.self_attention(concats, show_attention)
        out = self.final(fea)
        up_out = self.final_up(self.up(fea))

        if not show_attention:
            return [up_out, out]
        else:
            return [up_out, out], weight


class OutLayer_15(nn.Module):
    def __init__(self, channel_list, class_num, show_attnetion=False):
        super(OutLayer_15, self).__init__()
        up_samples = [None]
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
        self.self_attention = MultiARM(8,len(channel_list) * channel_list[0])
        self.up = None
        self.up = nn.Sequential(DeConvBlock(len(channel_list) * channel_list[0], channel_list[0]),
                                ConvBlock(channel_list[0], channel_list[0]))
        self.final_up = nn.Conv2d(in_channels=channel_list[0], out_channels=class_num, kernel_size=1, stride=1)
        self.final = nn.Conv2d(in_channels=len(channel_list) * channel_list[0], out_channels=class_num, kernel_size=1,
                               stride=1)

    def forward(self, inputs, show_attention=False):
        concats = inputs[0]
        for i in range(1, len(self.up_samples)):
            concats = torch.cat((concats, self.up_samples[i](inputs[i])), dim=1)

        if not show_attention:
            fea = self.self_attention(concats)
        else:
            fea, weight = self.self_attention(concats, show_attention)
        out = self.final(fea)
        up_out = self.final_up(self.up(fea))

        if not show_attention:
            return [up_out, out]
        else:
            return [up_out, out], weight

def getOutLayerByType(outlayer_type):
    if outlayer_type == "4":
        return OutLayer_4
    elif outlayer_type == "4_":
        return OutLayer_4_
    elif outlayer_type == "5":
        return OutLayer_5
    elif outlayer_type == "6":
        return OutLayer_6
    elif outlayer_type == "7":
        return OutLayer_7
    elif outlayer_type == "8":
        return OutLayer_8
    elif outlayer_type == "9":
        return OutLayer_9
    elif outlayer_type == "10":
        return OutLayer_10
    elif outlayer_type == "11":
        return OutLayer_11
    elif outlayer_type == "12":
        return OutLayer_12
    elif outlayer_type == "13":
        return OutLayer_13
    elif outlayer_type == "14":
        return OutLayer_14
    elif outlayer_type == "15":
        return OutLayer_15
    else:
        raise RuntimeWarning("no such OutLayer !")
