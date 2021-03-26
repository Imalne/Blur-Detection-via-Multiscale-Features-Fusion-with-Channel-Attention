import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19_bn, resnet101, resnet18
from model.resnet import resnet50, resnet34
from model.se_resnet import se_resnet18, se_resnet34
from model.modules import *
from pretrainedmodels.models.resnext import resnext101_32x4d


channel_lists = {
    "vgg_19": [64, 128, 256, 512, 512],
    "vgg_16": [64, 128, 256, 512, 512],
    "resnet_18": [64, 64, 128, 256, 512],
    "resnet_34": [64, 64, 128, 256, 512],
    "resnet_50": [64, 256, 512, 1024, 2048],
    "resnet_101": [64, 256, 512, 1024, 2048],
    "resnext_101": [64, 256, 512, 1024, 2048],
    "seresnet_18": [64, 64, 128, 256, 512],
    "seresnet_34": [64, 64, 128, 256, 512],
}

def getencoder(cfg):
    if cfg == "vgg19_bn":
        return VGG19Encoder(), channel_lists["vgg_19"]
    elif cfg == "vgg_16":
        return VGG16Encoder(), channel_lists["vgg_16"]
    elif cfg == "resnet_50":
        return ResNet50Encoder(), channel_lists["resnet_50"]
    elif cfg == "resnet_50_dilate":
        return ResNet50Encoder_large(), channel_lists["resnet_50"]
    elif cfg == "resnet_34":
        return ResNet34Encoder(), channel_lists["resnet_34"]
    elif cfg == "resnet_34_dilate":
        return ResNet34Encoder_large(), channel_lists["resnet_34"]
    elif cfg == "resnet_18":
        return ResNet18Encoder(), channel_lists["resnet_18"]
    elif cfg == "resnet_101":
        return ResNet101Encoder(), channel_lists["resnet_101"]
    elif cfg == "resnext_101":
        return ResNeXtEncoder(), channel_lists["resnext_101"]
    elif cfg == "seresnet_34":
        return SeResnet34Encoder(), channel_lists["seresnet_34"]
    elif cfg == "seresnet_18":
        return SeResnet18Encoder(), channel_lists["seresnet_18"]
    elif cfg == "seresnet_18_MI":
        return SeResNet18Encoder_MI(), channel_lists["seresnet_18"]
    elif cfg == "seresnet_18":
        return SeResnet18Encoder(), channel_lists["resnet_18"]
    elif "combine_" in cfg:
        cfgs = CombineEncoder.getEncoderTypes(cfg)
        return CombineEncoder(cfgs), CombineEncoder.getChannel_list(cfgs)
    else:
        raise RuntimeError("no such backbone type")



class VGG16Encoder(nn.Module):
    def __init__(self):
        super(VGG16Encoder, self).__init__()
        self.vgg = vgg16(pretrained=True)
        self.encoder_1 = self.vgg.features[0:5]
        self.encoder_2 = self.vgg.features[5:10]
        self.encoder_3 = self.vgg.features[10:17]
        self.encoder_4 = self.vgg.features[17:24]
        self.encoder_5 = self.vgg.features[24:]

    def forward(self, x):
        skip_1 = self.encoder_1(x)
        skip_2 = self.encoder_2(skip_1)
        skip_3 = self.encoder_3(skip_2)
        skip_4 = self.encoder_4(skip_3)
        x = self.encoder_5(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x


class VGG19Encoder(nn.Module):
    def __init__(self):
        super(VGG19Encoder, self).__init__()
        self.vgg = vgg19_bn(pretrained=True)
        self.encoder_1 = self.vgg.features[0:5]
        self.encoder_2 = self.vgg.features[5:12]
        self.encoder_3 = self.vgg.features[12:25]
        self.encoder_4 = self.vgg.features[25:38]
        self.encoder_5 = self.vgg.features[38:-1]

    def forward(self, x):
        skip_1 = self.encoder_1(x)
        skip_2 = self.encoder_2(skip_1)
        skip_3 = self.encoder_3(skip_2)
        skip_4 = self.encoder_4(skip_3)
        x = self.encoder_5(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x


class ResNeXtEncoder(nn.Module):
    def __init__(self):
        super(ResNeXtEncoder, self).__init__()
        self.resnext = resnext101_32x4d()
        self.encoder_1 = self.resnext.features[0:3]
        self.encoder_2 = self.resnext.features[3:5]
        self.encoder_3 = self.resnext.features[5]
        self.encoder_4 = self.resnext.features[6]
        self.encoder_5 = self.resnext.features[7]

    def forward(self, x):
        skip_1 = self.encoder_1(x)
        skip_2 = self.encoder_2(skip_1)
        skip_3 = self.encoder_3(skip_2)
        skip_4 = self.encoder_4(skip_3)
        x = self.encoder_5(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x


class ResNet50Encoder(nn.Module):
    def __init__(self, pretrain=True):
        super(ResNet50Encoder, self).__init__()
        self.resnet = resnet50(pretrained=pretrain)
        self.inputLayer = nn.Sequential(self.resnet.conv1,
                                        self.resnet.bn1,
                                        )
        self.encoder_1 = nn.Sequential(self.resnet.relu,
                                       self.resnet.maxpool,
                                       self.resnet.layer1)
        self.encoder_2 = self.resnet.layer2
        self.encoder_3 = self.resnet.layer3
        self.encoder_4 = self.resnet.layer4

    def forward(self, x):
        skip_1 = self.inputLayer(x)
        skip_2 = self.encoder_1(skip_1)
        skip_3 = self.encoder_2(skip_2)
        skip_4 = self.encoder_3(skip_3)
        x = self.encoder_4(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x


class ResNet34Encoder(nn.Module):
    def __init__(self):
        super(ResNet34Encoder, self).__init__()
        self.resnet = resnet34(True)
        self.inputLayer = nn.Sequential(self.resnet.conv1,
                                        self.resnet.bn1,
                                        self.resnet.relu
                                        )
        self.encoder_1 = nn.Sequential(self.resnet.maxpool,
                                       self.resnet.layer1)
        self.encoder_2 = self.resnet.layer2
        self.encoder_3 = self.resnet.layer3
        self.encoder_4 = self.resnet.layer4

    def forward(self, x):
        skip_1 = self.inputLayer(x)
        skip_2 = self.encoder_1(skip_1)
        skip_3 = self.encoder_2(skip_2)
        skip_4 = self.encoder_3(skip_3)
        x = self.encoder_4(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x

class ResNet50Encoder_large(nn.Module):
    def __init__(self):
        super(ResNet50Encoder_large, self).__init__()
        self.resnet = resnet50(pretrained=True, replace_stride_with_dilation=[True,True,True])
        self.inputLayer = nn.Sequential(self.resnet.conv1,
                                        self.resnet.bn1,
                                        self.resnet.relu
                                        )
        self.encoder_1 = nn.Sequential(self.resnet.maxpool,
                                       self.resnet.layer1)
        self.encoder_2 = self.resnet.layer2
        self.encoder_3 = self.resnet.layer3
        self.encoder_4 = self.resnet.layer4

    def forward(self, x):
        skip_1 = self.inputLayer(x)
        skip_2 = self.encoder_1(skip_1)
        skip_3 = self.encoder_2(skip_2)
        skip_4 = self.encoder_3(skip_3)
        x = self.encoder_4(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x

class ResNet34Encoder_large(nn.Module):
    def __init__(self):
        super(ResNet34Encoder_large, self).__init__()
        self.resnet = resnet34(pretrained=True, replace_stride_with_dilation=[True,True,True])
        self.inputLayer = nn.Sequential(self.resnet.conv1,
                                        self.resnet.bn1,
                                        self.resnet.relu
                                        )
        self.encoder_1 = nn.Sequential(self.resnet.maxpool,
                                       self.resnet.layer1)
        self.encoder_2 = self.resnet.layer2
        self.encoder_3 = self.resnet.layer3
        self.encoder_4 = self.resnet.layer4

    def forward(self, x):
        skip_1 = self.inputLayer(x)
        skip_2 = self.encoder_1(skip_1)
        skip_3 = self.encoder_2(skip_2)
        skip_4 = self.encoder_3(skip_3)
        x = self.encoder_4(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        self.resnet = resnet18(True)
        self.inputLayer = nn.Sequential(self.resnet.conv1,
                                        self.resnet.bn1,
                                        self.resnet.relu
                                        )
        self.encoder_1 = nn.Sequential(self.resnet.maxpool,
                                       self.resnet.layer1)
        self.encoder_2 = self.resnet.layer2
        self.encoder_3 = self.resnet.layer3
        self.encoder_4 = self.resnet.layer4

    def forward(self, x):
        skip_1 = self.inputLayer(x)
        skip_2 = self.encoder_1(skip_1)
        skip_3 = self.encoder_2(skip_2)
        skip_4 = self.encoder_3(skip_3)
        x = self.encoder_4(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x


class ResNet101Encoder(nn.Module):
    def __init__(self):
        super(ResNet101Encoder, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.inputLayer = nn.Sequential(self.resnet.conv1,
                                        self.resnet.bn1,

                                        )
        self.encoder_1 = nn.Sequential(self.resnet.relu,
                                       self.resnet.maxpool,
                                       self.resnet.layer1)
        self.encoder_2 = self.resnet.layer2
        self.encoder_3 = self.resnet.layer3
        self.encoder_4 = self.resnet.layer4

    def forward(self, x):
        skip_1 = self.inputLayer(x)
        skip_2 = self.encoder_1(skip_1)
        skip_3 = self.encoder_2(skip_2)
        skip_4 = self.encoder_3(skip_3)
        x = self.encoder_4(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x


class CombineEncoder(nn.Module):
    def __init__(self, encoder_types):
        super(CombineEncoder, self).__init__()
        self.encoders = nn.ModuleList([getencoder(i)[0] for i in encoder_types])

    def forward(self, x):
        combines = self.encoders[0](x)
        for i in range(1, len(self.encoders)):
            feas = self.encoders[i](x)
            combines = [torch.cat((combines[i], feas[i]), dim=1) for i in range(len(combines))]
        return combines

    @classmethod
    def getEncoderTypes(cls, cfg):
        cfgs = cfg[8:].split('+')
        return cfgs

    @classmethod
    def getChannel_list(cls, cfgs):
        channel_list = channel_lists[cfgs[0]]
        for cfg in cfgs[1:]:
            channel_list = [channel_list[i] + channel_lists[cfg][i] for i in range(len(channel_list))]
        return channel_list


class SeResnet18Encoder(nn.Module):
    def __init__(self):
        super(SeResnet18Encoder, self).__init__()
        self.resnet = se_resnet18(pretrained=True)
        self.inputLayer = nn.Sequential(self.resnet.conv1,
                                        self.resnet.bn1,
                                        self.resnet.relu
                                        )
        self.encoder_1 = nn.Sequential(self.resnet.maxpool,
                                       self.resnet.layer1)
        self.encoder_2 = self.resnet.layer2
        self.encoder_3 = self.resnet.layer3
        self.encoder_4 = self.resnet.layer4

    def forward(self, x):
        skip_1 = self.inputLayer(x)
        skip_2 = self.encoder_1(skip_1)
        skip_3 = self.encoder_2(skip_2)
        skip_4 = self.encoder_3(skip_3)
        x = self.encoder_4(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x


class SeResNet18Encoder_MI(nn.Module):
    def __init__(self):
        super(SeResNet18Encoder_MI, self).__init__()
        self.resnet = se_resnet18(pretrained=True)
        self.inputLayer = nn.Sequential(self.resnet.conv1,
                                        self.resnet.bn1,
                                        self.resnet.relu
                                        )

        self.dilated_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channel_lists['resnet_18'][1] // 2, kernel_size=3, dilation=2),
            nn.Conv2d(in_channels=channel_lists['resnet_18'][1] // 2, out_channels=channel_lists['resnet_18'][1],
                      kernel_size=3, stride=2),
            nn.BatchNorm2d(channel_lists['resnet_18'][1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_lists['resnet_18'][1], out_channels=channel_lists['resnet_18'][1],
                      kernel_size=3, stride=1))
        self.sk_conv_1 = SKConv(features=channel_lists['resnet_18'][1], M=2, L=channel_lists['resnet_18'][1])
        self.encoder_1 = nn.Sequential(self.resnet.maxpool,
                                       self.resnet.layer1)

        self.dilated_2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channel_lists['resnet_18'][2] // 2, kernel_size=3, dilation=2),
            nn.Conv2d(in_channels=channel_lists['resnet_18'][2] // 2, out_channels=channel_lists['resnet_18'][2],
                      kernel_size=3, stride=4),
            nn.BatchNorm2d(channel_lists['resnet_18'][2]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_lists['resnet_18'][2], out_channels=channel_lists['resnet_18'][2],
                      kernel_size=3, stride=1))
        self.sk_conv_2 = SKConv(features=channel_lists['resnet_18'][2], M=2, L=channel_lists['resnet_18'][2])
        self.encoder_2 = self.resnet.layer2

        self.dilated_3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channel_lists['resnet_18'][3] // 2, kernel_size=3, dilation=2),
            nn.Conv2d(in_channels=channel_lists['resnet_18'][3] // 2, out_channels=channel_lists['resnet_18'][3],
                      kernel_size=3, stride=8),
            nn.BatchNorm2d(channel_lists['resnet_18'][3]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_lists['resnet_18'][3], out_channels=channel_lists['resnet_18'][3],
                      kernel_size=3, stride=1))
        self.sk_conv_3 = SKConv(features=channel_lists['resnet_18'][3], M=2, L=channel_lists['resnet_18'][3])
        self.encoder_3 = self.resnet.layer3

        self.encoder_4 = self.resnet.layer4

    def forward(self, x):
        skip_1 = self.inputLayer(x)
        dilated_1 = self.dilated_1(x)
        fuse_1 = self.sk_conv_1([skip_1, dilated_1])
        skip_2 = self.encoder_1(skip_1)
        skip_3 = self.encoder_2(skip_2)
        skip_4 = self.encoder_3(skip_3)
        x = self.encoder_4(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x


class SeResnet34Encoder(nn.Module):
    def __init__(self):
        super(SeResnet34Encoder, self).__init__()
        self.resnet = se_resnet34(pretrained=True)
        self.inputLayer = nn.Sequential(self.resnet.conv1,
                                        self.resnet.bn1,
                                        self.resnet.relu
                                        )
        self.encoder_1 = nn.Sequential(self.resnet.maxpool,
                                       self.resnet.layer1)
        self.encoder_2 = self.resnet.layer2
        self.encoder_3 = self.resnet.layer3
        self.encoder_4 = self.resnet.layer4

    def forward(self, x):
        skip_1 = self.inputLayer(x)
        skip_2 = self.encoder_1(skip_1)
        skip_3 = self.encoder_2(skip_2)
        skip_4 = self.encoder_3(skip_3)
        x = self.encoder_4(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x



if __name__ == '__main__':
    inp = torch.rand((1,3,224,224))
    resxnetEncoder = ResNeXtEncoder()
    out = resxnetEncoder(inp)
    exit()