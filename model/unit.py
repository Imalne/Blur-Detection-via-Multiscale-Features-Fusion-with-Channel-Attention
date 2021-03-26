import torch.nn as nn
from functools import partial
from model.modules import *


def getUnitByType(unit_type, skips):
    if type(unit_type) != type([]):
        if unit_type == "1":
            return partial(ReUnit1, skipconnect=(skips == 1))
        elif unit_type == "3":
            return partial(ReUnit3, skipconnect=(skips == 1))
        else:
            raise RuntimeError("no such reunit type !")
    else:
        assert len(unit_type) == len(skips)
        return [getUnitByType(unit_type[i], skips[i]) for i in range(len(unit_type))]


class ReUnit1(nn.Module):
    def __init__(self, channel_list, skipconnect=False):
        super(ReUnit1, self).__init__()
        self.feature_num = len(channel_list)
        self.skipconnect = skipconnect
        fusion_layer = []
        post_proc_blocks = []
        for i in range(self.feature_num):
            fusion_layer.append(nn.ModuleList([]))
            for j in range(self.feature_num):
                if i == j:
                    fusion_layer[i].append(None)
                elif i < j:
                    fusion_layer[i].append(
                        ConvDownSample(in_channel=channel_list[i], out_channel=channel_list[j], factor_log=j - i)
                    )
                else:
                    fusion_layer[i].append(
                        nn.Sequential(
                            nn.Conv2d(
                                channel_list[i],
                                channel_list[j],
                                stride=1, kernel_size=1, padding=0, bias=False
                            ),
                            nn.BatchNorm2d(channel_list[j]),
                            nn.UpsamplingNearest2d(scale_factor=2 ** (i - j))
                        )
                    )
        self.fusion_layer = nn.ModuleList(fusion_layer)
        for i in range(self.feature_num):
            post_proc_blocks.append(nn.Sequential(nn.Conv2d(in_channels=channel_list[i], out_channels=channel_list[i],
                                                            kernel_size=1, stride=1),
                                                  nn.BatchNorm2d(channel_list[i])
                                                  )
                                    )
        self.post_proc_blocks = nn.ModuleList(post_proc_blocks)

    def forward(self, inputs):
        inputs_fused = []
        for i in range(self.feature_num):
            y = inputs[i]
            for j in range(self.feature_num):
                if i != j:
                    y = y + self.fusion_layer[j][i](inputs[j])
            if not self.skipconnect:
                fused = self.post_proc_blocks[i](y)
            else:
                fused = self.post_proc_blocks[i](y) + inputs[i]
            inputs_fused.append(fused)
        return inputs_fused


class Up2DownUnit(nn.Module):
    def __init__(self, channel_list):
        super(Up2DownUnit, self).__init__()
        down_samples = []
        up_samples = [None]
        for i in range(0, len(channel_list)):
            if i != len(channel_list) - 1:
                down_samples.append(
                    ConvDownSample(in_channel=channel_list[i], out_channel=channel_list[i + 1], factor_log=1)
                )
            if i != 0:
                up_samples.append(nn.Sequential(
                    nn.Conv2d(
                        channel_list[i],
                        channel_list[i - 1],
                        1, 1, 0, bias=False
                    ),
                    nn.BatchNorm2d(channel_list[i - 1]),
                    nn.UpsamplingNearest2d(scale_factor=2,)
                ))
        down_samples.append(None)
        self.down_samples = nn.ModuleList(down_samples)
        self.up_samples = nn.ModuleList(up_samples)

    def forward(self, inputs):
        input_fused = []
        for i in range(len(inputs)):
            y = inputs[i]
            if i != len(inputs) - 1:
                y = y + self.up_samples[i + 1](inputs[i + 1])
            if i != 0:
                y = y + self.down_samples[i - 1](inputs[i - 1])
            input_fused.append(y)
        return input_fused


class Down2UpUnit(nn.Module):
    def __init__(self, channel_list):
        super(Down2UpUnit, self).__init__()
        down_samples = []
        up_samples = [None]
        for i in range(0, len(channel_list)):
            if i != len(channel_list) - 1:
                down_samples.append(
                    ConvDownSample(in_channel=channel_list[i], out_channel=channel_list[i + 1], factor_log=1)
                )
            if i != 0:
                up_samples.append(nn.Sequential(
                    nn.Conv2d(
                        channel_list[i],
                        channel_list[i - 1],
                        1, 1, 0, bias=False
                    ),
                    nn.BatchNorm2d(channel_list[i - 1]),
                    nn.UpsamplingNearest2d(scale_factor=2)
                ))
        down_samples.append(None)
        self.down_samples = nn.ModuleList(down_samples)
        self.up_samples = nn.ModuleList(up_samples)

    def forward(self, inputs):
        input_fused = []
        for i in range(len(inputs) - 1, -1, -1):
            y = inputs[i]
            if i != len(inputs) - 1:
                y = y + self.up_samples[i + 1](input_fused[-1])
            if i != 0:
                y = y + self.down_samples[i - 1](inputs[i - 1])
            input_fused.append(y)
        input_fused.reverse()
        return input_fused


class ReUnit3(nn.Module):
    def __init__(self, channel_list, skipconnect):
        super(ReUnit3, self).__init__()
        self.feature_num = len(channel_list)
        self.skipconnect = skipconnect
        self.up_stream = Up2DownUnit(channel_list)
        self.down_stream = Down2UpUnit(channel_list)
        post_proc_blocks = []
        for i in range(self.feature_num):
            post_proc_blocks.append(nn.Sequential(nn.Conv2d(in_channels=channel_list[i], out_channels=channel_list[i],
                                                            kernel_size=1, stride=1),
                                                  nn.BatchNorm2d(channel_list[i])
                                                  )
                                    )
        self.post_proc_blocks = nn.ModuleList(post_proc_blocks)

    def forward(self, inputs):
        input_fused_1 = self.up_stream(inputs)
        input_fused_2 = self.down_stream(inputs)
        inputs_fused = []
        for i in range(len(inputs)):
            if not self.skipconnect:
                fused = self.post_proc_blocks[i](input_fused_1[i] + input_fused_2[i])
            else:
                fused = self.post_proc_blocks[i](input_fused_1[i] + input_fused_2[i]) + inputs[i]
            inputs_fused.append(fused)
        return inputs_fused