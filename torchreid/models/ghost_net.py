"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch
"""
import torch
import torch.nn as nn
import math
from copy import deepcopy

from torchreid.components import branches
from torchreid.components.shallow_cam import ShallowCAM

import logging

__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride == 2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, 3, stride, relu=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, cfgs, width_mult=1.):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)

        self.conv1 = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

        input_channel = output_channel

        # building inverted residual blocks
        # block = GhostBottleneck
        # for k, exp_size, c, use_se, s in self.cfgs:
        #     output_channel = _make_divisible(c * width_mult, 4)
        #     hidden_channel = _make_divisible(exp_size * width_mult, 4)
        #     layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
        #     input_channel = output_channel
        # self.features = nn.Sequential(*layers)

        output_channel, self.layer1 = self._make_layer(0, 4, input_channel, width_mult)
        output_channel, self.layer2 = self._make_layer(4, 6, output_channel, width_mult)
        output_channel, self.layer3 = self._make_layer(6, 12, output_channel, width_mult)
        output_channel, self.last = self._make_layer(12, 16, output_channel, width_mult)
        input_channel = output_channel

        # building last several layers
        output_channel = _make_divisible(960 * width_mult, 4)
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.layer4 = nn.Sequential(
            self.last,
            self.squeeze,
        )
        input_channel = output_channel

        output_channel = 1280
        # self.classifier = nn.Sequential(
        #     nn.Linear(input_channel, output_channel, bias=False),
        #     nn.BatchNorm1d(output_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(output_channel, num_classes),
        # )

        self._initialize_weights()

    def _make_layer(self, start, end, input_channel, width_mult=1.):
        block = GhostBottleneck
        output_channel = 0
        layers = []
        for k, exp_size, c, use_se, s in self.cfgs[start: end]:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        return output_channel, nn.Sequential(*layers)

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.squeeze(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ghost_net_backbone(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, s         kernel size
        [3, 16, 16, 0, 1],
        [3, 48, 24, 0, 2],
        [3, 72, 24, 0, 1],
        [5, 72, 40, 1, 2],
        [5, 120, 40, 1, 1],
        [3, 240, 80, 0, 2],
        [3, 200, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, **kwargs)


class GhostNetCommonBranch(nn.Module):

    def __init__(self, owner, backbone, args):
        super().__init__()

        self.backbone1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            # backbone.maxpool,
            backbone.layer1
        )
        self.shallow_cam = ShallowCAM(args, 40)
        self.backbone2 = nn.Sequential(
            backbone.layer2,
            backbone.layer3,
        )

    def backbone_modules(self):
        return [self.backbone1, self.backbone2]

    def forward(self, x):
        x = self.backbone1(x)
        intermediate = x = self.shallow_cam(x)
        x = self.backbone2(x)

        return x, intermediate


class GhostNetDeepBranch(nn.Module):

    def __init__(self, owner, backbone, args):
        super().__init__()

        self.backbone = deepcopy(backbone.layer4)

        self.out_dim = 960

    def backbone_modules(self):
        return [self.backbone]

    def forward(self, x):
        return self.backbone(x)


class MultiBranchGhostNet(branches.MultiBranchNetwork):

    def _get_common_branch(self, backbone, args):
        return GhostNetCommonBranch(self, backbone, args)

    def _get_middle_subbranch_for(self, backbone, args, last_branch_class):
        return GhostNetDeepBranch(self, backbone, args)


def ghost_net(num_classes, args, **kw):
    backbone = ghost_net_backbone()
    return MultiBranchGhostNet(backbone, args, num_classes)


if __name__ == '__main__':
    model = ghost_net(12, args=vars(args))
    model.eval()
    print(model)
    input = torch.randn(32, 3, 224, 224)
    y = model(input)
    print(y)
