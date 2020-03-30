from __future__ import absolute_import
from __future__ import division

from torch import nn
import torch.utils.model_zoo as model_zoo
from copy import deepcopy

from torchreid.components import branches
from torchreid.components.shallow_cam import ShallowCAM

import logging

logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample  # shortcut操作
        self.stride = stride

    def forward(self, x):
        residual = x  # 获得上层的输出

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:  # 当shortcut存在的时候
            residual = self.downsample(x)  # 将上一层的输出x输入进这个downsample所拥有一些操作（卷积等），将结果赋给residual
            #  简单说，这个目的就是为了应对上下层输出输入深度不一致问题
        out += residual
        out = self.relu(out)

        return out


#  三个卷积层，1x1 3x3 1x1 用来降维，卷积处理，升维
class Bottleneck(nn.Module):
    expansion = 4  # 若我们输入深度为64，那么扩张4倍后就变为了256

    # 其目的在于使得当前块的输出深度与下一个块的输入深度保持一致
    # 而为什么是4，这是因为在设计网络的时候就规定了的
    # 我想应该可以在保证各层之间的输入输出一致的情况下修改扩张的倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 这层1*1卷积层，是为了降维，把输出深度降到与3*3卷积层的输入深度一致
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 3x3卷积操作 # 这层3*3卷积层的channels是下面_make_layer中的第二个参数规定的
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # 这层1*1卷积层，是在升维，四倍的升
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample  # shortcut信号
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, block, layers, last_stride=2):
        self.inplanes = 64  # 输入的深度或通道数
        # block:为上边的基础块BasicBlock或瓶颈块Bottleneck，它其实就是一个对象
        # layers:每个大layer中的block个数，设为blocks更好，但每一个block实际上也很是一些小layer
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 输入深度为3(正好是彩色图片的3个通道)，输出深度为64，滤波器为7*7，步长为2，填充3层，特征图缩小1/2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # inplace-选择是否进行覆盖运算，即对原值进行操作，然后将得到的值又直接复制到该值中
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 最大池化，滤波器为3*3，步长为2，填充1层，特征图又缩小1/2
        # 此时，特征图的尺寸已成为输入的1/4

        # 下面的每一个layer都是一个由多个block组成的大layer,block可以是basicblock或bottleneck
        # 第二个参数是残差块中3*3卷积层的输入输出深度，第三个参数是该层残差快的个数
        self.layer1 = self._make_layer(block, 64, layers[0])  # 特征图大小不变
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 特征图缩小1/2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 特征图缩小1/2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)  # 特征图缩小1/2

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print('Initialized model with pretrained weights from {}'.format(model_url))


class ResNetCommonBranch(nn.Module):

    def __init__(self, owner, backbone, args):
        super().__init__()

        self.backbone1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.shallow_cam = ShallowCAM(args, 256)
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


class ResNetDeepBranch(nn.Module):

    def __init__(self, owner, backbone, args):
        super().__init__()

        self.backbone = deepcopy(backbone.layer4)

        self.out_dim = 2048

    def backbone_modules(self):
        return [self.backbone]

    def forward(self, x):
        return self.backbone(x)


class ResNetMGNLikeCommonBranch(nn.Module):

    def __init__(self, owner, backbone, args):
        super().__init__()

        self.backbone1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.shallow_cam = ShallowCAM(args, 256)
        self.backbone2 = nn.Sequential(
            backbone.layer2,
            backbone.layer3[0],
        )

    def backbone_modules(self):
        return [self.backbone1, self.backbone2]

    def forward(self, x):
        x = self.backbone1(x)
        intermediate = x = self.shallow_cam(x)
        x = self.backbone2(x)

        return x, intermediate


class ResNetMGNLikeDeepBranch(nn.Module):

    def __init__(self, owner, backbone, args):
        super().__init__()

        self.backbone = nn.Sequential(
            *deepcopy(backbone.layer3[1:]),
            deepcopy(backbone.layer4)
        )
        self.out_dim = 2048

    def backbone_modules(self):
        return [self.backbone]

    def forward(self, x):
        return self.backbone(x)


class MultiBranchResNet(branches.MultiBranchNetwork):

    def _get_common_branch(self, backbone, args):
        return ResNetCommonBranch(self, backbone, args)

    def _get_middle_subbranch_for(self, backbone, args, last_branch_class):
        return ResNetDeepBranch(self, backbone, args)


class MultiBranchMGNLikeResNet(branches.MultiBranchNetwork):

    def _get_common_branch(self, backbone, args):
        return ResNetMGNLikeCommonBranch(self, backbone, args)

    def _get_middle_subbranch_for(self, backbone, args, last_branch_class):
        return ResNetMGNLikeDeepBranch(self, backbone, args)


def resnet50_backbone():
    network = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,  # Always remove down-sampling
    )
    init_pretrained_weights(network, model_urls['resnet50'])

    return network


def resnet50(num_classes, args, **kw):
    backbone = resnet50_backbone()
    return MultiBranchResNet(backbone, args, num_classes)


def resnet50_mgn_like(num_classes, args, **kw):
    backbone = resnet50_backbone()
    return MultiBranchMGNLikeResNet(backbone, args, num_classes)
