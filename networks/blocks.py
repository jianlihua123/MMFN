import torch
import torch.nn as nn
import math
from collections import OrderedDict
import sys

################
# Basic blocks
################

def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!'%act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type =='bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!'%norm_type)
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!'%pad_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict'%sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False, valid_padding=True, padding=0,
              act_type='prelu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    norm_type = None
    if mode == 'CNA': # 这个是优先选择的
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)


def DeConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False, valid_padding=True, padding=0,
              act_type='prelu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)

    if mode == 'CNA': # 这个是优先选择的
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)


class BeginBlock(nn.Module):
    def __init__(self, in_channel, out_channle, kernel_size, stride=1, valid_padding=True, padding=0, dilation=1, bias=True,
                 pad_type='zero', norm_type='bn', act_type='prelu', mode='CNA', res_scale=1):
        super(BeginBlock, self).__init__()
        self.conv0 = ConvBlock(in_channel, out_channle, kernel_size=3, padding=1)
        act_type = "prelu"
        # norm_type = None
        self.conv1 = ConvBlock(out_channle, 2 * out_channle, kernel_size=3, stride=1, dilation=dilation, bias=bias,
                            valid_padding=valid_padding,
                            padding=padding, act_type=act_type,
                            norm_type=norm_type, pad_type=pad_type, mode=mode)  # size减半
        self.conv2 = ConvBlock(2 * out_channle, out_channle, kernel_size,
                               stride, dilation, bias, valid_padding, padding,
                            act_type, norm_type, pad_type, mode)
        self.res_scale = res_scale

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        return x3


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channle, kernel_size, stride=1, valid_padding=True, padding=0, dilation=1, bias=True,
                 pad_type='zero', norm_type='bn', act_type='prelu', mode='CNA', res_scale=1):
        super(ResBlock, self).__init__()
        self.conv0 = ConvBlock(in_channel, out_channle, kernel_size=3, padding=1)
        act_type = "prelu"
        # norm_type = None
        self.conv1 = ConvBlock(out_channle, 2 * out_channle, kernel_size=3, stride=1, dilation=dilation, bias=bias,
                            valid_padding=valid_padding,
                            padding=padding, act_type=act_type,
                            norm_type=norm_type, pad_type=pad_type, mode=mode)  # size减半
        self.conv2 = ConvBlock(2 * out_channle, out_channle, kernel_size,
                               stride, dilation, bias, valid_padding, padding,
                            act_type, norm_type, pad_type, mode)
        self.res_scale = res_scale

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        out = torch.add(x, x3)
        return out


class Net1(nn.Module):
    def __init__(self, in_channel, out_channle, feature_num, kernel_size, stride=1,
                 valid_padding=True, padding=0, dilation=1, bias=True,
                 pad_type='zero', norm_type='bn', act_type='prelu', mode='CNA', res_scale=1):
        super(Net1, self).__init__()
        self.conv0 = ConvBlock(in_channel, feature_num, kernel_size=3, padding=1)
        act_type = "prelu"
        # norm_type = None
        self.conv1 = ResBlock(feature_num, feature_num, kernel_size=3, stride=1)
        self.conv2 = ConvBlock(feature_num, out_channle, kernel_size,
                               stride, dilation, bias, valid_padding, padding,
                            act_type, norm_type, pad_type, mode)
        self.res_scale = res_scale

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        return x3


class Net2(nn.Module):  # 需要做一次上采样
    def __init__(self, in_channel, out_channle, feature_num, kernel_size, stride=1,
                 valid_padding=True, padding=0, dilation=1, bias=True,
                 pad_type='zero', norm_type='bn', act_type='prelu', mode='CNA', res_scale=1):
        super(Net2, self).__init__()
        self.conv0 = ConvBlock(in_channel, feature_num, kernel_size=3, padding=1)
        act_type = "prelu"
        self.conv1 = ResBlock(feature_num, feature_num, kernel_size=3, stride=1, dilation=dilation, bias=bias,
                            valid_padding=valid_padding,
                            padding=padding, act_type=act_type,
                            norm_type=norm_type, pad_type=pad_type, mode=mode)
        # self.conv2 = DeConvBlock(2 * feature_num, feature_num, kernel_size=4, stride=2, padding=2)
        self.conv3 = ConvBlock(feature_num, out_channle, kernel_size=3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        # x3 = self.conv2(x2)
        x3 = self.conv3(x2)
        return x3


class Net3(nn.Module):  # 需要做一次上采样
    def __init__(self, in_channel, out_channle, feature_num,
                 kernel_size, stride=1, valid_padding=True, padding=0, dilation=1, bias=True,
                 pad_type='zero', norm_type='bn', act_type='prelu', mode='CNA', res_scale=1):
        super(Net3, self).__init__()
        self.conv0 = ConvBlock(in_channel, feature_num, kernel_size=3, padding=1)
        act_type = "prelu"
        self.conv1 = ResBlock(feature_num, feature_num, kernel_size=5, stride=1, dilation=dilation, bias=bias,
                            valid_padding=valid_padding,
                            padding=2, act_type=act_type,
                            norm_type=norm_type, pad_type=pad_type, mode=mode)
        self.conv2 = ConvBlock(feature_num, out_channle, kernel_size=1, padding=0)
        self.res_scale = res_scale

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        return x3

class Net4(nn.Module):
    def __init__(self, in_channel, out_channle, feature_num, kernel_size, stride=1,
                 valid_padding=True, padding=0, dilation=1, bias=True,
                 pad_type='zero', norm_type='bn', act_type='prelu', mode='CNA', res_scale=1):
        super(Net4, self).__init__()
        self.conv0 = ConvBlock(in_channel, feature_num, kernel_size=3, padding=1)
        act_type = "prelu"
        # norm_type = None
        self.conv1 = ResBlock(feature_num, feature_num, kernel_size=3, stride=1)
        self.conv2 = ConvBlock(feature_num, out_channle, kernel_size,
                               stride, dilation, bias, valid_padding, padding,
                            act_type, norm_type, pad_type, mode)
        self.res_scale = res_scale

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        return x3

class Net5(nn.Module):
    def __init__(self, in_channel, out_channle, feature_num, kernel_size, stride=1,
                 valid_padding=True, padding=0, dilation=1, bias=True,
                 pad_type='zero', norm_type='bn', act_type='prelu', mode='CNA', res_scale=1):
        super(Net5, self).__init__()
        self.conv0 = ConvBlock(in_channel, feature_num, kernel_size=3, padding=1)
        act_type = "prelu"
        # norm_type = None
        self.conv1 = ResBlock(feature_num, feature_num, kernel_size=3, stride=1)
        self.conv2 = ConvBlock(feature_num, out_channle, kernel_size,
                               stride, dilation, bias, valid_padding, padding,
                            act_type, norm_type, pad_type, mode)
        self.res_scale = res_scale

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        return x3


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

################
# Advanced blocks
################
#
# class ResBlock(nn.Module):
#     def __init__(self, in_channel, out_channle, mid_channel, kernel_size, stride=1, valid_padding=True, padding=0, dilation=1, bias=True,
#                  pad_type='zero', norm_type='bn', act_type='relu', mode='CNA', res_scale=1):
#         super(ResBlock, self).__init__()
#         conv0 = ConvBlock(in_channel, mid_channel, kernel_size, stride, dilation, bias, valid_padding, padding, act_type, norm_type, pad_type, mode)
#         act_type = "prelu"
#         norm_type = None
#         conv1 = ConvBlock(mid_channel, out_channle, kernel_size, stride, dilation, bias, valid_padding, padding, act_type, norm_type, pad_type, mode)
#         self.res = sequential(conv0, conv1)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.res(x).mul(self.res_scale)
#         return x + res


# def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0,
#                 act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
#     assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]
#
#     p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
#     deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
#
#     if mode == 'CNA':
#         act = activation(act_type) if act_type else None
#         n = norm(out_channels, norm_type) if norm_type else None
#         return sequential(p, deconv, n, act)
#     elif mode == 'NAC':
#         act = activation(act_type, inplace=False) if act_type else None
#         n = norm(in_channels, norm_type) if norm_type else None
#         return sequential(n, act, p, deconv)
################
# helper funcs
################

def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
    padding = (kernel_size-1) // 2
    return padding
