# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .blocks import BeginBlock, Net1, Net2, Net3, Net4, Net5, DeConvBlock, ConvBlock
import numpy as np

class SubNetMsCatPan(nn.Module): # Single Stream

    def __init__(self, in_channels, out_channels, num_features):
        super(SubNetMsCatPan, self).__init__()
        self.net1 = Net1(in_channels + 1, out_channels, num_features, kernel_size=3)

    def forward(self, x, pan):
        x_pan = torch.cat((x, pan), dim=1)
        out = self.net1(x_pan)
        return out

class SubNetMsPan(nn.Module): # Dual Stream

    def __init__(self, in_channels, out_channels, num_features):
        super(SubNetMsPan, self).__init__()
        self.p_feature = BeginBlock(1, num_features, kernel_size=3)
        self.M_feature = BeginBlock(in_channels, num_features, kernel_size=3)
        self.net2 = Net2(num_features * 2, out_channels, num_features, kernel_size=3)

    def forward(self, x, pan):
        m_feature = self.M_feature(x)
        p_feature = self.p_feature(pan)
        m_feat_p_feat = torch.cat((m_feature, p_feature), dim=1)
        out = self.net2(m_feat_p_feat)
        return out

class SubNet(nn.Module): # Multiple Stream
    def __init__(self, in_channels, out_channels, num_features):
        super(SubNet, self).__init__()
        self.p_feature = BeginBlock(1, num_features, kernel_size=3)
        self.M_feature = BeginBlock(in_channels, num_features, kernel_size=3)
        self.net1 = Net1(in_channels + 1, out_channels, num_features, kernel_size=3)
        self.net2 = Net2(num_features * 2, out_channels, num_features, kernel_size=3)
        self.net3 = Net3(out_channels * 2, out_channels, num_features, kernel_size=3)

    def forward(self, x, pan):
        m_feature = self.M_feature(x)
        p_feature = self.p_feature(pan)
        x_pan = torch.cat((x, pan), dim=1)

        x_pan_feature = self.net1(x_pan)

        m_feat_p_feat = torch.cat((m_feature, p_feature), dim=1)
        mp_feature = self.net2(m_feat_p_feat)

        net3_in = torch.cat((x_pan_feature, mp_feature), dim=1)
        out = self.net3(net3_in)

        return out

######## Ablation Study of Diffrent Scales #############

class MFPS(nn.Module): # SingleScale--MultiStream(SSMS)
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, upscale_factor, act_type='prelu',
                 norm_type=None):
        super(MFPS, self).__init__()
        self.p_feature = BeginBlock(1, num_features, kernel_size=3)
        self.M_feature = BeginBlock(in_channels, num_features, kernel_size=3)
        self.net1 = Net1(in_channels + 1, out_channels, num_features, kernel_size=3)
        self.net2 = Net2(num_features * 2, out_channels, num_features, kernel_size=3)
        self.net3 = Net3(out_channels * 2, out_channels, num_features, kernel_size=3)

    def forward(self, x, pan):
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        m_feature = self.M_feature(x)
        p_feature = self.p_feature(pan)
        x_pan = torch.cat((x, pan), dim=1)
        x_pan_feature = self.net1(x_pan)
        m_feat_p_feat = torch.cat((m_feature, p_feature), dim=1)
        mp_feature = self.net2(m_feat_p_feat)

        net3_in = torch.cat((x_pan_feature, mp_feature), dim=1)
        x_feature = self.net3(net3_in)
        out = torch.add(x, x_feature) # spectral preservation
        return out

class MMF(nn.Module):  # DualScale--MultiStream(DSMS)
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, upscale_factor, act_type='prelu',
                 norm_type=None):
        super(MMF, self).__init__()
        self.net1 = SubNet(in_channels, out_channels, num_features)  ##尺度128x128
        self.upconv1 = DeConvBlock(in_channels, out_channels, kernel_size=4, padding=2, stride=2) ##2倍上采样
        self.net2 = SubNet(in_channels, out_channels, num_features)  ##尺度256x256
        self.net4 = Net4(out_channels * 2, out_channels, num_features * 4, kernel_size=3) ##拼接后降维 num_features=32

    def forward(self, x, pan):
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)# 256x256x4
        pan_down = nn.functional.interpolate(pan, scale_factor=0.5, mode='bilinear', align_corners=False) #128x128
        x_down = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False) # 128x128

        out2_1 = self.net1(x_down, pan_down) # 128
        out2_2 = self.upconv1(out2_1) # 256

        x1 = torch.cat((x, out2_2), dim=1)
        x1 = self.net4(x1)

        out = self.net2(x1, pan)
        out = torch.add (x, out) # spectral preservation
        return out, out2_1

class MMFTHREE(nn.Module): # MultiScale--MultiStream (Ours MMFN)
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, upscale_factor, act_type='prelu',
                 norm_type=None):
        super(MMFTHREE, self).__init__()
        self.net1 = SubNet(in_channels, out_channels, num_features) ##尺度256x256
        self.upconv1 = DeConvBlock(in_channels, out_channels, kernel_size=4, padding=2, stride=2) ##2倍上采样
        self.net2 = SubNet(in_channels, out_channels, num_features) #尺度128x128
        self.upconv2_1 = DeConvBlock(in_channels, out_channels, kernel_size=4, padding=2, stride=2) ##2倍上采样
        self.net3 = SubNet(in_channels, out_channels, num_features) ##尺度64x64

        self.net4 = Net4(out_channels * 2, out_channels, num_features * 4, kernel_size=3) ##拼接后降维
        self.net5 = Net4(out_channels * 2, out_channels, num_features * 4, kernel_size=3) ##拼接后降维

    def forward(self, x, pan):

        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)# 256x256x4
        x_down = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False) # 128x128
        x_down2 = nn.functional.interpolate(x_down, scale_factor=0.5, mode='bilinear', align_corners=False) # 64x64

        pan_down = nn.functional.interpolate(pan, scale_factor=0.5, mode='bilinear', align_corners=False) # 128x128
        pan_down2 = nn.functional.interpolate(pan_down, scale_factor=0.5, mode='bilinear', align_corners=False) # 64x64

        out1_1 = self.net3(x_down2, pan_down2) # 64
        out1_2 = self.upconv2_1(out1_1) # 128 上采样

        out1_3 = self.net4(torch.cat((x_down, out1_2), dim=1)) # 128x128x4 拼接方式降维

        out2_1 = self.net1(out1_3, pan_down) # 128
        out2_2 = self.upconv1(out2_1) # 256

        x = self.net5(torch.cat((x, out2_2), dim=1)) # 256x256x4 拼接方式降维
        out3 = self.net2(x, pan) #256
        out = torch.add(x, out3)  # spectral preservation
        return out, out2_1, out1_1


######## Ablation Study of Diffrent Streams #############

class MMFMsPan(nn.Module): # MultiScale--DualStream(MSDS)
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, upscale_factor, act_type='prelu',
                 norm_type=None):
        super(MMFMsPan, self).__init__()
        self.net1 = SubNetMsPan(in_channels, out_channels, num_features) ##尺度256x256
        self.upconv1 = DeConvBlock(in_channels, out_channels, kernel_size=4, padding=2, stride=2) ##2倍上采样
        self.net2 = SubNetMsPan(in_channels, out_channels, num_features) #尺度128x128
        self.upconv2_1 = DeConvBlock(in_channels, out_channels, kernel_size=4, padding=2, stride=2) ##2倍上采样
        self.net3 = SubNetMsPan(in_channels, out_channels, num_features) ##尺度64x64

        self.net4 = Net4(out_channels * 2, out_channels, num_features * 4, kernel_size=3) ##拼接后降维
        self.net5 = Net4(out_channels * 2, out_channels, num_features * 4, kernel_size=3) ##拼接后降维

    def forward(self, x, pan):

        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)# 256x256x4
        x_down = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False) # 128x128
        x_down2 = nn.functional.interpolate(x_down, scale_factor=0.5, mode='bilinear', align_corners=False) # 64x64

        pan_down = nn.functional.interpolate(pan, scale_factor=0.5, mode='bilinear', align_corners=False) # 128x128
        pan_down2 = nn.functional.interpolate(pan_down, scale_factor=0.5, mode='bilinear', align_corners=False) # 64x64

        out1_1 = self.net3(x_down2, pan_down2) # 64
        out1_2 = self.upconv2_1(out1_1) # 128 上采样

        out1_3 = self.net4(torch.cat((x_down, out1_2), dim=1)) # 128x128x4 拼接方式降维

        out2_1 = self.net1(out1_3, pan_down) # 128
        out2_2 = self.upconv1(out2_1) # 256

        x = self.net5(torch.cat((x, out2_2), dim=1)) # 256x256x4 拼接方式降维

        out3 = self.net2(x, pan) #256

        out = torch.add(x, out3)  # spectral preservation
        return out, out2_1, out1_1


class MMFMsCatPan(nn.Module): # MultiScale--SingleStream(MSSS)
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, upscale_factor, act_type='prelu',
                 norm_type=None):
        super(MMFMsCatPan, self).__init__()
        self.net1 = SubNetMsCatPan(in_channels, out_channels, num_features) ##尺度256x256
        self.upconv1 = DeConvBlock(in_channels, out_channels, kernel_size=4, padding=2, stride=2) ##2倍上采样
        self.net2 = SubNetMsCatPan(in_channels, out_channels, num_features) #尺度128x128
        self.upconv2_1 = DeConvBlock(in_channels, out_channels, kernel_size=4, padding=2, stride=2) ##2倍上采样
        self.net3 = SubNetMsCatPan(in_channels, out_channels, num_features) ##尺度64x64

        self.net4 = Net4(out_channels * 2, out_channels, num_features * 4, kernel_size=3) ##拼接后降维
        self.net5 = Net4(out_channels * 2, out_channels, num_features * 4, kernel_size=3) ##拼接后降维

    def forward(self, x, pan):

        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)# 256x256x4
        x_down = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False) # 128x128
        x_down2 = nn.functional.interpolate(x_down, scale_factor=0.5, mode='bilinear', align_corners=False) # 64x64

        pan_down = nn.functional.interpolate(pan, scale_factor=0.5, mode='bilinear', align_corners=False) # 128x128
        pan_down2 = nn.functional.interpolate(pan_down, scale_factor=0.5, mode='bilinear', align_corners=False) # 64x64

        out1_1 = self.net3(x_down2, pan_down2) # 64
        out1_2 = self.upconv2_1(out1_1) # 128 上采样

        out1_3 = self.net4(torch.cat((x_down, out1_2), dim=1)) # 128x128x4 拼接方式降维

        out2_1 = self.net1(out1_3, pan_down) # 128
        out2_2 = self.upconv1(out2_1) # 256

        x = self.net5(torch.cat((x, out2_2), dim=1)) # 256x256x4 拼接方式降维

        out3 = self.net2(x, pan) #256

        out = torch.add(x, out3)  # spectral preservation
        return out, out2_1, out1_1

