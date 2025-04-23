#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Caixia Dong
# @File   : fgnet_3d.py
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)

class upsample(nn.Module):
    def __init__(self, in_ch, out_ch, factor):
        super(upsample, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0), 
                                                nn.Upsample(scale_factor=factor, mode='trilinear'),
                                                )

    def forward(self, input):
        return self.dsv(input)

def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResEncoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.relu(out)
        return out

class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
        
class TCE(nn.Module):
    def __init__(self, channel):
        super(TCE, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.relu = nn.ReLU(inplace=True)
        # self.para = torch.nn.Parameter(torch.ones((1,512,64,64), dtype = torch.float32))
        # self.adj = torch.nn.Parameter(torch.ones((512,512), dtype = torch.float32))
        self.para = torch.nn.Parameter(torch.ones((1,256,6,8,8), dtype = torch.float32))
        # self.para = torch.nn.Parameter(torch.ones((1,256,8,8,8), dtype = torch.float32))
        # self.para = torch.nn.Parameter(torch.ones((1,256,8,10,10), dtype = torch.float32))
        self.adj = torch.nn.Parameter(torch.ones((256,256), dtype = torch.float32))
             
    def forward(self, x):
        # y = torch.nn.functional.relu(self.adj)
        b, c, H, W, D = x.size()
        fea_matrix = x.view(b,c,-1)
        c_adj = self.avg_pool(x).view(b,c)

        m = torch.ones((b,c,H,W,D), dtype = torch.float32)

        for i in range(0,b):

            t1 = c_adj[i].unsqueeze(0)
            t2 = t1.t()
            c_adj_s = torch.abs(torch.abs(torch.sigmoid(t1-t2)-0.5)-0.5)*2
            c_adj_s = (c_adj_s.t() + c_adj_s)/2

            output0 = torch.mul(torch.mm(self.adj*c_adj_s,fea_matrix[i]).view(1,c,H,W,D),self.para)

            m[i] = output0

        output = torch.nn.functional.relu(m.cuda())

        return output      
class FGNet3D(nn.Module):
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(FGNet3D, self).__init__()
        self.enc_input = ResEncoder3d(channels, 16)
        self.encoder1 = ResEncoder3d(16, 32)
        self.encoder2 = ResEncoder3d(32, 64)
        self.encoder3 = ResEncoder3d(64, 128)
        self.encoder4 = ResEncoder3d(128, 256)
        self.downsample = downsample()
        self.gcn = TCE(256)
        # self.affinity_attention = AffinityAttention3d(256)  
        # self.attention_fuse = nn.Conv3d(256 * 2, 256, kernel_size=1)
        self.decoder4 = Decoder3d(256, 128)
        self.decoder3 = Decoder3d(128, 64)
        self.decoder2 = Decoder3d(64, 32)
        self.decoder1 = Decoder3d(32, 16)
        self.deconv4 = deconv(256, 128)
        self.deconv3 = deconv(128, 64)
        self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(32, 16)
        self.fuse4 = BCE(256)
        self.fuse3 = BCE(128)
        self.fuse2 = BCE(64)
        self.fuse1 = BCE(32)
        self.final = nn.Conv3d(16, classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        input_feature = self.encoder4(down4)

        # Do Attenttion operations here
        attention_fuse = self.gcn(input_feature)
        # attention = self.affinity_attention(input_feature)
        # attention_fuse = input_feature + attention

        # Do decoder operations here
        up4 = self.deconv4(attention_fuse)
        # up4 = torch.cat((enc3, up4), dim=1)
        # print(enc3.shape,up4.shape)
        up4 = self.fuse4(enc3, up4)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        # up3 = torch.cat((enc2, up3), dim=1)
        up3 = self.fuse3(enc2, up3)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        # up2 = torch.cat((enc1, up2), dim=1)
        up2 = self.fuse2(enc1, up2)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        # up1 = torch.cat((enc_input, up1), dim=1)
        up1 = self.fuse1(enc_input, up1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final    
        
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(CSNet3D_w_GDPSa_wo_GAC, self).__init__()
        self.enc_input = ResEncoder3d(channels, 16)
        self.encoder1 = ResEncoder3d(16, 32)
        self.encoder2 = ResEncoder3d(32, 64)
        self.encoder3 = ResEncoder3d(64, 128)
        self.encoder4 = ResEncoder3d(128, 256)
        self.downsample = downsample()
        self.gdp = GPD_SA(256)
        # self.affinity_attention = AffinityAttention3d(256)  
        # self.attention_fuse = nn.Conv3d(256 * 2, 256, kernel_size=1)
        self.decoder4 = Decoder3d(256, 128)
        self.decoder3 = Decoder3d(128, 64)
        self.decoder2 = Decoder3d(64, 32)
        self.decoder1 = Decoder3d(32, 16)
        self.deconv4 = deconv(256, 128)
        self.deconv3 = deconv(128, 64)
        self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(32, 16)
        # self.fuse4 = FuseModule(256)
        # self.fuse3 = FuseModule(128)
        # self.fuse2 = FuseModule(64)
        # self.fuse1 = FuseModule(32)
        self.final = nn.Conv3d(16, classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        input_feature = self.encoder4(down4)

        # Do Attenttion operations here
        attention_fuse = self.gdp(input_feature)
        # attention = self.affinity_attention(input_feature)
        # attention_fuse = input_feature + attention

        # Do decoder operations here
        up4 = self.deconv4(attention_fuse)
        up4 = torch.cat((enc3, up4), dim=1)
        # up4 = self.fuse4(enc3, up4)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        # up3 = self.fuse3(enc2, up3)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        # up2 = self.fuse2(enc1, up2)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        # up1 = self.fuse1(enc_input, up1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final
class BCE(nn.Module):
    def __init__(self, fuzzynum,channel):
        super(BCE,self).__init__()
        self.n = fuzzynum
        self.channel = channel
        self.conv1 = nn.Conv3d(self.channel,1,3,padding=1)
        self.conv2 = nn.Conv3d(1,self.channel,3,padding=1)
        self.mu = nn.Parameter(torch.randn((self.channel,self.n)))
        self.sigma = nn.Parameter(torch.randn((self.channel,self.n)))
        self.bn1 = nn.BatchNorm3d(1, affine=True)
        self.bn2 = nn.BatchNorm3d(self.channel,affine=True)

    def forward(self, x):
        x = self.conv1(x)
        tmp = torch.tensor(np.zeros((x.size()[0],x.size()[1],x.size()[2],x.size()[3],x.size()[4])),dtype = torch.float).cuda()
        for num,channel,w,h,d in itertools.product(range(x.size()[0]),range(x.size()[1]),range(x.size()[2]),range(x.size()[3]),range(x.size()[4])):
            for f in range(self.n):
                tmp[num][channel][w][h][d] -= ((x[num][channel][w][h][d]-self.mu[channel][f])/self.sigma[channel][f])**2
        fNeural = self.bn2(self.conv2(self.bn1(torch.exp(tmp))))
        return fNeural
        