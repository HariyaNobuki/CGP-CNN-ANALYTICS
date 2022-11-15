#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride, bias=False, pad=True):
        super(ConvBlock, self).__init__()
        if isinstance(kernel, tuple):
            pad_size = tuple(np.array(kernel) // 2) if pad else 0
        else:
            pad_size = kernel // 2 if pad else 0
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=pad_size, bias=bias),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class DeConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel, stride=2, padding=pad_size, output_padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class ConvBlock_last(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_last, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding=pad_size, bias=False))
        # nn.BatchNorm2d(out_size),
        # nn.Tanh())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class DeConvBlock_last(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock_last, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel, padding=pad_size, bias=False))
        # nn.BatchNorm2d(out_size),
        # nn.Tanh())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class ConvBlock_s(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride):
        super(ConvBlock_s, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=pad_size, bias=False),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class ConvBlock_sum(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_sum, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding=pad_size, bias=False),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True), )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = self.conv1(inputs1)
        in_data = [outputs, inputs2]
        # check of the channel size
        small_ch_id, large_ch_id = (0, 1) if in_data[0].size(1) < in_data[1].size(1) else (1, 0)
        offset = int(in_data[large_ch_id].size()[1] - in_data[small_ch_id].size()[1])
        if offset != 0:
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat([in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return self.relu(out)


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride):
        super(ResBlock, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=pad_size, bias=False),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_size, out_size, kernel, stride=stride, padding=pad_size, bias=False),
                                   nn.BatchNorm2d(out_size))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        x = self.conv1(inputs1)
        in_data = [x, inputs2]
        # # check of the image size
        # if (in_data[0].size(2) - in_data[1].size(2)) != 0:
        #     small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
        #     pool_num = math.floor(in_data[large_in_id].size(2) / in_data[small_in_id].size(2))
        #     for _ in range(pool_num-1):
        #         in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2, 0)

        # check of the channel size
        small_ch_id, large_ch_id = (0, 1) if in_data[0].size(1) < in_data[1].size(1) else (1, 0)
        offset = int(in_data[large_ch_id].size()[1] - in_data[small_ch_id].size()[1])
        if offset != 0:
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat([in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return self.relu(out)


class InceptionResA(nn.Module):
    def __init__(self, in_size):
        super(InceptionResA, self).__init__()
        self.in_size = in_size
        self.relu = nn.ReLU(inplace=False)
        
        self.conv_1 = nn.Sequential(
            ConvBlock(in_size, 32, kernel=1, stride=1, bias=True),
        )
        
        self.conv_2 = nn.Sequential(
            ConvBlock(in_size, 32, kernel=1, stride=1, bias=True),
            ConvBlock(32, 32, kernel=3, stride=1, bias=True),
        )
        
        self.conv_3 = nn.Sequential(
            ConvBlock(in_size, 32, kernel=1, stride=1, bias=True),
            ConvBlock(32, 48, kernel=3, stride=1, bias=True),
            ConvBlock(48, 64, kernel=3, stride=1, bias=True),
        )
        
        self.conv_last = nn.Sequential(
            ConvBlock(128, in_size, kernel=1, stride=1, bias=True),
        )
        
    def forward(self, x):
        x = self.relu(x)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        conv_out_l = self.conv_last(out) 
        out = self.relu(x + conv_out_l)
        assert(out.size() == x.size())
        return out


class InceptionResB(nn.Module):
    def __init__(self, in_size):
        super(InceptionResB, self).__init__()
        self.in_size = in_size
        self.relu = nn.ReLU(inplace=False)
        
        self.conv_1 = nn.Sequential(
            ConvBlock(in_size, 192, kernel=1, stride=1, bias=True),
        )
        
        self.conv_2 = nn.Sequential(
            ConvBlock(in_size, 128, kernel=1, stride=1, bias=True),
            ConvBlock(128, 160, kernel=(1, 7), stride=1, bias=True),
            ConvBlock(160, 192, kernel=(7, 1), stride=1, bias=True),
        )
        
        self.conv_last = nn.Sequential(
            ConvBlock(384, in_size, kernel=1, stride=1, bias=True),
        )
        
    def forward(self, x):
        x = self.relu(x)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        out = torch.cat([x1, x2], dim=1)
        conv_out_l = self.conv_last(out) 
        out = self.relu(x + conv_out_l)
        assert(out.size() == x.size())
        return out


class InceptionResC(nn.Module):
    def __init__(self, in_size):
        super(InceptionResC, self).__init__()
        self.in_size = in_size
        self.relu = nn.ReLU(inplace=False)
        
        self.conv_1 = nn.Sequential(
            ConvBlock(in_size, 192, kernel=1, stride=1, bias=True),
        )
        
        self.conv_2 = nn.Sequential(
            ConvBlock(in_size, 192, kernel=1, stride=1, bias=True),
            ConvBlock(192, 224, kernel=(1, 3), stride=1, bias=True),
            ConvBlock(224, 256, kernel=(3, 1), stride=1, bias=True),
        )
        
        self.conv_last = nn.Sequential(
            ConvBlock(448, in_size, kernel=1, stride=1, bias=True),
        )
        
    def forward(self, x):
        x = self.relu(x)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        out = torch.cat([x1, x2], dim=1)
        conv_out_l = self.conv_last(out) 
        out = self.relu(x + conv_out_l)
        assert(out.size() == x.size())
        return out


class InceptionResDiv1(nn.Module):
    def __init__(self, in_size):
        super(InceptionResDiv1, self).__init__()
        self.in_size = in_size
        self.relu = nn.ReLU(inplace=False)
        
        self.block_1 = nn.Sequential(
            nn.MaxPool2d(3, 2),
        )
        
        self.block_2 = nn.Sequential(
            ConvBlock(in_size, 256, kernel=1, stride=1),
            ConvBlock(256, 384, kernel=3, stride=2, bias=True, pad=False),
        )
        
        self.block_3 = nn.Sequential(
            ConvBlock(in_size, 256, kernel=1, stride=1),
            ConvBlock(256, 288, kernel=3, stride=2, bias=True, pad=False),
        )
        
        self.block_4 = nn.Sequential(
            ConvBlock(in_size, 256, kernel=1, stride=1),
            ConvBlock(256, 288, kernel=3, stride=1, bias=True),
            ConvBlock(288, 320, kernel=3, stride=2, bias=True, pad=False),
        )
        
        
    def forward(self, x):
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        
        return out


class InceptionResDiv2(nn.Module):
    def __init__(self, in_size):
        super(InceptionResDiv2, self).__init__()
        self.in_size = in_size
        self.relu = nn.ReLU(inplace=False)
        
        self.block_1 = nn.Sequential(
            ConvBlock(in_size, 384, kernel=3, stride=2, pad=False),
        )
        
        self.block_2 = nn.Sequential(
            ConvBlock(in_size, 256, kernel=1, stride=1),
            ConvBlock(256, 256, kernel=3, stride=1, bias=True),
            ConvBlock(256, 384, kernel=3, stride=2, bias=True, pad=False),
        )
        
        self.block_3 = nn.Sequential(
            nn.MaxPool2d(3, 2),
        )        
        
    def forward(self, x):
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        
        return out


class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, inputs1, inputs2):
        in_data = [inputs1, inputs2]
        # check of the image size
        if (in_data[0].size(2) - in_data[1].size(2)) != 0:
            small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
            pool_num = math.floor(math.log(in_data[large_in_id].size(2) / in_data[small_in_id].size(2)) / math.log(2))
            for _ in range(pool_num):
                in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2, 0)
            if (in_data[large_in_id].size(2) - in_data[small_in_id].size(2)):
                pad_size = math.ceil((2 * in_data[small_in_id].size(2) - 
                    in_data[large_in_id].size(2)) / 2)
                if 2 * pad_size <= in_data[small_in_id].size(2) * 0.5:
                    in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2, pad_size)
                else:
                    large_mp = in_data[large_in_id].size(2) // 2
                    small_hf_1 = in_data[small_in_id].size(2) // 2
                    small_hf_2 = in_data[small_in_id].size(2) - small_hf_1
                    in_data[large_in_id] = in_data[large_in_id][:, :, 
                        large_mp - small_hf_1:large_mp + small_hf_2,
                        large_mp - small_hf_1:large_mp + small_hf_2]
        # check of the channel size
        small_ch_id, large_ch_id = (0, 1) if in_data[0].size(1) < in_data[1].size(1) else (1, 0)
        offset = int(in_data[large_ch_id].size()[1] - in_data[small_ch_id].size()[1])
        if offset != 0:
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat([in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return out


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, inputs1, inputs2):
        in_data = [inputs1, inputs2]
        # check of the image size
        if (in_data[0].size(2) - in_data[1].size(2)) != 0:
            small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
            pool_num = math.floor(math.log(in_data[large_in_id].size(2) / in_data[small_in_id].size(2)) / math.log(2))
            for _ in range(pool_num):
                in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2, 0)
            if (in_data[large_in_id].size(2) - in_data[small_in_id].size(2)):
                pad_size = math.ceil((2 * in_data[small_in_id].size(2) - 
                    in_data[large_in_id].size(2)) / 2)
                if 2 * pad_size <= in_data[small_in_id].size(2) * 0.5:
                    in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2, pad_size)
                else:
                    large_mp = in_data[large_in_id].size(2) // 2
                    small_hf_1 = in_data[small_in_id].size(2) // 2
                    small_hf_2 = in_data[small_in_id].size(2) - small_hf_1
                    in_data[large_in_id] = in_data[large_in_id][:, :, 
                        large_mp - small_hf_1:large_mp + small_hf_2,
                        large_mp - small_hf_1:large_mp + small_hf_2]
        return torch.cat([in_data[0], in_data[1]], 1)


class DeConvBlock_sum(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock_sum, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel, stride=2, padding=pad_size, output_padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True), )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs1 = self.conv1(inputs1)
        offset = outputs1.size()[2] - inputs2.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs2 = F.pad(inputs2, padding)
        out = torch.add(outputs1, outputs2)
        return self.relu(out)


class CGP2CNN(nn.Module):   # torchと言いたいっす
    def __init__(self, cgp, in_channel, n_class, imgSize):      # レイヤーを定義する
        super(CGP2CNN, self).__init__()
        self.cgp = cgp
        self.pool_size = 2
        self.arch = OrderedDict()
        self.encode = []        # encode内容の保存
        # self.channel_num = [None for _ in range(len(self.cgp))]   # この行を踏まえると500にこだわりはなさそう
        # self.size = [None for _ in range(len(self.cgp))]
        self.channel_num = [None for _ in range(500)]   # なんの500なの？
        self.size = [None for _ in range(500)]
        self.channel_num[0] = in_channel    # ３次元で表現している
        self.size[0] = imgSize
        # encoder
        i = 0       # こいつの意味は不明
        for name, in1, in2 in self.cgp:
            # if i >= 1: print(i - 1, self.size[i - 1])
            if name == 'input' in name:
                i += 1
                continue
            elif name == 'full':    # output
                # print("val", self.channel_num[in1] * self.size[in1] * self.size[in1])
                self.encode.append(nn.Linear(self.channel_num[in1] * self.size[in1] * self.size[in1], n_class))
            elif name == 'Max_Pool' or name == 'Avg_Pool':
                self.channel_num[i] = self.channel_num[in1]
                
                if self.size[in1] < 2:
                    self.size[i] = self.size[in1]
                else:
                    self.size[i] = int(self.size[in1] / 2)
                key = name.split('_')
                func = key[0]
                if func == 'Max':
                    self.encode.append(nn.MaxPool2d(2, 2))
                else:
                    self.encode.append(nn.AvgPool2d(2, 2))
            elif name == 'Concat':
                self.channel_num[i] = self.channel_num[in1] + self.channel_num[in2]
                small_in_id, large_in_id = (in1, in2) if self.size[in1] < self.size[in2] else (in2, in1)
                self.size[i] = self.size[small_in_id]
                self.encode.append(Concat())
            elif name == 'Sum':
                small_in_id, large_in_id = (in1, in2) if self.channel_num[in1] < self.channel_num[in2] else (in2, in1)
                self.channel_num[i] = self.channel_num[large_in_id]
                small_in_id, large_in_id = (in1, in2) if self.size[in1] < self.size[in2] else (in2, in1)
                self.size[i] = self.size[small_in_id]
                self.encode.append(Sum())
            else:
                key = name.split('_')
                down = key[0]
                func = key[1]
                out_size = int(key[2])
                kernel = int(key[3])
                if down == 'S':
                    if func == 'ConvBlock':
                        self.channel_num[i] = out_size
                        self.size[i] = self.size[in1]
                        self.encode.append(ConvBlock(self.channel_num[in1], out_size, kernel, stride=1))
                    elif func == 'ResBlock':
                        in_data = [out_size, self.channel_num[in1]]
                        small_in_id, large_in_id = (0, 1) if in_data[0] < in_data[1] else (1, 0)
                        self.channel_num[i] = in_data[large_in_id]
                        # small_in_id, large_in_id = (in1, in2) if self.size[in1] < self.size[in2] else (in2, in1)
                        # self.size[i] = self.size[small_in_id]
                        self.size[i] = self.size[in1]
                        self.encode.append(ResBlock(self.channel_num[in1], out_size, kernel, stride=1))
                    elif func == 'InceptionResA':
                        self.channel_num[i] = self.channel_num[in1]
                        self.size[i] = self.size[in1]
                        self.encode.append(InceptionResA(self.channel_num[in1]))
                    elif func == 'InceptionResB':
                        self.channel_num[i] = self.channel_num[in1]
                        self.size[i] = self.size[in1]
                        self.encode.append(InceptionResB(self.channel_num[in1]))
                    elif func == 'InceptionResC':
                        self.channel_num[i] = self.channel_num[in1]
                        self.size[i] = self.size[in1]
                        self.encode.append(InceptionResC(self.channel_num[in1]))
                    else:
                        raise ValueError
                        
                else:
                    if func == "InceptionResDiv1":
                        if self.size[in1] <= 2:
                            self.channel_num[i] = self.channel_num[in1]
                            self.size[i] = self.size[in1]
                        else:
                            self.channel_num[i] = 992 + self.channel_num[in1]
                            self.size[i] = int((self.size[in1] - 3) / 2 + 1)
                        self.encode.append(InceptionResDiv1(self.channel_num[in1]))
                    elif func == "InceptionResDiv2":
                        if self.size[in1] <= 2:
                            self.channel_num[i] = self.channel_num[in1]
                            self.size[i] = self.size[in1]
                        else:
                            self.channel_num[i] = 768 + self.channel_num[in1]
                            self.size[i] = int((self.size[in1] - 3) / 2 + 1)
                        self.encode.append(InceptionResDiv2(self.channel_num[in1]))
                        
            i += 1 

        self.layer_module = nn.ModuleList(self.encode)  # イテレーターを作成してくれる（forward文を簡単にかけるようになる効果）
        self.outputs = [None for _ in range(len(self.cgp))] # 各層の出力については配列で管理するつもりなのか？

    def main(self, x):
        outputs = self.outputs
        outputs[0] = x  # input image
        nodeID = 1
        for layer in self.layer_module:
            if isinstance(layer, ConvBlock):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, ResBlock):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, InceptionResA):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, InceptionResB):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, InceptionResC):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, torch.nn.modules.linear.Linear):
                tmp = outputs[self.cgp[nodeID][1]].view(outputs[self.cgp[nodeID][1]].size(0), -1)
                outputs[nodeID] = layer(tmp)
            elif isinstance(layer, torch.nn.modules.pooling.MaxPool2d) or isinstance(layer, torch.nn.modules.pooling.AvgPool2d):
                if outputs[self.cgp[nodeID][1]].size(2) > 1:
                    outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
                else:
                    outputs[nodeID] = outputs[self.cgp[nodeID][1]]
            elif isinstance(layer, InceptionResDiv1) or isinstance(layer, InceptionResDiv2):
                if outputs[self.cgp[nodeID][1]].size(2) > 2:
                    outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
                else:
                    outputs[nodeID] = outputs[self.cgp[nodeID][1]]
            elif isinstance(layer, Concat):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]])
            elif isinstance(layer, Sum):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]])
            else:
                sys.exit("Error at CGP2CNN forward")
            nodeID += 1
        return outputs[nodeID - 1]

    def forward(self, x, t):
        return self.main(x)
