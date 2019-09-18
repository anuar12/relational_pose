# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.checkpoint as checkpoint


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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


class RNOutputModel(nn.Module):
    def __init__(self, f_hid, size_out):
        super(RNOutputModel, self).__init__()
        self.size_out = size_out

        self.fc2 = nn.Linear(f_hid, f_hid)
        self.fc2_bn = nn.BatchNorm1d(f_hid)
        self.fc3 = nn.Linear(f_hid, size_out)

    def forward(self, x):
        x = self.fc2(x).permute(0, 2, 1)
        x = self.fc2_bn(x).permute(0, 2, 1)
        x = F.relu(x)
        x = self.fc3(x)
        #x = x.view(1, (self.inp_dim_size**2)*self.size_out)
        return x


class GModule(nn.Module):

    def __init__(self, f, f_hid):
        super(GModule, self).__init__()
        self.g_fc1 = nn.Linear(2*(f+2), f_hid)
        self.g_fc2 = nn.Linear(f_hid, f_hid)
        self.g_fc3 = nn.Linear(f_hid, f_hid)
        #self.g_fc4 = nn.Linear(f_hid, f_hid)
        self.g_fc1_bn = nn.BatchNorm1d(f_hid)
        self.g_fc2_bn = nn.BatchNorm1d(f_hid)
        self.g_fc3_bn = nn.BatchNorm1d(f_hid)
        #self.g_fc4_bn = nn.BatchNorm1d(f_hid)

    def forward(self, x):
        x_ = self.g_fc1(x).permute(0, 2, 1)
        x_ = self.g_fc1_bn(x_).permute(0, 2, 1)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_).permute(0, 2, 1)
        x_ = self.g_fc2_bn(x_).permute(0, 2, 1)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_).permute(0, 2, 1)
        x_ = self.g_fc3_bn(x_).permute(0, 2, 1)
        x_ = F.relu(x_)
        #x_ = self.g_fc4(x_).permute(0, 2, 1)
        #x_ = self.g_fc4_bn(x_).permute(0, 2, 1)
        #x_ = F.relu(x_)
        return x_


class RelationalNetwork(nn.Module):
    def __init__(self, b, d, f, f_hid, is_cuda=True):
        """
        b - batch size
        d - dimension of the image (assuming it's square)
        h - height of the feature map
        w - width of the feature map
        f - number of features (corresponds to number of joints)
        """
        super(RelationalNetwork, self).__init__()
        self.f_hid = f_hid

        self.g = GModule(f, f_hid)

        self.affine_aggregate = nn.Linear(d * d, 1)

        #aself.f_fc1 = nn.Linear(f_hid, f_hid)
        #self.f_fc1_bn = nn.BatchNorm1d(f_hid)

        self.coord_oi = torch.FloatTensor(b, 2)
        self.coord_oj = torch.FloatTensor(b, 2)
        if is_cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i, d):
            return [( (i+1) / d - d/2) / (d/2), ( (i+1) % d - d/2) / (d/2)]

        self.coord_tensor = torch.FloatTensor(b, d**2, 2)
        if is_cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((b, d**2, 2))
        for i in range(d**2):
            np_coord_tensor[:, i, :] = np.array(cvt_coord(i, d))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        self.fcout = RNOutputModel(f_hid, f)  # TODO: argument is number of joints

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, x):
        # x.shape = (b x n_channels x d x d)
        b = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(b, n_channels, d * d).permute(0, 2, 1)

        # add coordinates
        if b != self.coord_tensor.shape:  # due to last batch != cfg.BATCH_SIZE
            self.coord_tensor = self.coord_tensor[:b, ...]
        x_flat = torch.cat([x_flat, self.coord_tensor], 2)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (b x 1 x d^2 x f+2)
        x_i = x_i.repeat(1, d**2, 1, 1)  # (b x d^2 x d^2 x f+2)
        x_j = torch.unsqueeze(x_flat, 2)  # (b x d^2 x 1 x f+2)
        x_j = x_j.repeat(1, 1, d**2, 1)  # (b x d^2 x d^2 x f+2)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)  # (b x d^2 x d^2 x 2*(f+2))

        # reshape for passing through network
        x_ = x_full.view(b, d * d * d * d, 2*(n_channels+2))
        x_ = checkpoint.checkpoint(self.custom(self.g), x_)

        # reshape again and sum
        x_g = x_.view(b, d * d, d * d, self.f_hid)
        #x_g = x_g.sum(2).squeeze()
        x_g = x_g.permute(0, 1, 3, 2)
        x_g = self.affine_aggregate(x_g).squeeze()
        x_g = x_g.view(b, d * d, self.f_hid)

        """f"""
        x_f = self.f_fc1(x_g).permute(0, 2, 1)
        x_f = self.f_fc1_bn(x_f).permute(0, 2, 1)
        x_f = F.relu(x_f)
        out = self.fcout(x_f)
        out = out.view(b, d, d, n_channels).permute(0, 3, 1, 2)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        self.rn = RelationalNetwork(cfg.TRAIN.BATCH_SIZE_PER_GPU,
                                    int(cfg.MODEL.HEATMAP_SIZE[0]), # * 0.875),
                                    cfg.MODEL.NUM_JOINTS,
                                    128,
                                    is_cuda=True)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        #x = F.interpolate(x, size=(42, 42), mode='bilinear')
        #for_vis = x.data.cpu()[0, :3, :, :]
        #for_vis = np.transpose(for_vis, (1, 2, 0))
        #print(for_vis.shape)
        #plt.imshow(for_vis)
        #plt.show()
        #plt.close()
        x = self.rn(x)
        #x = F.interpolate(x, size=(48, 48), mode='bilinear')

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                #elif isinstance(m, nn.Linear):
                #    nn.init.normal_(m.weight, std=0.001)
                #    nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
