from cv2 import INTER_LINEAR
from matplotlib import pyplot as plt
import cv2
from MODELS.cbam import CBAM
from MyModel import *
from nnmodels.inception_resnet_v1 import InceptionResnetV1, InceptionResnetV1_f
from Attention_block import PaB
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import numpy as np
import torch.nn as nn
import math
import pdb
from pathlib import Path
import torchvision.transforms as trans
from PIL import Image


class PFDiscriminator(nn.Module):
    def __init__(self, num_input):
        super(PFDiscriminator, self).__init__()
        self.fc1 = nn.Linear(num_input, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.pr1 = nn.LeakyReLU(0.2, True)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.pr2 = nn.LeakyReLU(0.2, True)
        self.fc3 = nn.Linear(128, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.pr1(self.bn1(self.fc1(x)))
        x = self.pr2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = self.activation(x)
        return x


#################*******************Pose_Aware_Resnet_Modules**********#################
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks


class Backbone_frontal(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir', FMS=14):
        super(Backbone_frontal, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))

    def forward(self, x):
        x = self.input_layer(x)
        b = self.body(x)
        x = self.output_layer(b)
        return l2_norm(x)


class Backbone_profile(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir', FMS=14):
        super(Backbone_profile, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))

    def forward(self, x, z, y):
        x = self.input_layer(x)
        b = self.body(x)
        s = z.unsqueeze(2).unsqueeze(3).expand_as(b)
        channel_refined_feature = b * s
        spatial_refind_feature = channel_refined_feature * y
        x = spatial_refind_feature + b
        x = self.output_layer(x)
        return l2_norm(x)


############################*********************************###########################

class FacePoseAwareNet(nn.Module):
    def __init__(self, pose=None):
        super(FacePoseAwareNet, self).__init__()
        self.l0 = msedmodel()
        self.l_a = PaB(512)
        self.lf = resnet_f()
        self.lp = resnet_p()
        self.pose = pose

    def forward(self, x, pose):
        emb = None
        if pose == 'frontal':
            emb = self.lf(x)
        elif pose == 'profile':
            imgs = x.detach().cpu().numpy()[:, :, :, :]
            x1 = torch.from_numpy(imgs).permute(0, 1, 2, 3).float()
            x2 = F.interpolate(x1, size=(224, 224), mode='bilinear')
            x3 = torch.from_numpy(np.asarray(x2)).cuda()
            yaw, yaw_predicted, att_layer4 = self.l0(x3)  ####  when HopeNet resizes 160x160 to 224x224
            sp_matx, ch_matx = self.l_a(att_layer4)
            emb = self.lp(x, ch_matx, sp_matx)
            # emb = self.lp(x)

        return emb


class resnet_f(nn.Module):
    def __init__(self):
        super(resnet_f, self).__init__()
        #########FaceNet_Stem########
        # # self.inception_resnet_f = InceptionResnetV1_f(classify=False, pretrained='vggface2', num_classes=None,num_bins=256)
        # self.inception_resnet_f = InceptionResnetV1_f(classify=False, pretrained='vggface2', num_classes=None,
        #                                               num_bins=None)
        # # self.inception_resnet_f = InceptionResnetV1_f(pretrained='vggface2', num_bins=512)
        # # self.num_bins = 256
        #
        # layer = 0
        # for child in self.inception_resnet_f.children():
        #     layer += 1
        #     if layer < 14:
        #         for param in child.parameters():
        #             param.requires_grad = False
        #     else:
        #         for param in child.parameters():
        #             param.requires_grad = True
        #
        # # num_features = self.inception_resnet_f.last_linear.in_features
        # # self.inception_resnet_f.last_linear = nn.Linear(num_features, self.num_bins)
        # # self.inception_resnet_f.last_bn = nn.BatchNorm1d(self.num_bins)

        ###################********Resnet_stem***********############
        self.inception_resnet_f = Backbone_frontal(num_layers=50, drop_ratio=0.6, mode='ir')
        # pretrained_dict_f = torch.load('/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_attention_7x7_Memory/checkpoint/checkpoint_fine-tune_cmu/train_cmu_300wlp/32_LR0.000113_VALID_BEST.pt')
        # model_dict_f = self.inception_resnet_f.state_dict()
        # # 1. filter out unnecessary keys
        # pretrained_dict_f = {k: v for k, v in pretrained_dict_f.items() if k in model_dict_f}
        # # 2. overwrite entries in the existing state dict
        # model_dict_f.update(pretrained_dict_f)
        # # 3. load the new state dict
        # self.inception_resnet_f.load_state_dict(model_dict_f)
        #
        # layer = 0
        # for child in self.inception_resnet_f.children():
        #     layer += 1
        #     if layer < 3:
        #         for param in child.parameters():
        #             param.requires_grad = False
        #     else:
        #         for param in child.parameters():
        #             param.requires_grad = True

    def forward(self, x):
        x = self.inception_resnet_f(x)
        return x


class resnet_p(nn.Module):
    def __init__(self):
        super(resnet_p, self).__init__()

        ##############*******FaceNet_stem********#############
        # self.inception_resnet = InceptionResnetV1(classify=False, pretrained='vggface2', num_classes=None, num_bins=None)
        # self.inception_resnet_p = InceptionResnetV1(classify=False, pretrained='vggface2', num_classes=None,
        #                                             num_bins=None)
        # # self.inception_resnet_p = InceptionResnetV1(pretrained='vggface2', num_bins=512)
        # # self.num_bins = 256
        #
        # layer = 0
        # for child in self.inception_resnet_p.children():
        #     layer += 1
        #     if layer < 14:
        #         for param in child.parameters():
        #             param.requires_grad = False
        #     else:
        #         for param in child.parameters():
        #             param.requires_grad = True
        #
        # # num_features = self.inception_resnet_p.last_linear.in_features
        # # self.inception_resnet_p.last_linear = nn.Linear(num_features, self.num_bins)
        # # self.inception_resnet_p.last_bn = nn.BatchNorm1d(self.num_bins)

        ######################********Resnet_stem*******##############################
        self.inception_resnet_p = Backbone_profile(num_layers=50, drop_ratio=0.6, mode='ir')
        # pretrained_dict_p = torch.load(
        #     '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_attention_7x7_Memory/checkpoint/checkpoint_fine-tune_cmu/train_cmu_300wlp/32_LR0.000113_VALID_BEST.pt')
        # model_dict_p = self.inception_resnet_p.state_dict()
        # # 1. filter out unnecessary keys
        # pretrained_dict_p = {k: v for k, v in pretrained_dict_p.items() if k in model_dict_p}
        # # 2. overwrite entries in the existing state dict
        # model_dict_p.update(pretrained_dict_p)
        # # 3. load the new state dict
        # self.inception_resnet_p.load_state_dict(model_dict_p)
        #
        # layer = 0
        # for child in self.inception_resnet_p.children():
        #     layer += 1
        #     if layer < 3:
        #         for param in child.parameters():
        #             param.requires_grad = False
        #     else:
        #         for param in child.parameters():
        #             param.requires_grad = True

    def forward(self, x, z, y):
        x = self.inception_resnet_p(x, z, y)
        return x
