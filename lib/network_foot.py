import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class PoseNetFeat(nn.Module):  # point
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)  # point feature
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)  # image embedding
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)  # num_points = 1000

        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))  # 64
        emb = F.relu(self.e_conv1(emb))  # 64
        pointfeat_1 = torch.cat((x, emb), dim=1)  # 64+64 = 128

        x = F.relu(self.conv2(x))  # 1*128*1000
        emb = F.relu(self.e_conv2(emb))  # 1*128*1000
        pointfeat_2 = torch.cat((x, emb), dim=1)  # 128+128 = 256

        x = F.relu(self.conv5(pointfeat_2))  # 1*512×1000
        x = F.relu(self.conv6(x))  # 1*1024*1000

        ap_x = self.ap1(x)  # x_dim= 1*1024*1000=>1*1024*1

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)  # (1,1024,1000)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)  # 128 + 256 + 1024 = 1408


class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)  # point feature

        # self.conv1_c = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_cent = torch.nn.Conv1d(1408, 640, 1)
        # self.conv1_self1 = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_self2 = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_self3 = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_choose = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_mode = torch.nn.Conv1d(1408, 640, 1)

        # self.conv2_c = torch.nn.Conv1d(640, 256, 1)
        self.conv2_cent = torch.nn.Conv1d(640, 256, 1)
        # self.conv2_self1 = torch.nn.Conv1d(640, 256, 1)
        self.conv2_self2 = torch.nn.Conv1d(640, 256, 1)
        self.conv2_self3 = torch.nn.Conv1d(640, 256, 1)
        self.conv2_choose = torch.nn.Conv1d(640, 256, 1)
        self.conv2_mode = torch.nn.Conv1d(640, 256, 1)

        # self.conv3_c = torch.nn.Conv1d(256, 128, 1)
        self.conv3_cent = torch.nn.Conv1d(256, 128, 1)
        # self.conv3_self1 = torch.nn.Conv1d(256, 128, 1)
        self.conv3_self2 = torch.nn.Conv1d(256, 128, 1)
        self.conv3_self3 = torch.nn.Conv1d(256, 128, 1)
        self.conv3_choose = torch.nn.Conv1d(256, 128, 1)
        self.conv3_mode = torch.nn.Conv1d(256, 128, 1)

        # self.conv4_c = torch.nn.Conv1d(128, 1, 1)  # confidence
        self.conv4_cent = torch.nn.Conv1d(128, 3, 1)
        # self.conv4_self1 = torch.nn.Conv1d(128, 9, 1)
        self.conv4_self2 = torch.nn.Conv1d(128, 9, 1)
        self.conv4_self3 = torch.nn.Conv1d(128, 3, 1)
        self.conv4_choose = torch.nn.Conv1d(128, 3, 1)
        self.conv4_mode = torch.nn.Conv1d(128, 3, 1)

        # self.conv5_c = torch.nn.Conv1d(64, 1, 1)  # confidence
        # self.conv5_cent = torch.nn.Conv1d(64, 3, 1)
        # self.conv5_s = torch.nn.Conv1d(64, 9, 1)
        # self.conv5_choose = torch.nn.Conv1d(64, 3, 1)

        # self.choose_avg_pool = torch.nn.AvgPool1d(num_points)

        self.num_obj = num_obj

    def forward(self, img, x, choose):  # choose = [1,1,1000],img.shape= [1,3,160,160]
        out_img = self.cnn(img)  # out_img size = [1,32,120,160]

        bs, di, _, _ = out_img.size()  # bs=1  di=32
        emb = out_img.view(bs, di, -1)  # (1,32,120*160)
        choose = choose.repeat(1, di, 1)  # (1,32,1000)
        emb = torch.gather(emb, 2,
                           choose).contiguous()  # choose is the index to select emb of dim2, now emb=(1,32,1000)

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        # cx = F.relu(self.conv1_c(ap_x))
        cent_x = F.relu(self.conv1_cent(ap_x))
        # self1_x = F.relu(self.conv1_self1(ap_x))
        self2_x = F.relu(self.conv1_self2(ap_x))
        self3_x = F.relu(self.conv1_self3(ap_x))
        choose_x = F.relu(self.conv1_choose(ap_x))
        mode_x = F.relu(self.conv1_mode(ap_x))

        # cx = F.relu(self.conv2_c(cx))
        cent_x = F.relu(self.conv2_cent(cent_x))
        # self1_x = F.relu(self.conv2_self1(self1_x))
        self2_x = F.relu(self.conv2_self2(self2_x))
        self3_x = F.relu(self.conv2_self3(self3_x))
        choose_x = F.relu(self.conv2_choose(choose_x))
        mode_x = F.relu(self.conv2_mode(mode_x))

        # cx = F.relu(self.conv3_c(cx))
        cent_x = F.relu(self.conv3_cent(cent_x))
        # self1_x = F.relu(self.conv3_self1(self1_x))
        self2_x = F.relu(self.conv3_self2(self2_x))
        self3_x = F.relu(self.conv3_self3(self3_x))
        choose_x = F.relu(self.conv3_choose(choose_x))
        mode_x = F.relu(self.conv3_mode(mode_x))

        # cx = torch.sigmoid(self.conv4_c(cx)).view(bs, 1, self.num_points)  # 1×21×1×1000
        cent_x = self.conv4_cent(cent_x).view(bs, 3, self.num_points)
        # self1_x = self.conv4_self1(self1_x).view(bs, 9, self.num_points)
        self2_x = self.conv4_self2(self2_x).view(bs, 9, self.num_points)
        self3_x = self.conv4_self3(self3_x).view(bs, 3, self.num_points)
        choose_x = torch.sigmoid(self.conv4_choose(choose_x)).view(bs, 3, self.num_points)
        mode_x = torch.sigmoid(self.conv4_mode(mode_x)).view(bs, 3, self.num_points)

        b = 0

        # out_cx = cx.contiguous().transpose(2, 1).contiguous()  # (1,1000,1)
        out_cent = cent_x.contiguous().transpose(2, 1).contiguous()
        # out_self1 = self1_x.contiguous().transpose(2, 1).contiguous()  # 3 possible reflection point
        out_self2 = self2_x.contiguous().transpose(2, 1).contiguous()  # 3 possible foot point
        out_self3 = self3_x.contiguous().transpose(2, 1).contiguous()  # foot point, axis, circle point

        out_choose = choose_x.contiguous().transpose(2, 1).contiguous()
        out_mode = mode_x.contiguous().transpose(2, 1).contiguous()
        # out_self = torch.cat([out_self1, out_self2, out_self3], 2)
        # out_ref = out_self1
        out_foot_ref = out_self2
        out_rot = out_self3


        return out_cent, \
               out_foot_ref,\
               out_rot,\
               out_choose,\
               out_mode, emb.detach()


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x


class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)

        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)
        self.conv1_s = torch.nn.Linear(1024, 512)  # add symmetry

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)
        self.conv2_s = torch.nn.Linear(512, 128)  # add symmetry

        self.conv3_r = torch.nn.Linear(128, num_obj * 4)  # quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj * 3)  # translation
        self.conv3_s = torch.nn.Linear(128, num_obj * 15)  # add symmetry
        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)  # (1,1408,1000)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        sx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        sx = F.relu(self.conv2_t(sx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)
        sx = self.conv3_r(sx).view(bs, self.num_obj, 15)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_sx = torch.index_select(sx[b], 0, obj[b])

        return out_rx, out_tx, out_sx
        # return out_sx
