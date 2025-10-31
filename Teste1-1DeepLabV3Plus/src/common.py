# common

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair, _quadruple
from torch.autograd import Variable


def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2)+dilation-1, bias=bias, dilation=dilation)


def default_conv1(in_channels, out_channels, kernel_size, bias=True, groups=3):
    return nn.Conv2d(
        in_channels,out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups=groups)



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def pixel_down_shuffle(x, downsacale_factor):
    batchsize, num_channels, height, width = x.size()

    out_height = height // downsacale_factor
    out_width = width // downsacale_factor
    input_view = x.contiguous().view(batchsize, num_channels, out_height, downsacale_factor, out_width,
                                     downsacale_factor)

    num_channels *= downsacale_factor ** 2
    unshuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()

    return unshuffle_out.view(batchsize, num_channels, out_height, out_width)


def sp_init(x):

    x01 = x[:, :, 0::2, :]
    x02 = x[:, :, 1::2, :]
    x_LL = x01[:, :, :, 0::2]
    x_HL = x02[:, :, :, 0::2]
    x_LH = x01[:, :, :, 1::2]
    x_HH = x02[:, :, :, 1::2]


    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)



def dwt_init_one(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class Channel_Shuffle(nn.Module):
    def __init__(self, conv_groups):
        super(Channel_Shuffle, self).__init__()
        self.conv_groups = conv_groups
        self.requires_grad = False

    def forward(self, x):
        return channel_shuffle(x, self.conv_groups)

class SP(nn.Module):
    def __init__(self):
        super(SP, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return sp_init(x)

class Pixel_Down_Shuffle(nn.Module):
    def __init__(self):
        super(Pixel_Down_Shuffle, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return pixel_down_shuffle(x, 2)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x), None # Extra token return added to match with normal Maxpooling2D input



class DWTone(nn.Module):
    def __init__(self):
        super(DWTone, self).__init__()
        self.requires_grad = False
    
    def forward(self, x):
        return dwt_init_one(x), None


class DualPooling(nn.Module):
    def __init__(self, in_channels):
        super(DualPooling, self).__init__()
        self.branch_0_1 = nn.Conv2d(self.in_channels*4, self.in_channels*2, 1)
        self.branch_0_2 = nn.Conv2d(self.in_channels, self.in_channels*2, 1)
        self.pool_1 = DWT()
        self.pool_2 = nn.MaxPool2d(2, stride=2)
        
        

    def forward(self, x):
         x1 = self.branch_0_1(self.pool_1(x))
         x2 = self.branch_0_2(self.pool_2(x))
         x = torch.cat([x1, x2], dim=1)
         return x


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x, garbage_param): # Extra token input added to match with normal MaxUnpool2D input
        return iwt_init(x)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign==-1:
            self.create_graph = False
            self.volatile = True


class MeanShift2(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift2, self).__init__(4, 4, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(4).view(4, 4, 1, 1)
        self.weight.data.div_(std.view(4, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign==-1:
            self.volatile = True

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=False, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class BBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x

class DBlock_com(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_com1(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv1(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_com2(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv2(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class ShuffleBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,conv_groups=1):

        super(ShuffleBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        m.append(Channel_Shuffle(conv_groups))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class DWBlock(nn.Module):
    def __init__(
        self, conv, conv1, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DWBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        m.append(conv1(in_channels, out_channels, 1, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Block(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(Block, self).__init__()
        m = []
        for i in range(4):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        # res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class AdapGaussianPool2d(nn.Module):

    def __init__(self, kernel_size = 3, stride = 1, padding = 1, same = False):
        super(AdapGaussianPool2d, self).__init__()
        self.k, self.stride = _pair(kernel_size), _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same
    
    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        x = F.pad(x, self._padding(x), mode = 'reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        print(x)
        x = x.contiguous().view(x.size()[:4] + (-1,))
        print(x)
        #x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

'''
class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
'''