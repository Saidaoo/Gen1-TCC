"""Models definitions archictures"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import DWT, IWT, DWTone
import segmentation_models_pytorch as smp
import numpy as np
from torchinfo import summary

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros(
        (in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64
    )
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN8s(nn.Module):

    def __init__(self, n_class=6):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0]
                )
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5 : 5 + upscore2.size()[2], 5 : 5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :, 9 : 9 + upscore_pool4.size()[2], 9 : 9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31 : 31 + x.size()[2], 31 : 31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)


class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(
        self,
        in_channels,
        out_channels,
        pretrained=False,
        pool_type="dwt",
        n_features=64,
        kernel_size=3,
    ):  # Out channels usually n_classes
        super(SegNet, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == "dwt" else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        # self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        self.pool = (
            DWT()
            if self.pool_type == "dwt"
            else nn.MaxPool2d(2, stride=2, return_indices=True)
        )
        self.unpool = IWT() if self.pool_type == "dwt" else nn.MaxUnpool2d(2, stride=2)

        self.conv1_1 = nn.Conv2d(
            in_channels, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_1 = nn.Conv2d(
            self.n_features * self.multiplier,
            self.n_features * 2,
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features * 2)
        self.conv2_2 = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features * 2)

        self.conv3_1 = nn.Conv2d(
            self.n_features * self.multiplier * 2,
            self.n_features * 4,
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_3 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features * 4)

        self.conv4_1 = nn.Conv2d(
            self.n_features * self.multiplier * 4,
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features * 8)

        self.conv5_1 = nn.Conv2d(
            (
                self.n_features * 8
                if self.pool_type == "gap"
                else self.n_features * self.multiplier * 8
            ),
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features * 8)

        # Retirar a camada 5 caso fique pior
        self.conv5_3_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == "dwt" else 8
        self.conv5_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv5_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv4_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv4_1_D_out_sz))

        self.conv3_3_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features * 4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(
            self.n_features * 4,
            self.n_features * int(conv3_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv3_1_D_out_sz))

        self.conv2_2_D = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features * 2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(
            self.n_features * 2,
            self.n_features * int(conv2_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv2_1_D_out_sz))

        self.conv1_2_D = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)

        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)

        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        # x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        if not os.path.isfile("./vgg16_bn-6c64b313.pth"):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, "./vgg16_bn-6c64b313.pth")

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load("./vgg16_bn-6c64b313.pth")
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))

        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()
        self.conv_in = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        residual = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out += residual
        return out


class SegNet_resblock(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(
        self,
        in_channels,
        out_channels,
        pretrained=False,
        pool_type="dwt",
        n_features=64,
        kernel_size=3,
    ):  # Out channels usually n_classes
        super(SegNet_resblock, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == "dwt" else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        number_of_kernels = 128
        # self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        self.pool = (
            DWT()
            if self.pool_type == "dwt"
            else nn.MaxPool2d(2, stride=2, return_indices=True)
        )
        self.unpool = IWT() if self.pool_type == "dwt" else nn.MaxUnpool2d(2, stride=2)

        self.conv1_1 = nn.Conv2d(
            in_channels, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        self.res_block_1 = ResidualBlock(self.n_features, number_of_kernels)

        self.conv2_1 = nn.Conv2d(
            self.n_features * self.multiplier,
            self.n_features * 2,
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features * 2)
        self.conv2_2 = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features * 2)
        self.res_block_2 = ResidualBlock(self.n_features * 2, number_of_kernels)

        self.conv3_1 = nn.Conv2d(
            self.n_features * self.multiplier * 2,
            self.n_features * 4,
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_3 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features * 4)
        self.res_block_3 = ResidualBlock(self.n_features * 4, number_of_kernels)

        self.conv4_1 = nn.Conv2d(
            self.n_features * self.multiplier * 4,
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features * 8)
        self.res_block_4 = ResidualBlock(self.n_features * 8, number_of_kernels)

        self.conv5_1 = nn.Conv2d(
            (
                self.n_features * 8
                if self.pool_type == "gap"
                else self.n_features * self.multiplier * 8
            ),
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features * 8)
        self.res_block_5 = ResidualBlock(self.n_features * 8, number_of_kernels)

        # Retirar a camada 5 caso fique pior
        self.conv5_3_D = nn.Conv2d(
            self.n_features * 8 + self.n_features * 2,
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == "dwt" else 8
        self.conv5_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv5_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(
            self.n_features * 8 + self.n_features * 2,
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv4_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv4_1_D_out_sz))

        self.conv3_3_D = nn.Conv2d(
            self.n_features * 4 + self.n_features * 2,
            self.n_features * 4,
            self.kernel_size,
            padding=1,
        )
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features * 4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(
            self.n_features * 4,
            self.n_features * int(conv3_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv3_1_D_out_sz))

        self.conv2_2_D = nn.Conv2d(
            self.n_features * 2 + self.n_features * 2,
            self.n_features * 2,
            self.kernel_size,
            padding=1,
        )
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features * 2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(
            self.n_features * 2,
            self.n_features * int(conv2_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv2_1_D_out_sz))

        self.conv1_2_D = nn.Conv2d(
            self.n_features + self.n_features * 2,
            self.n_features,
            self.kernel_size,
            padding=1,
        )
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)

        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(conv1)

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(conv2)

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        conv3 = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(conv3)

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        conv4 = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(conv4)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        conv5 = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(conv5)

        # Decoder block 5
        conv5 = self.res_block_5(conv5)
        x = self.unpool(x, mask5)
        x = torch.cat([x, conv5], dim=1)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        conv4 = self.res_block_4(conv4)
        x = self.unpool(x, mask4)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        conv3 = self.res_block_3(conv3)
        x = self.unpool(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        conv2 = self.res_block_2(conv2)
        x = self.unpool(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        conv1 = self.res_block_1(conv1)
        x = self.unpool(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        # x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        if not os.path.isfile("./vgg16_bn-6c64b313.pth"):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, "./vgg16_bn-6c64b313.pth")

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load("./vgg16_bn-6c64b313.pth")
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))

        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class SegNet_two_pools(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(
        self,
        in_channels,
        out_channels,
        pretrained=False,
        pool_type="dwt",
        n_features=64,
        kernel_size=3,
    ):  # Out channels usually n_classes
        super(SegNet_two_pools, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == "dwt" else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWT()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)

        self.conv1_1 = nn.Conv2d(
            in_channels, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_0_1 = nn.Conv2d(self.n_features * 4, self.n_features * 2, 1)
        self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features * 2, 1)
        self.conv2_1 = nn.Conv2d(
            self.n_features * 4, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features * 2)
        self.conv2_2 = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features * 2)

        self.conv3_0_1 = nn.Conv2d(self.n_features * 8, self.n_features * 4, 1)
        self.conv3_0_2 = nn.Conv2d(self.n_features * 2, self.n_features * 4, 1)
        self.conv3_1 = nn.Conv2d(
            self.n_features * 8, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_3 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features * 4)

        self.conv4_0_1 = nn.Conv2d(self.n_features * 16, self.n_features * 8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features * 4, self.n_features * 8, 1)
        self.conv4_1 = nn.Conv2d(
            self.n_features * 16, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features * 8)

        self.conv5_0_1 = nn.Conv2d(self.n_features * 32, self.n_features * 8, 1)
        self.conv5_0_2 = nn.Conv2d(self.n_features * 8, self.n_features * 8, 1)
        self.conv5_1 = nn.Conv2d(
            self.n_features * 16, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features * 8)

        # Retirar a camada 5 caso fique pior
        self.conv5_0_1_D = nn.Conv2d(self.n_features * 32, self.n_features * 8, 1)
        self.conv5_0_2_D = nn.Conv2d(self.n_features * 8, self.n_features * 8, 1)
        self.conv5_3_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == "dwt" else 8
        self.conv5_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv5_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv4_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv4_1_D_out_sz))

        self.conv3_3_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features * 4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(
            self.n_features * 4,
            self.n_features * int(conv3_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv3_1_D_out_sz))

        self.conv2_2_D = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features * 2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(
            self.n_features * 2,
            self.n_features * int(conv2_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv2_1_D_out_sz))

        self.conv1_2_D = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)

        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(x)
        x2, mask1_2 = self.pool2(x)

        # Encoder block 2
        x1 = self.conv2_0_1(x1)
        # x1 = torch.zeros_like(x1)
        # x2 = torch.zeros(6, 128, 128, 128)
        x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(x)
        x2, mask2_2 = self.pool2(x)

        # Encoder block 3
        x1 = self.conv3_0_1(x1)
        x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(x)
        x2, mask3_2 = self.pool2(x)

        # Encoder block 4
        x1 = self.conv4_0_1(x1)
        x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(x)
        x2, mask4_2 = self.pool2(x)

        # Encoder block 5
        x1 = self.conv5_0_1(x1)
        x2 = self.conv5_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(x)
        x2, mask5_2 = self.pool2(x)

        # Decoder block 5
        x1 = self.conv5_0_1_D(x1)
        x2 = self.conv5_0_2_D(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.unpool1(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        # x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        if not os.path.isfile("./vgg16_bn-6c64b313.pth"):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, "./vgg16_bn-6c64b313.pth")

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load("./vgg16_bn-6c64b313.pth")
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))

        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class SegNet_two_pools_test(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(
        self,
        in_channels,
        out_channels,
        pretrained=False,
        pool_type="dwt",
        n_features=64,
        kernel_size=3,
    ):  # Out channels usually n_classes
        super(SegNet_two_pools_test, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == "dwt" else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWTone()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)

        self.conv1_1 = nn.Conv2d(
            in_channels, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_0_1 = nn.Conv2d(self.n_features * 3, self.n_features * 2, 1)
        # self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features*2, 1)
        self.conv2_1 = nn.Conv2d(
            self.n_features * 4, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features * 2)
        self.conv2_2 = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features * 2)

        self.conv3_0_1 = nn.Conv2d(self.n_features * 6, self.n_features * 4, 1)
        # self.conv3_0_2 = nn.Conv2d(self.n_features*2, self.n_features*4, 1)
        self.conv3_1 = nn.Conv2d(
            self.n_features * 8, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_3 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features * 4)

        self.conv4_0_1 = nn.Conv2d(self.n_features * 12, self.n_features * 8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features * 4, self.n_features * 8, 1)
        self.conv4_1 = nn.Conv2d(
            self.n_features * 16, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features * 8)

        self.conv5_0_1 = nn.Conv2d(self.n_features * 24, self.n_features * 8, 1)
        # self.conv5_0_2 = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_1 = nn.Conv2d(
            self.n_features * 24, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features * 8)

        # Retirar a camada 5 caso fique pior
        self.conv5_0_1_D = nn.Conv2d(self.n_features * 24, self.n_features * 8, 1)
        self.conv5_0_2_D = nn.Conv2d(self.n_features * 8, self.n_features * 8, 1)
        self.conv5_3_D = nn.Conv2d(
            self.n_features * 6, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == "dwt" else 8
        self.conv5_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv5_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv4_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv4_1_D_out_sz))

        self.conv3_3_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features * 4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(
            self.n_features * 4,
            self.n_features * int(conv3_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv3_1_D_out_sz))

        self.conv2_2_D = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features * 2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(
            self.n_features * 2,
            self.n_features * int(conv2_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv2_1_D_out_sz))

        self.conv1_2_D = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)

        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask1_2 = self.pool2(x)

        # Encoder block 2
        x1_h = self.conv2_0_1(x1_h)
        # x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask2_2 = self.pool2(x)

        # Encoder block 3
        x1_h = self.conv3_0_1(x1_h)
        # x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask3_2 = self.pool2(x)

        # Encoder block 4
        x1_h = self.conv4_0_1(x1_h)
        # x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask4_2 = self.pool2(x)

        # Encoder block 5
        x1_h = self.conv5_0_1(x1_h)
        # x2 = self.conv5_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask5_2 = self.pool2(x)

        # Decoder block 5
        x1_h = self.conv5_0_1_D(x1_h)
        # x2 = self.conv5_0_2_D(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.unpool1(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        # x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x), dim=1)
        return x

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        if not os.path.isfile("./vgg16_bn-6c64b313.pth"):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, "./vgg16_bn-6c64b313.pth")

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load("./vgg16_bn-6c64b313.pth")
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))

        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class SegNet_two_pools_skip(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(
        self,
        in_channels,
        out_channels,
        pretrained=False,
        pool_type="dwt",
        n_features=64,
        kernel_size=3,
    ):  # Out channels usually n_classes
        super(SegNet_two_pools_skip, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == "dwt" else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWT()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)

        self.conv1_1 = nn.Conv2d(
            in_channels, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_0_1 = nn.Conv2d(self.n_features * 4, self.n_features * 2, 1)
        self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features * 2, 1)
        self.conv2_1 = nn.Conv2d(
            self.n_features * 4, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features * 2)
        self.conv2_2 = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features * 2)

        self.conv3_0_1 = nn.Conv2d(self.n_features * 8, self.n_features * 4, 1)
        self.conv3_0_2 = nn.Conv2d(self.n_features * 2, self.n_features * 4, 1)
        self.conv3_1 = nn.Conv2d(
            self.n_features * 8, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_3 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features * 4)

        self.conv4_0_1 = nn.Conv2d(self.n_features * 16, self.n_features * 8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features * 4, self.n_features * 8, 1)
        self.conv4_1 = nn.Conv2d(
            self.n_features * 16, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features * 8)

        self.conv5_0_1 = nn.Conv2d(self.n_features * 32, self.n_features * 8, 1)
        self.conv5_0_2 = nn.Conv2d(self.n_features * 8, self.n_features * 8, 1)
        self.conv5_1 = nn.Conv2d(
            self.n_features * 16, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features * 8)

        # Retirar a camada 5 caso fique pior
        self.conv5_0_1_D = nn.Conv2d(self.n_features * 32, self.n_features * 8, 1)
        self.conv5_0_2_D = nn.Conv2d(self.n_features * 8, self.n_features * 8, 1)
        self.conv5_3_D = nn.Conv2d(
            self.n_features * 4 + self.n_features * 8,
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv5_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == "dwt" else 8
        self.conv5_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv5_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(
            self.n_features * 8 + self.n_features * 8,
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv4_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv4_1_D_out_sz))

        self.conv3_3_D = nn.Conv2d(
            self.n_features * 4 + self.n_features * 4,
            self.n_features * 4,
            self.kernel_size,
            padding=1,
        )
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features * 4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(
            self.n_features * 4,
            self.n_features * int(conv3_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv3_1_D_out_sz))

        self.conv2_2_D = nn.Conv2d(
            self.n_features * 2 + self.n_features * 2,
            self.n_features * 2,
            self.kernel_size,
            padding=1,
        )
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features * 2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(
            self.n_features * 2,
            self.n_features * int(conv2_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv2_1_D_out_sz))

        self.conv1_2_D = nn.Conv2d(
            self.n_features + self.n_features,
            self.n_features,
            self.kernel_size,
            padding=1,
        )
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)

        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(conv1)
        x2, mask1_2 = self.pool2(conv1)

        # Encoder block 2
        x1 = self.conv2_0_1(x1)
        x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(conv2)
        x2, mask2_2 = self.pool2(conv2)

        # Encoder block 3
        x1 = self.conv3_0_1(x1)
        x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        conv3 = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(conv3)
        x2, mask3_2 = self.pool2(conv3)

        # Encoder block 4
        x1 = self.conv4_0_1(x1)
        x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        conv4 = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(conv4)
        x2, mask4_2 = self.pool2(conv4)

        # Encoder block 5
        x1 = self.conv5_0_1(x1)
        x2 = self.conv5_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        conv5 = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(conv5)
        x2, mask5_2 = self.pool2(conv5)

        # Decoder block 5
        x1 = self.conv5_0_1_D(x1)
        x2 = self.conv5_0_2_D(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.unpool1(x, mask5)
        x = torch.cat([x, conv5], dim=1)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        # x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        if not os.path.isfile("./vgg16_bn-6c64b313.pth"):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, "./vgg16_bn-6c64b313.pth")

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load("./vgg16_bn-6c64b313.pth")
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))

        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class SegNet_two_pools_2(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(
        self,
        in_channels,
        out_channels,
        pretrained=False,
        pool_type="dwt",
        n_features=64,
        kernel_size=3,
    ):  # Out channels usually n_classes
        super(SegNet_two_pools, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == "dwt" else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWT()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2)

        self.conv1_1 = nn.Conv2d(
            in_channels, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_1 = nn.Conv2d(
            self.n_features * 1 + self.n_features * 4,
            self.n_features * 2,
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features * 2)
        self.conv2_2 = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features * 2)

        self.conv3_1 = nn.Conv2d(
            self.n_features * 2 + self.n_features * 8,
            self.n_features * 4,
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_3 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features * 4)

        self.conv4_1 = nn.Conv2d(
            self.n_features * 4 + self.n_features * 16,
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features * 8)

        self.conv5_1 = nn.Conv2d(
            self.n_features * 8 + self.n_features * 32,
            self.n_features * 16,
            self.kernel_size,
            padding=1,
        )
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features * 16)
        self.conv5_2 = nn.Conv2d(
            self.n_features * 16, self.n_features * 16, self.kernel_size, padding=1
        )
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features * 16)
        self.conv5_3 = nn.Conv2d(
            self.n_features * 16, self.n_features * 16, self.kernel_size, padding=1
        )
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features * 16)

        # Retirar a camada 5 caso fique pior
        self.conv5_3_D = nn.Conv2d(
            self.n_features * 16 + self.n_features * 4,
            self.n_features * 16,
            self.kernel_size,
            padding=1,
        )
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features * 16)
        self.conv5_2_D = nn.Conv2d(
            self.n_features * 16, self.n_features * 16, self.kernel_size, padding=1
        )
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features * 16)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == "dwt" else 8
        self.conv5_1_D = nn.Conv2d(
            self.n_features * 16,
            self.n_features * int(conv5_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv4_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv4_1_D_out_sz))

        self.conv3_3_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features * 4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(
            self.n_features * 4,
            self.n_features * int(conv3_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv3_1_D_out_sz))

        self.conv2_2_D = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features * 2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(
            self.n_features * 2,
            self.n_features * int(conv2_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv2_1_D_out_sz))

        self.conv1_2_D = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)

        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(x)
        x2, mask1_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(x)
        x2, mask2_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(x)
        x2, mask3_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(x)
        x2, mask4_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(x)
        x2, mask5_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)

        # Decoder block 5
        x = self.unpool1(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        # x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        if not os.path.isfile("./vgg16_bn-6c64b313.pth"):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, "./vgg16_bn-6c64b313.pth")

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load("./vgg16_bn-6c64b313.pth")
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))

        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class SegNet_skip(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(
        self,
        in_channels,
        out_channels,
        pretrained=False,
        pool_type="dwt",
        n_features=64,
        kernel_size=3,
    ):  # Out channels usually n_classes
        super(SegNet_skip, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == "dwt" else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.pool = (
            DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        )
        self.unpool = IWT() if self.pool_type == "dwt" else nn.MaxUnpool2d(2)

        self.conv1_1 = nn.Conv2d(
            in_channels, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_1 = nn.Conv2d(
            self.n_features * self.multiplier,
            self.n_features * 2,
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features * 2)
        self.conv2_2 = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features * 2)

        self.conv3_1 = nn.Conv2d(
            self.n_features * self.multiplier * 2,
            self.n_features * 4,
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_3 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features * 4)

        self.conv4_1 = nn.Conv2d(
            self.n_features * self.multiplier * 4,
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_3 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features * 8)

        self.conv4_3_D = nn.Conv2d(
            self.n_features * 8 + self.n_features * 8,
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2_D = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features * 8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(
            self.n_features * 8,
            self.n_features * int(conv4_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv4_1_D_out_sz))

        self.conv3_3_D = nn.Conv2d(
            self.n_features * 4 + self.n_features * 4,
            self.n_features * 4,
            self.kernel_size,
            padding=1,
        )
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2_D = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features * 4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(
            self.n_features * 4,
            self.n_features * int(conv3_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv3_1_D_out_sz))

        self.conv2_2_D = nn.Conv2d(
            self.n_features * 2 + self.n_features * 2,
            self.n_features * 2,
            self.kernel_size,
            padding=1,
        )
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features * 2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(
            self.n_features * 2,
            self.n_features * int(conv2_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv2_1_D_out_sz))

        self.conv1_2_D = nn.Conv2d(
            self.n_features + self.n_features,
            self.n_features,
            self.kernel_size,
            padding=1,
        )
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)

        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(conv1)

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(conv2)

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        conv3 = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(conv3)

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        conv4 = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(conv4)

        # Decoder block 4
        x = self.unpool(x, mask4)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        # x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        if not os.path.isfile("./vgg16_bn-6c64b313.pth"):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, "./vgg16_bn-6c64b313.pth")

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load("./vgg16_bn-6c64b313.pth")
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))

        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class UNet(nn.Module):
    # Unet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(
        self,
        in_channels,
        out_channels,
        pretrained=False,
        pool_type="up",
        n_features=64,
        kernel_size=3,
    ):  # Out channels usually n_classes
        super(UNet, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == "dwt" else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.pool = (
            DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        )
        self.unpool = (
            IWT()
            if self.pool_type == "dwt"
            else nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.conv1_1 = nn.Conv2d(
            in_channels, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_1 = nn.Conv2d(
            self.n_features * self.multiplier,
            self.n_features * 2,
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features * 2)
        self.conv2_2 = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features * 2)

        self.conv3_1 = nn.Conv2d(
            self.n_features * self.multiplier * 2,
            self.n_features * 4,
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features * 4)

        self.conv4_1 = nn.Conv2d(
            self.n_features * self.multiplier * 4,
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features * 8)

        conv3_2_D_in_sz = 8 if self.pool_type == "up" else 2
        # self.conv3_2_D = nn.Conv2d(self.n_features*8 + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D = nn.Conv2d(
            self.n_features * int(conv3_2_D_in_sz) + self.n_features * 4,
            self.n_features * 4,
            self.kernel_size,
            padding=1,
        )
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features * 4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5 if self.pool_type != "up" else 4
        self.conv3_1_D = nn.Conv2d(
            self.n_features * 4,
            self.n_features * int(conv3_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv3_1_D_out_sz))

        conv2_2_D_in_sz = 4 if self.pool_type == "up" else 2
        # self.conv2_2_D = nn.Conv2d(self.n_features*4 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D = nn.Conv2d(
            self.n_features * int(conv2_2_D_in_sz) + self.n_features * 2,
            self.n_features * 2,
            self.kernel_size,
            padding=1,
        )
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features * 2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5 if self.pool_type != "up" else 2
        self.conv2_1_D = nn.Conv2d(
            self.n_features * 2,
            self.n_features * int(conv2_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv2_1_D_out_sz))

        # self.conv1_2_D = nn.Conv2d(self.n_features*2 + self.n_features, self.n_features, self.kernel_size, padding=1)
        conv1_2_D_in_sz = 2 if self.pool_type == "up" else 1
        self.conv1_2_D = nn.Conv2d(
            self.n_features * int(conv1_2_D_in_sz) + self.n_features,
            self.n_features,
            self.kernel_size,
            padding=1,
        )
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_1_D_bn = nn.BatchNorm2d(self.n_features)

        self.conv1_1_D_F = nn.Conv2d(self.n_features, out_channels, 3, padding=1)

        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(conv1)

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(conv2)

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        conv3 = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x, mask3 = self.pool(conv3)

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))

        # Decoder block 3
        x = self.unpool(x) if self.pool_type == "up" else self.unpool(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool(x) if self.pool_type == "up" else self.unpool(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool(x) if self.pool_type == "up" else self.unpool(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = self.conv1_1_D_bn(F.relu(self.conv1_1_D(x)))
        # x = self.conv1_1_D_F(x)
        x = F.log_softmax(self.conv1_1_D_F(x))
        return x

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        if not os.path.isfile("./vgg16_bn-6c64b313.pth"):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, "./vgg16_bn-6c64b313.pth")

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load("./vgg16_bn-6c64b313.pth")
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))

        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class UNet_two_pools(nn.Module):
    # Unet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(
        self,
        in_channels,
        out_channels,
        pretrained=False,
        pool_type="up",
        n_features=64,
        kernel_size=3,
    ):  # Out channels usually n_classes
        super(UNet_two_pools, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == "dwt" else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.pool1 = DWT()
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.unpool1 = IWT()
        self.unpool2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.pool = DWT() if self.pool_type == "dwt" else
        # self.unpool = IWT() if self.pool_type == "dwt" else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Conv2d(
            in_channels, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_0_1 = nn.Conv2d(self.n_features * 4, self.n_features * 2, 1)
        self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features * 2, 1)
        self.conv2_1 = nn.Conv2d(
            self.n_features * self.multiplier,
            self.n_features * 2,
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features * 2)
        self.conv2_2 = nn.Conv2d(
            self.n_features * 2, self.n_features * 2, self.kernel_size, padding=1
        )
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features * 2)

        self.conv3_0_1 = nn.Conv2d(self.n_features * 8, self.n_features * 4, 1)
        self.conv3_0_2 = nn.Conv2d(self.n_features * 2, self.n_features * 4, 1)
        self.conv3_1 = nn.Conv2d(
            self.n_features * self.multiplier * 2,
            self.n_features * 4,
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features * 4)
        self.conv3_2 = nn.Conv2d(
            self.n_features * 4, self.n_features * 4, self.kernel_size, padding=1
        )
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features * 4)

        self.conv4_0_1 = nn.Conv2d(self.n_features * 16, self.n_features * 8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features * 4, self.n_features * 8, 1)
        self.conv4_1 = nn.Conv2d(
            self.n_features * self.multiplier * 4,
            self.n_features * 8,
            self.kernel_size,
            padding=1,
        )
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features * 8)
        self.conv4_2 = nn.Conv2d(
            self.n_features * 8, self.n_features * 8, self.kernel_size, padding=1
        )
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features * 8)

        conv3_2_D_in_sz = 8 if self.pool_type == "up" else 2
        # self.conv3_2_D = nn.Conv2d(self.n_features*8 + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D = nn.Conv2d(
            self.n_features * int(conv3_2_D_in_sz) + self.n_features * 4,
            self.n_features * 4,
            self.kernel_size,
            padding=1,
        )
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features * 4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5 if self.pool_type != "up" else 4
        self.conv3_1_D = nn.Conv2d(
            self.n_features * 4,
            self.n_features * int(conv3_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv3_1_D_out_sz))

        conv2_2_D_in_sz = 4 if self.pool_type == "up" else 2
        # self.conv2_2_D = nn.Conv2d(self.n_features*4 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D = nn.Conv2d(
            self.n_features * int(conv2_2_D_in_sz) + self.n_features * 2,
            self.n_features * 2,
            self.kernel_size,
            padding=1,
        )
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features * 2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5 if self.pool_type != "up" else 2
        self.conv2_1_D = nn.Conv2d(
            self.n_features * 2,
            self.n_features * int(conv2_1_D_out_sz),
            self.kernel_size,
            padding=1,
        )
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features * int(conv2_1_D_out_sz))

        # self.conv1_2_D = nn.Conv2d(self.n_features*2 + self.n_features, self.n_features, self.kernel_size, padding=1)
        conv1_2_D_in_sz = 2 if self.pool_type == "up" else 1
        self.conv1_2_D = nn.Conv2d(
            self.n_features * int(conv1_2_D_in_sz) + self.n_features,
            self.n_features,
            self.kernel_size,
            padding=1,
        )
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(
            self.n_features, self.n_features, self.kernel_size, padding=1
        )
        self.conv1_1_D_bn = nn.BatchNorm2d(self.n_features)

        self.conv1_1_D_F = nn.Conv2d(self.n_features, out_channels, 3, padding=1)

        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(conv1)
        x2, mask1_2 = self.pool2(conv1)

        # Encoder block 2
        x1 = self.conv2_0_1(x1)
        x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(conv2)
        x2, mask2_1 = self.pool2(conv2)

        # Encoder block 3
        x1 = self.conv3_0_1(x1)
        x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        conv3 = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x1, mask3 = self.pool1(conv3)
        x2, mask3_2 = self.pool2(conv3)

        # Encoder block 4
        x1 = self.conv4_0_1(x1)
        x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))

        # Decoder block 3
        x = self.unpool2(x) if self.pool_type == "up" else self.unpool1(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool2(x) if self.pool_type == "up" else self.unpool1(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool2(x) if self.pool_type == "up" else self.unpool1(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = self.conv1_1_D_bn(F.relu(self.conv1_1_D(x)))
        # x = self.conv1_1_D_F(x)
        x = F.log_softmax(self.conv1_1_D_F(x))
        return x

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        if not os.path.isfile("./vgg16_bn-6c64b313.pth"):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, "./vgg16_bn-6c64b313.pth")

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load("./vgg16_bn-6c64b313.pth")
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))

        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class BlockWithDropout(nn.Module):
    def __init__(self, block: nn.Module, p: float):
        super().__init__()
        self.block = block
        self.dropout = (
            nn.Dropout2d(p=p) if (p is not None and p > 0.0) else nn.Identity()
        )

    def forward(self, *args, **kwargs):
        # chama o bloco original com quaisquer args/kwargs (preserva skip signature)
        x = self.block(*args, **kwargs)
        return self.dropout(x)


class UNetWithProgressiveDropout(smp.Unet):
    def __init__(self, encoder_dropout=None, decoder_dropout=None, *args, **kwargs):
        encoder_name = kwargs.get("encoder_name", "") or ""
        super().__init__(*args, **kwargs)

        # helper para transformar float/list em lista do tamanho de blocks
        def _expand_values(val, n):
            if isinstance(val, (float, int)):
                return [float(val)] * n
            if val is None:
                return [0.0] * n
            vals = list(val)
            assert len(vals) <= n, f"lista maior que n_blocks ({n})"
            # complete com zeros se necessário
            return vals + [0.0] * (n - len(vals))

        # --- Encoder: um dropout por bloco (se blocks existirem) ---
        if encoder_dropout:
            enc_blocks = getattr(self.encoder, "blocks", None)
            if enc_blocks is not None:
                p_list = _expand_values(encoder_dropout, len(enc_blocks))
                for i, blk in enumerate(enc_blocks):
                    p = p_list[i]
                    enc_blocks[i] = BlockWithDropout(blk, p)

            else:
                # fallback: aplicar por filhos diretos do encoder (menos recomendado)
                add_dropout_recursively(
                    self.encoder, encoder_dropout, dropout_layer=nn.Dropout2d
                )

        # --- Decoder: um dropout por bloco ---
        if decoder_dropout:
            dec_blocks = getattr(self.decoder, "blocks", None)
            if dec_blocks is not None:
                p_list = _expand_values(decoder_dropout, len(dec_blocks))
                for i, blk in enumerate(dec_blocks):
                    p = p_list[i]
                    dec_blocks[i] = BlockWithDropout(blk, p)
            else:
                add_dropout_recursively(
                    self.decoder, decoder_dropout, dropout_layer=nn.Dropout2d
                )

    def forward(self, x):
        return super().forward(x)


def build_model(model_name: str, params: list):

    print(f"🔧 Building model: {model_name}")
    print(f"   - Encoder: efficientnet-b4")
    print(f"   - Pretrained: imagenet")
    print(f"   - Classes: {params['n_classes']}")
    print(f"   - Input channels: 3")

    if model_name == "unet":
        print(f"   - Decoder channels: (256, 128, 64, 32, 16)")
        print(f"   - Encoder depth: 5")
        model = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=params["n_classes"],
            activation=None,
            decoder_use_norm=True,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_interpolation="nearest",
        )
    elif model_name == "segformer":
        model = smp.Segformer(
            encoder_name="xception",
            in_channels=3,
            classes=params["n_classes"],
            activation=None,
        )
    elif model_name == "deeplabv3plus":
        print(f"   - Architecture: DeepLabV3+ with ASPP")
        print(f"   - Decoder: Fixed structure (ASPP + Decoder)")
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b4",  # MESMO encoder da UNet
            encoder_weights=None,  # MESMO pré-treinamento
            in_channels=3,  # MESMO número de canais
            classes=params["n_classes"],  # MESMO número de classes
            activation=None,  # MESMA ativação
            # A estrutura do decoder é fixa (com ASPP)
            # Não precisa (e não aceita) decoder_channels
            # Não precisa (e não aceita) encoder_depth customizado
        )
        print(f"   - ASPP rates: (12, 24, 36) [default]")
    elif model_name == "segnet_modificada":
        model = SegNet_two_pools_test(
            in_channels=3,
            out_channels=params["n_classes"],
            pretrained=True,
            pool_type="dwt",
        )
    else:
        raise Exception("{} -> invalid model name.".format(model_name))

    model.to(params["device"])
    print(f"   - Device: {params['device']}")
    print("✅ Model built successfully!")
    return model
