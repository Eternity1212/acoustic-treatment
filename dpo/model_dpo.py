"""DPO 声乐偏好训练的模型定义。

本模块单独维护 CAMPPlus 网络及其依赖层，避免把模型结构、数据读取
和训练调度都堆在入口脚本中。
"""

import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        return torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_filter_attention(self, x):
        return torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        return torch.sigmoid(spatial_attention / self.temperature)

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        return F.softmax(kernel_attention / self.temperature, dim=1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        reduction=0.0625,
        kernel_num=4,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(
            in_planes,
            out_planes,
            kernel_size,
            groups=groups,
            reduction=reduction,
            kernel_num=kernel_num,
        )
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
            requires_grad=True,
        )
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode="fan_out", nonlinearity="relu")

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)

        batch_size, _, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)

        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size]
        )

        output = F.conv2d(
            x,
            weight=aggregate_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * batch_size,
        )

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output * filter_attention

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, _, _ = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(
            x,
            weight=self.weight.squeeze(dim=0),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output * filter_attention

    def forward(self, x):
        return self._forward_impl(x)


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU())
        elif name == "sigmod":
            nonlinear.add_module("sigmod", nn.Sigmoid())
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm2d(channels))
        else:
            raise ValueError(f"Unexpected module ({name}).")
    return nonlinear


def statistics_pooling(x, axis=1, keepdim=True, unbiased=True):
    mean = x.mean(dim=axis)
    std = x.std(dim=axis, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=1)
    if keepdim:
        stats = stats.unsqueeze(dim=axis)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class ODConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=2,
        padding=1,
        dilation=1,
        kernel_num=1,
        reduction=0.0625,
        config_str="batchnorm-relu",
    ):
        super().__init__()
        self.linear = ODConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            kernel_num=kernel_num,
            reduction=reduction,
        )
        self.nonlinear1 = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        return self.nonlinear1(self.linear(x))


class CAMLayer(nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, reduction=2):
        super().__init__()
        self.linear_local = nn.Conv2d(
            bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.linear1 = nn.Conv2d(1, bn_channels // reduction, 1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Conv2d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        x0 = x.max(1, keepdim=True)[0]
        context = x.mean(1, keepdim=True) + x0
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, config_str="batchnorm-relu"):
        super().__init__()
        assert kernel_size % 2 == 1, f"Expect equal paddings, but got even kernel size ({kernel_size})"
        padding = (kernel_size - 1) // 2 * dilation

        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv2d(in_channels, bn_channels, 1)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        return self.cam_layer(self.nonlinear2(self.bn_function(x)))


class CAMDenseTDNNBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, config_str="batchnorm-relu"):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                config_str=config_str,
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.linear(self.nonlinear(x))


class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        super().__init__()
        self.linear = nn.Conv2d(in_channels, out_channels, 1)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        return self.nonlinear(self.linear(x))


class BasicResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, expansion=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.expansion = expansion
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class FCM(nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=None, m_channels=64, in_channels=1):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2]
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(in_channels, m_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2, expansion=1)

    def _make_layer(self, block, planes, num_blocks, stride, expansion):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride, expansion))
            self.in_planes = planes * expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return F.relu(self.bn1(self.conv1(x)))


class CAMPPlus(nn.Module):
    def __init__(
        self,
        num_class,
        input_size,
        embd_dim=4096,
        growth_rate=32,
        bn_size=4,
        in_channels=64,
        init_channels=128,
        config_str="batchnorm-relu",
    ):
        super().__init__()
        self.head = FCM(block=BasicResBlock, num_blocks=[2, 2], m_channels=64, in_channels=input_size)

        self.xvector = nn.Sequential(
            ODConvLayer(
                in_channels,
                in_channels * 2,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
                kernel_num=1,
                reduction=0.0625,
                config_str="batchnorm-relu",
            ),
            ODConvLayer(
                in_channels * 2,
                in_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                padding=0,
                kernel_num=1,
                reduction=0.0625,
                config_str="batchnorm-relu",
            ),
        )

        self.xvector0 = nn.Sequential(
            ODConvLayer(
                in_channels,
                in_channels * 2,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
                kernel_num=1,
                reduction=0.0625,
                config_str="batchnorm-relu",
            ),
            ODConvLayer(
                in_channels * 2,
                in_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                padding=0,
                kernel_num=1,
                reduction=0.0625,
                config_str="batchnorm-relu",
            ),
        )

        self.xvector1 = nn.Sequential(
            CAMDenseTDNNBlock(
                num_layers=3,
                in_channels=in_channels,
                out_channels=growth_rate,
                bn_channels=growth_rate * bn_size,
                kernel_size=3,
                stride=1,
                dilation=1,
                config_str=config_str,
            ),
            TransitLayer(
                in_channels=init_channels + growth_rate * 2,
                out_channels=init_channels,
                bias=False,
                config_str=config_str,
            ),
        )

        self.relu = get_nonlinear(config_str, init_channels)
        self.pool = StatsPool()
        self.xvector_3 = LinearLayer(1, 1, config_str="batchnorm-relu")
        self.output_1 = nn.Linear(embd_dim, num_class)

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=1,
            time_stripes_num=16,
            freq_drop_width=1,
            freq_stripes_num=16,
        )
        self.bn0 = nn.BatchNorm2d(40)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.head(x)
        x1 = self.xvector(x)
        x2 = self.xvector0(x1)
        x3 = self.relu(self.xvector1(x))

        x = torch.cat([x1, x2, x3], dim=1)
        feature = self.pool(x)
        x = self.xvector_3(feature)
        feature1 = x.view(x.size(0), -1)
        logits = self.output_1(feature1)
        return logits, feature, feature1


def build_model(config, device):
    """根据扁平化 DPO 配置创建 CAMPPlus 模型。"""

    model = CAMPPlus(
        num_class=config["num_classes"],
        input_size=config["input_size"],
        embd_dim=config["embd_dim"],
        growth_rate=config["growth_rate"],
        bn_size=config["bn_size"],
        init_channels=config["init_channels"],
        config_str=config["config_str"],
    ).to(device)
    return model
