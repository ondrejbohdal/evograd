import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_fw, self).__init__(in_features, out_features, bias=bias)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels,
                                        kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None,
                               stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast,
                               stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        return out


class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm2d_fw, self).__init__(num_features,
                                             momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var,
                               weight, bias, training=self.training, momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(
                x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
        return out


class BatchNorm2d_fix(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm2d_fix, self).__init__(num_features,
                                              momentum=momentum, track_running_stats=track_running_stats)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        weight = self.weight
        bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var,
                               weight, bias, training=self.training, momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(
                x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
        return out


def _weights_init(L):
    if isinstance(L, nn.Conv2d) or isinstance(L, nn.Linear):
        init.kaiming_normal_(L.weight)
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class LambdaLayer(nn.Module):
    def __init__(self, planes):
        super(LambdaLayer, self).__init__()
        self.planes = planes
        
    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0,
                self.planes//4, self.planes//4), "constant", 0)


class BasicBlock(nn.Module):
    expansion = 1
    maml = False

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        if self.maml:
            self.conv1 = Conv2d_fw(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = BatchNorm2d_fw(planes)
            self.conv2 = Conv2d_fw(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = BatchNorm2d_fw(planes)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = BatchNorm2d_fix(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = BatchNorm2d_fix(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(planes)
            elif option == 'B':
                if self.maml:
                    self.shortcut = nn.Sequential(
                        Conv2d_fw(in_planes, self.expansion * planes,
                                  kernel_size=1, stride=stride, bias=False),
                        BatchNorm2d_fw(self.expansion * planes)
                    )
                else:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion * planes,
                                  kernel_size=1, stride=stride, bias=False),
                        BatchNorm2d_fix(self.expansion * planes)
                    )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32(nn.Module):
    maml = False

    def __init__(self, num_classes, block=BasicBlock, num_blocks=[5, 5, 5], width=1):
        super(ResNet32, self).__init__()
        self.width = width
        self.in_planes = 16 * self.width

        if self.maml:
            self.conv1 = Conv2d_fw(3, 16 * self.width, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = BatchNorm2d_fw(16 * self.width)
            self.linear = Linear_fw(64 * self.width, num_classes)
        else:
            self.conv1 = nn.Conv2d(3, 16 * self.width, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = BatchNorm2d_fix(16 * self.width)
            self.linear = nn.Linear(64 * self.width, num_classes)

        self.layer1 = self._make_layer(
            block, 16 * self.width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            block, 32 * self.width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            block, 64 * self.width, num_blocks[2], stride=2)
        # verified this initialization works
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class VNet(nn.Module):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, output)
        # self.linear3 = nn.Linear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)







