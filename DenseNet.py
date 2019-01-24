# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out

# -*- coding: utf-8 -*-
"""DenseNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tv3dxRris3VfwZPwQtqKM9g8bZyVaj6h
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets


# Implemented DenseNet from https://arxiv.org/pdf/1608.06993.pdf

# TODO Normalize input data. 


# DenseBlock input should have k layers input and k layers output
class MyDenseBlock(nn.Module):
    def __init__(self, n_layers, k):
        super(DenseBlock, self).__init__()
        self.activation = nn.ReLU()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()

        # Each layer in the block takes in (i+1) * k inputs and produces k outputs
        for i in range(n_layers):
            batch_norm = nn.BatchNorm2d(num_features=(i+1)*k).cuda()
            bottleneck = nn.Conv2d((i+1)*k, k, kernel_size=1, stride=1, padding=0).cuda()
            conv = nn.Conv2d(k, k, kernel_size=3, stride=1, padding=1).cuda()
            self.layers.extend([batch_norm, self.activation, bottleneck, conv])

    def forward(self, x):
        state = x
        i = 0
        while i < self.n_layers:
            l1 = self.layers[i](state)
            l2 = self.layers[i+1](l1)
            l3 = self.layers[i+2](l2)
            l4 = self.layers[i+3](l3)
            state = torch.cat([state, l4], dim=1)
            i += 4

        return state


# DenseNet Model
class MyDenseNet(nn.Module):
    def __init__(self, input_size, output_size, n_layers, n_dense_blocks=1, k=8, k0=1):
        super(DenseNet, self).__init__()
        self.k = k
        self.k0 = k0
        self.input_filters = k0+k
        self.dense_size = input_size
        self.n_dense_blocks = n_dense_blocks

        initial_conv = nn.Conv2d(k0, k, kernel_size=1, stride=1, padding=0).cuda()
        self.layers = nn.ModuleList([initial_conv])

        for i in range(self.n_dense_blocks):
            self.dense_block = DenseBlock(n_layers=n_layers, k=k)
            self.transition_conv = nn.Conv2d(k*2, k, kernel_size=1, stride=1, padding=0).cuda()
            self.transition_pool = nn.AvgPool2d(kernel_size=2, stride=2).cuda()
            self.dense_size = int(self.dense_size/2)
            self.layers.extend([self.dense_block, self.transition_conv, self.transition_pool])

        self.semi_final_fc = nn.Linear(int(((input_size/(2**n_dense_blocks))**2)*k), output_size*8).cuda()
        self.final_fc = nn.Linear(output_size*8, output_size).cuda()
        self.softmax = nn.Softmax()

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        semi_fc = self.semi_final_fc(x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]))
        fc = self.final_fc(semi_fc)
        output = self.softmax(fc)
        return output


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("cuda"), target.to("cuda")
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(args, model, test_loader, name):
    model.eval()
    test_loss = 0
    correct = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            num += pred.shape[0]
        print(name + "correct: ", str(correct/num*100)+"%")


if __name__ == '__main__':
    batch_size = 128
    args = dict()
    args["epochs"] = 100
    args["log_interval"] = 50
    args["lr"] = 3e-5

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True)

    model = DenseNet()
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    for epoch in range(1, args["epochs"] + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(args, model, train_loader, "Epoch " + str(epoch) + " Train Accuracy")
        test(args, model, test_loader, "Epoch " + str(epoch) + " Test Accuracy")

