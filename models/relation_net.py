import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import register


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


@register('relation_net')
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size=[64, 21, 21], hidden_size=8, loss_type='softmax'):
        super(RelationNetwork, self).__init__()

        self.loss_type = loss_type
        # when using Resnet,
        # conv map without avgpooling is 7x7, need padding in block to do pooling
        padding = 1 if (input_size[1] < 10) and (input_size[2] < 10) else 0
        shrink_s = lambda s: int((int((s - 2 + 2 * padding) / 2) - 2 + 2 * padding) / 2)

        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2, input_size[0], kernel_size=3, padding=padding),
                        nn.BatchNorm2d(input_size[0], affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(input_size[0],input_size[0], kernel_size=3, padding=padding),
                        nn.BatchNorm2d(input_size[0], affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size[0]*shrink_s(input_size[1])*shrink_s(input_size[1]), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = F.relu(self.fc1(out))
        if self.loss_type == 'mse':
            out = F.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)
        return out

