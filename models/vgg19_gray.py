from functools import reduce

import torch
import torch.nn as nn


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))

        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


layer_names = [
    "conv1_1",
    "relu1_1",
    "conv1_2",
    "relu1_2",
    "pool1",
    "conv2_1",
    "relu2_1",
    "conv2_2",
    "relu2_2",
    "pool2",
    "conv3_1",
    "relu3_1",
    "conv3_2",
    "relu3_2",
    "conv3_3",
    "relu3_3",
    "conv3_4",
    "relu3_4",
    "pool3",
    "conv4_1",
    "relu4_1",
    "conv4_2",
    "relu4_2",
    "conv4_3",
    "relu4_3",
    "conv4_4",
    "relu4_4",
    "pool4",
    "conv5_1",
    "relu5_1",
    "conv5_2",
    "relu5_2",
    "conv5_3",
    "relu5_3",
    "conv5_4",
    "relu5_4",
    "pool5",
    "view1",
    "fc6",
    "fc6_relu",
    "fc7",
    "fc7_relu",
    "fc8",
]

model = nn.Sequential(  # Sequential,
    nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    Lambda(lambda x: x.view(x.size(0), -1)),  # View,
    nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(25088, 4096)),  # Linear,
    nn.ReLU(),
    nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(4096, 4096)),  # Linear,
    nn.ReLU(),
    nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(4096, 1000)),  # Linear,
)


model.load_state_dict(torch.load("data/vgg19_gray.pth"))
vgg19_gray_net = torch.nn.Sequential()
for (name, layer) in model._modules.items():
    vgg19_gray_net.add_module(layer_names[int(name)], model[int(name)])

for param in vgg19_gray_net.parameters():
    param.requires_grad = False
vgg19_gray_net.eval()


class vgg19_gray(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(vgg19_gray, self).__init__()
        vgg_pretrained_features = vgg19_gray_net
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        for x in range(12):
            self.slice1.add_module(layer_names[x], vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice2.add_module(layer_names[x], vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice3.add_module(layer_names[x], vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu3_1 = h
        h = self.slice2(h)
        h_relu4_1 = h
        h = self.slice3(h)
        h_relu5_1 = h
        return h_relu3_1, h_relu4_1, h_relu5_1


class vgg19_gray_new(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(vgg19_gray_new, self).__init__()
        vgg_pretrained_features = vgg19_gray_net
        self.slice0 = torch.nn.Sequential()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        for x in range(7):
            self.slice0.add_module(layer_names[x], vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice1.add_module(layer_names[x], vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice2.add_module(layer_names[x], vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice3.add_module(layer_names[x], vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice0(X)
        h_relu2_1 = h
        h = self.slice1(h)
        h_relu3_1 = h
        h = self.slice2(h)
        h_relu4_1 = h
        h = self.slice3(h)
        h_relu5_1 = h
        return h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1
