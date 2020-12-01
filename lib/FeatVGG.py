import torch.nn as nn
import utils.vgg_util as vgg_util
from torchvision import models


def conv_to_relu(layer_names):
    out_layernames = []
    for name in layer_names:
        if name.startswith("conv"):
            out_layernames.append("relu" + name[4:])
        else:
            out_layernames.append(name)
    return out_layernames


class FeatVGG(nn.Module):
    def __init__(self, content_layers=["relu3_1"]):
        super(FeatVGG, self).__init__()
        self.content_layers = conv_to_relu(content_layers)
        self.vgg19 = vgg_util.get_renamed_vgg()
        self.last_c_layer = self.content_layers[-1]
        is_last_content = False
        replace_layers, del_layers = [], []
        for name, mod in self.vgg19.named_children():
            if is_last_content:
                del_layers.append(name)
            else:
                if name == self.last_c_layer:
                    is_last_content = True

        for name in del_layers:
            delattr(self.vgg19, name)

        # no need for gradweight vgg19
        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, input_img):
        # input image is BGR image
        # each channel ranges in [0,255]
        # should be normalized with mean = [0.406*255, 0.456*255, 0.485*255] = [103,116,123]
        # out = {}
        return self.vgg19(input_img)


class VGGNet_multilayer(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet_multilayer, self).__init__()
        self.select = ["0", "5", "10", "19", "28"]
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        """Extract multiple convolutional feature maps.
        x: rgb image
        ranges in [0,1]
        should be normalzied with mean = [0.485, 0.456, 0.406]
        and variance = [0.229, 0.224, 0.225]
        """
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features
