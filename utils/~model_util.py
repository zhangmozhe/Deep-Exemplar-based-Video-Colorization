import torch.nn as nn
import torchvision.models as models

vgg_cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg_make_conv_layer_names(cfg, batch_norm=False):
    layers = []
    block_idx = 1
    layer_idx = 1
    for v in cfg:
        if v == "M":
            layers += ["pool%d" % block_idx]
            block_idx += 1
            layer_idx = 1
        else:
            if batch_norm:
                layers += [
                    "conv%d_%d" % (block_idx, layer_idx),
                    "conv%d_%dnorm" % (block_idx, layer_idx),
                    "relu%d_%d" % (block_idx, layer_idx),
                ]
            else:
                layers += ["conv%d_%d" % (block_idx, layer_idx), "relu%d_%d" % (block_idx, layer_idx)]
            layer_idx += 1
    return layers


def _vgg_make_fc_layer_names():
    layers = ["fc6", "relu6", "dropout6", "fc7", "relu7", "dropout7", "fc8"]
    return layers


class VGG19Before(nn.Module):
    def __init__(self, layer_name, part="conv"):
        super(VGG19Before, self).__init__()
        assert part in ["conv", "fc"], "invalid part name"
        vgg_19 = models.vgg19(pretrained=True)

        if part == "conv":
            vgg19_conv_layers = _vgg_make_conv_layer_names(vgg_cfg["E"], batch_norm=False)
            idx = vgg19_conv_layers.index(layer_name)
            assert 0 <= idx < len(vgg19_conv_layers), "cannot find %s in part %s" % (layer_name, part)
            self.features = nn.Sequential(*list(vgg_19.features.children())[: (idx + 1)])
        else:
            vgg19_fc_layers = _vgg_make_fc_layer_names()
            idx = vgg19_fc_layers.index(layer_name)
            assert 0 <= idx < len(vgg19_fc_layers), "cannot find %s in part %s" % (layer_name, part)
            self.features = vgg_19.features
            self.classifier = nn.Sequential(*list(vgg_19.classifier.children())[: (idx + 1)])

    def forward(self, x):
        x = self.features(x)
        if hasattr(self, "classifier"):
            x = self.classifier(x)
        return x
