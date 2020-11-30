import torch.nn as nn
import utils.vgg_util as vgg_util


def conv_to_relu(layer_names):
    out_layernames = []
    for name in layer_names:
        if name.startswith("conv"):
            out_layernames.append("relu" + name[4:])
        else:
            out_layernames.append(name)
    return out_layernames


class VGGFeatureLoss(object):
    def __init__(self, content_layers=["relu3_1"], content_weights=[1]):
        self.content_layers = conv_to_relu(content_layers)
        self.content_weights = content_weights
        # set up vgg19 model
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
        print("del_layers", del_layers)
        for name in del_layers:
            delattr(self.vgg19, name)

        # set up loss
        self.loss_layers = self.content_layers
        self.contentloss_fns = [nn.MSELoss()] * len(self.content_layers)
        self.content_loss = None

        # target
        self.content_targets = None

        # no need for gradweight vgg19
        for param in self.vgg19.parameters():
            param.requires_grad = False

    def vgg_forward(self, input_img, layers, last_layer):
        # out = {}
        x = input_img
        out = None
        # print(last_layer)
        assert len(layers) == 1 and layers[-1] == last_layer
        for name, mod in self.vgg19.named_children():
            # print('name is : ', name)
            x = mod(x)
            if name == last_layer:
                out = x
                break
            # out[name] = mod(prev_input)
            # prev_input = out[name]
            # if name == last_layer:
            #     break
        return [out]
        # return [out[key] for key in layers]

    def cuda(self, gpu_ids):
        if len(gpu_ids) > 1:
            self.vgg19 = nn.DataParallel(self.vgg19.cuda(), gpu_ids)
        else:
            self.vgg19.cuda()
        self.contentloss_fns = [loss_fn.cuda() for loss_fn in self.contentloss_fns]

    def set_content_targets(self, content_img):
        self.content_targets = [
            A.detach() for A in self.vgg_forward(content_img, self.content_layers, self.last_c_layer)
        ]

    def forward(self, img):
        assert self.content_targets is not None, "set up content targets first"
        out = self.vgg_forward(img, self.content_layers, self.last_c_layer)
        self.content_loss = sum(
            [
                self.content_weights[idx] * self.contentloss_fns[idx](out[idx], self.content_targets[idx])
                for idx in range(len(self.content_layers))
            ]
        )
        return self.content_loss

    # def backward(self):
    #     self.loss.backward()

