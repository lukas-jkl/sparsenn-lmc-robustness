import model.mlp as mlp
import model.resnet as resnet
import model.vgg as vgg
import torch.nn as nn


def build_model(arch: str, nchannels: int, nclasses: int, width: int, nlayers: int, quantization: bool = False) \
                 -> nn.Module:
    if arch == "MLP":
        if nlayers != 1 and quantization is True:
            raise NotImplementedError()
        if nlayers == 1:
            model = mlp.MLP1_layer(width, nchannels, nclasses, quantization=quantization)
        elif nlayers == 2:
            model = mlp.MLP2_layer(width, nchannels, nclasses)
        elif nlayers == 4:
            model = mlp.MLP4_layer(width, nchannels, nclasses)
        elif nlayers == 8:
            model = mlp.MLP8_layer(width, nchannels, nclasses)
        else:
            raise NotImplementedError("Architecture {} with {} layers not implemented".format(arch, nlayers))
    elif arch == "MLP_no_flatten":
        if nlayers == 1:
            model = nn.Sequential(
                nn.Linear(32 * 32 * nchannels, width),
                nn.ReLU(),
                nn.Linear(width, nclasses)
                )
        elif nlayers == 2:
            model = nn.Sequential(
                nn.Linear(32 * 32 * nchannels, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, nclasses)
                )
        else:
            raise NotImplementedError("Architecture {} with {} layers not implemented".format(arch, nlayers))
    elif "resnet" in arch:
        if nlayers == 18:
            model = resnet.ResNet18(nclasses, width, nchannels, quantization=quantization)
        elif nlayers == 34:
            model = resnet.ResNet34(nclasses, width, nchannels, quantization=quantization)
        elif nlayers == 50:
            model = resnet.ResNet50(nclasses, width, nchannels, quantization=quantization)
        else:
            raise NotImplementedError("Architecture {} with {} layers not implemented".format(arch, nlayers))
    elif "vgg" in arch:
        if nlayers == 11:
            model = vgg.vgg11(nclasses, width, nchannels, quantization=quantization)
        elif nlayers == 13:
            model = vgg.vgg13(nclasses, width, nchannels, quantization=quantization)
        elif nlayers == 16:
            model = vgg.vgg16(nclasses, width, nchannels, quantization=quantization)
        elif nlayers == 19:
            model = vgg.vgg19(nclasses, width, nchannels, quantization=quantization)
        else:
            raise NotImplementedError("Architecture {} with {} layers not implemented".format(arch, nlayers))
    else:
        raise NotImplementedError("Architecture {} not implemented".format(arch))
    return model
