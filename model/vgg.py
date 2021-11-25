# Based on VGG in PyTorch:
# https://github.com/pytorch/vision/blob/7b87af25ba03c2cd2579a66b6c2945624b25a277/torchvision/models/vgg.py
'''
Reference:
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition arXiv:1409.1556
'''
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        base_width: int = 64,
        num_classes: int = 1000,
        init_weights: bool = True,
        quantization: bool = False
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if quantization:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        else:
            self.quant, self.dequant = None, None

        self.classifier = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(base_width * 8 * 7 * 7, 4096)),
            ("relu1", nn.ReLU(True)),
            ("dropout1", nn.Dropout()),
            ("linear2", nn.Linear(4096, 4096)),
            ("relu2", nn.ReLU(True)),
            ("dropout2", nn.Dropout()),
            ("linear3", nn.Linear(4096, num_classes)),
        ]))

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x) if self.quant is not None else x
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x) if self.dequant is not None else x
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, in_channels: int = 3) -> nn.Sequential:
    layers: List[nn.Module] = []
    layers: OrderedDict[(str, nn.Module)] = OrderedDict()
    for i, v in enumerate(cfg):
        if v == 'M':
            layers["maxpool" + str(i)] = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers["conv" + str(i)] = conv2d
                layers["batchnorm2d" + str(i)] = nn.BatchNorm2d(v)
                layers["relu" + str(i)] = nn.ReLU(inplace=True)
            else:
                layers["conv" + str(i)] = conv2d
                layers["relu" + str(i)] = nn.ReLU(inplace=True)
            in_channels = v
    return nn.Sequential(layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

ratio_cfg: Dict[str, List[Union[str, int]]] = {
    'A': [1, 'M', 2, 'M', 4, 4, 'M', 8, 8, 'M', 8, 8, 'M'],
    'B': [1, 1, 'M', 2, 2, 'M', 4, 4, 'M', 8, 8, 'M', 8, 8, 'M'],
    'D': [1, 1, 'M', 2, 2, 'M', 4, 4, 4, 'M', 8, 8, 8, 'M', 8, 8, 8, 'M'],
    'E': [1, 1, 'M', 2, 2, 'M', 4, 4, 4, 4, 'M', 8, 8, 8, 8, 'M', 8, 8, 8, 8, 'M'],
}


def _vgg(arch: str, cfg: str, base_width: int, batch_norm: bool, num_classes: int = 1000, in_channels: int = 3,
         quantization: bool = False, **kwargs: Any) -> VGG:
    if base_width == 64:
        print("Building default {} base width {}".format(arch, base_width))
        config = cfgs[cfg]
    else:
        print("Building custom {} base width {}".format(arch, base_width))
        config = [value if type(value) == str else value * base_width for value in ratio_cfg[cfg]]

    model = VGG(make_layers(config, batch_norm, in_channels), base_width, num_classes,
                quantization=quantization, **kwargs)
    return model


def vgg11(num_classes: int = 1000, base_width: int = 64, in_channels: int = 3, quantization: bool = False,
          **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    """
    return _vgg('vgg11', 'A', base_width, False, num_classes, in_channels, quantization, **kwargs)


def vgg11_bn(num_classes: int = 1000, base_width: int = 64, in_channels: int = 3, quantization: bool = False,
             **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    """
    return _vgg('vgg11_bn', 'A', base_width, True, num_classes, in_channels, quantization, **kwargs)


def vgg13(num_classes: int = 1000, base_width: int = 64, in_channels: int = 3, quantization: bool = False,
          **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    """
    return _vgg('vgg13', 'B', base_width, False, num_classes, in_channels, quantization, **kwargs)


def vgg13_bn(num_classes: int = 1000, base_width: int = 64, in_channels: int = 3, quantization: bool = False,
             **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    """
    return _vgg('vgg13_bn', 'B', base_width, True, num_classes, in_channels, quantization, **kwargs)


def vgg16(num_classes: int = 1000, base_width: int = 64, in_channels: int = 3, quantization: bool = False,
          **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    """
    return _vgg('vgg16', 'D', base_width, False, num_classes, in_channels, quantization, **kwargs)


def vgg16_bn(num_classes: int = 1000, base_width: int = 64, in_channels: int = 3, quantization: bool = False,
             **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    """
    return _vgg('vgg16_bn', 'D', base_width, True, num_classes, in_channels, quantization, **kwargs)


def vgg19(num_classes: int = 1000, base_width: int = 64, in_channels: int = 3, quantization: bool = False,
          **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    """
    return _vgg('vgg19', 'E', base_width, False, num_classes, in_channels, quantization, **kwargs)


def vgg19_bn(num_classes: int = 1000, base_width: int = 64, in_channels: int = 3, quantization: bool = False,
             **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    """
    return _vgg('vgg19_bn', 'E', base_width, True, num_classes, in_channels, quantization, **kwargs)
