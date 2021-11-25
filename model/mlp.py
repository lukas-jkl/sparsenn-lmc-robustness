import torch.nn as nn
import torch


class MLP1_layer(nn.Module):
    def __init__(self, n_units: int, n_channels: int, n_classes: int = 10, bias: bool = False,
                 quantization: bool = False):
        super().__init__()

        if quantization:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        else:
            self.quant, self.dequant = None, None

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * n_channels, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_classes, bias=bias)
        )

    def forward(self, x):
        x = self.quant(x) if self.quant is not None else x
        x = self.layers(x)
        x = self.dequant(x) if self.dequant is not None else x
        return x


class MLP2_layer(nn.Module):
    def __init__(self, n_units: int, n_channels: int, n_classes: int = 10, bias: bool = False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * n_channels, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_classes, bias=bias)
        )

    def forward(self, x):
        return self.layers(x)


class MLP4_layer(nn.Module):
    def __init__(self, n_units: int, n_channels: int, n_classes: int = 10, bias: bool = False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * n_channels, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_classes, bias=bias)
        )

    def forward(self, x):
        return self.layers(x)


class MLP8_layer(nn.Module):

    def __init__(self, n_units: int, n_channels: int, n_classes: int = 10, bias: bool = False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * n_channels, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_units, bias=bias),
            nn.ReLU(),
            nn.Linear(n_units, n_classes, bias=bias)
        )

    def forward(self, x):
        return self.layers(x)
