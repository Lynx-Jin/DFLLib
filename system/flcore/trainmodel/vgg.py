"""VGG11 models sized for the 28x28 and 32x32 image datasets used here."""

import torch.nn as nn


_VGG11_CONFIGURATION = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class VGG11(nn.Module):
    """Small-image VGG11 with an adaptive classifier head.

    ``ceil_mode=True`` on pooling is intentional: it keeps the final 1x1
    feature map valid for MNIST's 28x28 inputs while retaining the standard
    five-pooling-stage VGG11 layout for CIFAR10's 32x32 inputs.
    """

    def __init__(self, in_channels, num_classes, batch_norm=True):
        super().__init__()
        layers = []
        channels = int(in_channels)
        for item in _VGG11_CONFIGURATION:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
                continue
            layers.append(nn.Conv2d(channels, item, kernel_size=3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(item))
            layers.append(nn.ReLU(inplace=True))
            channels = item
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # FedAvg/FedSGD split a model at ``.fc`` via BaseHeadSplit.  Keep the
        # final classifier as a top-level module, like the project's AlexNet
        # and ResNet implementations, rather than hiding it in a Sequential.
        self.fc = nn.Linear(channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


def vgg11(in_channels, num_classes, batch_norm=True):
    """Build the VGG11 variant used by the model selector."""
    return VGG11(in_channels=in_channels, num_classes=num_classes, batch_norm=batch_norm)
