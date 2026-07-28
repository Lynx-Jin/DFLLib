import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    """AlexNet with an ImageNet or small-image stem selected by input size."""

    def __init__(self, num_classes=1000, in_channels=3, input_size=224):
        super(AlexNet, self).__init__()
        if input_size <= 0:
            raise ValueError("AlexNet input_size must be positive")
        self.small_input = input_size < 64

        if self.small_input:
            # CIFAR-like inputs are too small for the ImageNet 11x11/stride-4
            # stem and three 3x3 pools.  This variant preserves AlexNet's
            # five convolution stages while retaining a valid 4x4 feature map.
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
            classifier_dims = (256 * 4 * 4, 1024, 512)
        else:
            # Canonical ImageNet AlexNet.  Keep this branch unchanged so
            # ImageNet use and official pretrained weights remain compatible.
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            classifier_dims = (256 * 6 * 6, 4096, 4096)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(classifier_dims[0], classifier_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(classifier_dims[1], classifier_dims[2]),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(classifier_dims[2], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.fc(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        if model.small_input or kwargs.get('in_channels', 3) != 3:
            raise ValueError("Official AlexNet pretrained weights require the ImageNet input configuration")
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)

        new_dict = {}
        for k, v in state_dict.items():
            if 'classifier.6' not in k:
                new_dict[k] = v
            else:
                new_dict[k.replace('classifier.6', 'fc')] = v

        model.load_state_dict(new_dict)
    return model
