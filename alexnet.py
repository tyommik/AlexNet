import torch
import torch.nn as nn


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        print(self.padding)


class AlexNet(nn.Module):
    """
    input: image with shape 224x224x3
    output: num_classes x 1
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            Conv2dAuto(in_channels=3,
                      out_channels=96,
                      kernel_size=(11, 11),
                      stride=4),
            nn.BatchNorm2d(num_features=96), # ???
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2),
            Conv2dAuto(in_channels=96,
                       out_channels=256,
                       kernel_size=(5, 5),
                       padding=2,
                       stride=1
                       ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2),
            Conv2dAuto(in_channels=256,
                       out_channels=384,
                       kernel_size=(3, 3),
                       padding=1,
                       stride=1
                       ),
            nn.ReLU(inplace=True),
            Conv2dAuto(in_channels=384,
                       out_channels=384,
                       kernel_size=(3, 3),
                       padding=1,
                       stride=1
                       ),
            nn.ReLU(inplace=True),
            Conv2dAuto(in_channels=384,
                       out_channels=256,
                       kernel_size=(3, 3),
                       padding=1,
                       stride=1
                       ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2)

        )
        self.max_pooling = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.max_pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


