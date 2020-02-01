import torch
import torchvision

class Posture(torch.nn.Module):
    def __init__(self, num_classes=3, is_all=True):
        """Declare all needed layers.

        Args:
            num_classes, int.
            is_all, bool: In the all/fc phase.
        """
        torch.nn.Module.__init__(self)
        self._is_all = is_all

        if self._is_all:
            # Convolution and pooling layers of VGG-16.
            self.features = torchvision.models.resnet50(pretrained=False)
            self.features = torch.nn.Sequential(*list(self.features.children())[:-1])
            #                                     [:-1])  # Remove fc.

        # self.gap = torch.nn.AdaptiveAvgPool2d((1,1))
        # Classification layer.
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(
            in_features=2048, out_features=3, bias=True)
        # # torch.nn.ReLU(True),
        # torch.nn.LeakyReLU(inplace=True),
        # torch.nn.Dropout(),
        # torch.nn.Linear(
        #     in_features=1024, out_features=1024, bias=True),
        # # torch.nn.ReLU(True),
        # torch.nn.LeakyReLU(inplace=True),
        # torch.nn.Dropout(),
        # torch.nn.Linear(in_features=1024, out_features=2, bias=True)

        )


        if not self._is_all:
            self.apply(Posture._initParameter)

    def _initParameter(module):
        """Initialize the weight and bias for each module.

        Args:
            module, torch.nn.Module.
        """
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, val=1.0)
            torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out',
                                          nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)

    def forward(self, X):
        x = self.features(X)
        # x = self.gap(x)

        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        # print(x)
        return torch.nn.functional.softmax(x,dim=1)

if __name__=='__main__':
    net = Posture()
    print(net)
    print('----')
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y)
