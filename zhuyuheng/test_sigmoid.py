import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(x)


if __name__ == '__main__':
    model = Model()
    a = torch.randn([1, 1, 3])
    out = model(a.view([-1,1]))
    print(out.shape)