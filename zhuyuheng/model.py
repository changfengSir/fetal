import torch.nn as nn
import torch

class Unit(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Unit, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch,out_ch),
            nn.ReLU(inplace=True)
            # nn.Dropout()
        )

    def forward(self,x):
        return self.fc(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Unit(3, 3)
        # self.fc2 = Unit(100, 100)
        self.fc3 = Unit(3, 4)

    def forward(self,x):
        x1 = x.view(-1,3)
        x = self.fc1(x1)
        # x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    input = torch.randn([3,1,3,1])
    model = Net()
    out = model(input)
#     # out = out.squeeze()
#     print(out.shape)
#     label = torch.randn([3,1]).long()
#     label = label.squeeze()
#     print(label.shape)
#     criterion = nn.CrossEntropyLoss()
#     loss = criterion(out, label)
#     print(loss)
