from model import Net
from dataset import Data
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from torch import optim
import torch
import numpy as np


# torch.set_default_dtype()
transform = transforms.ToTensor()


def train():
    print('-----')
    model = Net()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),weight_decay=5e-4)
    data = Data("./data.txt",transform=transform)
    dataloaders = DataLoader(data, batch_size=20, shuffle=True, num_workers=4)
    for epoch in range(51):
        epoch_loss = 0
        step = 0
        for i,(data,label) in enumerate(dataloaders):
            step += 1
            inputs = data
            labels = label.squeeze()
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d,train_loss:%0.3f" % (step, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))
        test(model)
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)

def test(model):
    # model = Net()
    # model.eval()
    # model.load_state_dict(torch.load('weights_10.pth'))
    dataset = Data('./datatest.txt')
    dataloaders = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=4)
    correct = 0
    total = len(dataset)
    with torch.no_grad():
        for i,(data,label) in enumerate(dataloaders):
            inputs = data
            label = label.long()
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1, keepdim=True)
            # total += targets.size(0)
            correct += predicted.eq(label.view_as(predicted)).sum().item()

        acc = 100. * correct / total
        with open('./result_2.csv',mode='a+',newline='')as f:
            import csv
            writer = csv.writer(f)
            writer.writerow([acc])
        print('Accuracy：%.3f' % acc)
# 0.088888889

if __name__ =='__main__':
    train()
    # test()