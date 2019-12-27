import torch as t
from torch.utils.data import DataLoader
from dataset.dataset import FetalPosture
from torchvision.transforms import transforms
from torch import optim
import torch.nn as nn
from model.vgg import VGG
from tqdm import tqdm
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Pytorch implement Fetal_posture detection')
parser.add_argument('--batch-size',default=16,type=int)
parser.add_argument('--lr',default=0.001,type=float)
parser.add_argument('--epoch',default=10,type=int)
parser.add_argument('--path',default=None,help='location of images')
parser.add_argument('--save',default=None,help='location of the saved model')
args = parser.parse_args()

def train(args):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize()
    ])



    train_dataset = FetalPosture(args.path,mode='train', transform=train_transform)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    train_model(args.epoch,train_dataloader)


def train_model(epochs,dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam()
    model = VGG('VGG13')
    for epoch in range(epochs):
        epoch_loss = 0
        step = 0
        for i ,(data,label) in tqdm(enumerate(dataloader)):
            input = data
            target = label
            output = model(input)
            step += 1
            optimizer.zero_grad()
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Epoch:%d Step%d Loss:%f '%(epoch,i,loss.item()))
    print('Epoch:%d Loss:%f ' % (epoch, epoch_loss / step ))



def test():
    test_transform = transforms.Compose([

    ])
    test_dataset = FetalPosture(args.path, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    model = VGG('VGG13')
    model = model.cuda()

    model.eval()
    test_loss = 0
    correct = 0
    for i, (data, label) in enumerate(test_dataloader):
        data, label = data.cuda(), label.cuda()
        output = model(data)
        test_loss = F.nll_loss(output, label, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)

    print('\nTest set:Average Loss:{:.4f} Accuracy:{}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)
    ))

    return 0

if __name__=="__main__":
    train()