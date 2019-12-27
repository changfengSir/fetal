import torch as t
from torch.utils.data import DataLoader
from .dataset import DogCat
from torchvision.transforms import transforms
from torch import optim
import torch.nn as nn
from .vgg import VGG
from tqdm import tqdm
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Pytorch implement Fetal_posture detection')
parser.add_argument('--batch-size',default=16,type=int)
parser.add_argument('--lr',default=0.001,type=float)
parser.add_argument('--epoch',default=10,type=int)
parser.add_argument('--train-path',default='')
parser.add_argument('--test-path',default='')
parser.add_argument('--path',default='',help='location of images')
parser.add_argument('--save',default='./checkpoint',help='location of the saved model')
parser.add_argument('--weight-decay',default='5e-4')

args = parser.parse_args()

def prepare(args):

    train_dataset = DogCat(args.path,train=True, test=False ,transform=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_dataset = DogCat(args.path, train=False,test=False, transform=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)

    train_model(args,args.epoch,train_dataloader)
    test()


def train_model(args,dataloader):
    criterion = nn.CrossEntropyLoss()
    model = VGG('VGG13')
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.we)
    for epoch in range(args.epoch):
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
        print('Epoch:%d %d Loss:%f ' % (epoch, i, epoch_loss / (i + 1)))
    print('Epoch:%d Loss:%f '%(epoch,epoch_loss/(i+1)))



def test():
    model = VGG('VGG13')
    model = model.cuda()

    model.eval()
    test_loss = 0
    correct = 0
    for i, (data, label) in enumerate(val_dataloader):
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
    test()