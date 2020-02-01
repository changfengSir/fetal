import torch as t
from torch.utils.data import DataLoader
from dataset.dataset import FetalPosture
from torchvision.transforms import transforms
from torch import optim
import torch.nn as nn
# from model.vgg import VGG
# # from model.ResNet34 import ResNet34
from model.posture_resnet import Posture
import argparse
# from utils import progress_bar
import logging

parser = argparse.ArgumentParser(description='Pytorch implement Fetal_posture detection')
parser.add_argument('--batch-size', default=5, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epoch', default=50, type=int)
# parser.add_argument('--model', default='VGG(\'VGG13\')')
# parser.add_argument('--train_path', default='E:/files/datasets/fetal/data', help='location of train images')
parser.add_argument('--train_path', default='/home/p920/changfeng/workspace/fetal_data/data/', help='location of train images')
# parser.add_argument('--test_path', default='E:/files/datasets/fetal/test', help='location of test images')
parser.add_argument('--test_path', default='/home/p920/changfeng/workspace/fetal_data/test', help='location of test images')
parser.add_argument('--save', default='./checkpoint/', help='location of the saved model')
parser.add_argument('--weight-decay', type=float, default=5e-4)
args = parser.parse_args()


def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# five_transform = transforms.Compose([
#     transforms.FiveCrop((224, 224)),
#     transforms.Lambda(lambda crops: t.stack([transforms.ToTensor()(crop) for crop in crops]))])
# 训练
def train(args):
    train_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(224,224),
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        # transforms.Normalize([0.15598613,0.15598613,0.15598613],[0.43895477,0.43895477,0.43895477])
    ])


    train_dataset = FetalPosture(args.train_path,mode='train',transform=train_transform)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    #
    train_model(args,args.epoch,train_dataloader)


def train_model(args,epochs,dataloader):
    model = Posture()
    model = nn.DataParallel(model)
    model = model.cuda()
    # model.load_state_dict(t.load('./checkpoint/weight_VGG13_2_epoch14.pth'), strict=False)
    # dsize = len(dataloader.dataset)
    model.train()
    criterion = nn.CrossEntropyLoss()
    if isinstance(model, t.nn.DataParallel):
        model = model.module
    optimizer = optim.Adam([
        {'params':model.features.parameters(),'lr':0.003,'weight_decay':0.9},
        {'params':model.fc.parameters(),'lr':0.001,'weight_decay':0.0005}
    ],lr=args.lr,weight_decay=args.weight_decay)
    global_step=0
    for epoch in range(epochs):
        epoch_loss = 0
        step = 0
        # adjust_learning_rate(optimizer,epoch)
        for i ,(data,label) in enumerate(dataloader):
            input = data.cuda()
            target = label.cuda()
            output = model(input)
            # print(output)
            step += 1
            global_step+=1
            optimizer.zero_grad()
            loss = criterion(output,target)
            if global_step%10==0:
                vis.line([loss.item()],[global_step],win='posture_loss',update='append')
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print('Epoch%d step%d loss:%0.3f'%(epoch,i,loss.item()))
        # print('Epoch:%d Loss:%0.3f '%(epoch,epoch_loss/step))
        logging.info('Epoch:%d Loss:%0.3f '%(epoch,epoch_loss/step))

        val(args,dataloader,model,epoch)
        test(args,model,epoch)

        logging.info('--'*30)

    t.save(model.state_dict(),'./checkpoint/weight_VGG13_transfer_argument_4_epoch%d.pth'%epoch)
    # t.save(model.state_dict(), 'weight_resnet34_%d_trainlabel.pth' % epoch)


# val
def val(args,test_dataloader,model,epoch):
    global best_acc

    model.eval()
    correct = 0
    total = len(test_dataloader.dataset)
    with t.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1,keepdim=True)
            # total += targets.size(0)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()
            # progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            # print('Acc: %.3f' %(100. * correct / total))
    # Save checkpoint.
        acc = 100. * correct / total
        vis.line([acc], [epoch], win='accuracy', update='append')
    # print('Accuracy：%.3f'%acc)
    logging.info('Val_Accuracy:%.3f'%acc)


# 测试
def test(args,model,epoch):
    test_transform = transforms.Compose([
        # transforms.Scale()
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        # transforms.Normalize([0.15598613,0.15598613,0.15598613],[0.43895477,0.43895477,0.43895477])
    ])
    test_dataset = FetalPosture(args.test_path, mode='test',transform=test_transform)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    global best_acc
    # model = Posture()
    # model = model.cuda()
    # model = nn.DataParallel(model)
    # model.load_state_dict(t.load('./checkpoint/weight_VGG13_transfer_argument_3_epoch9.pth'))

    model.eval()

    correct = 0
    total = len(test_dataloader.dataset)
    with t.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1,keepdim=True)
            # total += targets.size(0)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()
            # progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            # print('Acc: %.3f' %(100. * correct / total))
    # Save checkpoint.
        acc = 100. * correct / total
        vis.line([acc], [epoch], win='accuracy', update='append')
    # print('Accuracy：%.3f'%acc)
    logging.info('Test_Accuracy:%.3f'%acc)


# 测试结果写到文件
def test_label(args):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        # transforms.Normalize([0.15598613,0.15598613,0.15598613],[0.43895477,0.43895477,0.43895477])
    ])
    test_dataset = FetalPosture(args.test_path, mode='test', transform=test_transform)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    model = Posture()
    model = model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(t.load('./checkpoint/weight_VGG13_transfer_argument_1_epoch19.pth'))
    model.eval()

    total = len(test_dataloader.dataset)
    results = []
    with t.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            probability = t.nn.functional.softmax(outputs, dim=1)[:, 0].detach().tolist()
            # label = score.max(dim = 1)[1].detach().tolist()

            batch_results = [( probability_) for  probability_ in zip(probability)]

            results += batch_results
        write_csv(results, 'result.csv')


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

if __name__=="__main__":
    import datetime
    time = datetime.datetime.now()
    import visdom
    import os
    vis = visdom.Visdom(env='posture')
    vis.line([0.], [0.], win='posture_loss', opts=dict(title='posture_loss'))

    # 保存日志
    filename = str(time.month) + str(time.day) + str(time.hour) + '_' + 'fetal'
    save_dir = filename
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = save_dir + '/' + filename + '.log'
    trainlog(logfile)

    train(args)
    # test(args)
    # test_single(args)
    end = datetime.datetime.now()
    print(end-time)