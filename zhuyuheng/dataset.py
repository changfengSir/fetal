from torch.utils.data import Dataset
import numpy as np
import torch
import csv

def get_all_data(path):
    with open(path, 'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        data=[]
        for row in reader:
            data.append(row)
        return data

def get_data(file,index):
    data = file[index][0].split(' ')
    return np.array([[[data[0]],[data[1]],[data[2]]]]).astype(np.float32)

# 标签为4的算为0
def get_label(file,index):
    # label=-1
    data = file[index][0].split(' ')
    # if data[-1]=='4':
    #     # data[-1]=0
    #     label='0'
    return np.array([[[data[3]]]]).astype(np.float32)


class Data(Dataset):
    def __init__(self,path,transform=None):
        super(Data,self).__init__()
        self.path = path
        self.transform = transform
        self.data = get_all_data(self.path)

    def __getitem__(self, index):
        data = get_data(self.data,index)
        data = torch.from_numpy(data)
        # data = self.transform(data)
        label = get_label(self.data,index)
        label = torch.from_numpy(label)
        # label = self.transform(label)
        # label = label.view_as(1,1)
        # print(data.shape)
        return data,label
        # return data

    def __len__(self):
        return len(self.data)

if __name__ =='__main__':
    # from torchvision import transforms
    # transform = transforms.ToTensor()
    # dataset = Data('./data.txt',transform=transform)
    # from torch.utils.data import DataLoader
    # dataloaders = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    # for i,(data,label) in enumerate(dataloaders):
    #     print(data.shape)
    #     # label = label.view(-1,1)
    #     # label = label.squeeze()
    #     # print(label.shape)
    #     print(label.squeeze())
    #     # loss = criterion(outputs, labels)
    file = get_all_data('./data.txt')
    # length = len(file)
    data = get_label(file,40)
    print(data)
    # print(file[-1][0].split(' '))