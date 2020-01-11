from torch.utils.data import Dataset
import PIL.Image as Image
import xlrd
import os
import numpy as np

def read_xlrd(excelFile):
    img = []
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    for rowNum in range(table.nrows):
        rowVale = table.row_values(rowNum)
        img.append(rowVale)
    img = np.array(img)
    return img

def read_data_with_label(excelFile):
    img = []
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    for rowNum in range(table.nrows):
        rowVale = table.row_values(rowNum)
        img.append(rowVale)
    img = np.array(img)
    tmp_img=[]
    for i in range(img.shape[0]):
        tmp_name = img[i][0].split('.')[0]+'_Annotation.png'
        tmp_data = (tmp_name,img[i][1])
        tmp_img.append(tmp_data)
    img_ = np.array(tmp_img)
    return img_

def read_data_single(path):
    datas = os.listdir(path)
    data = np.array(datas)
    return data


def read_csv_data(path):
    import csv
    with open(path, 'r') as f:
        reader = csv.reader(f)
        csv_data=[]
        for row in reader:
            csv_data.append(row)
        csv_data_np=np.array(csv_data)
    return csv_data_np



class FetalPosture(Dataset):
    def __init__(self,path,mode=None,transform=None):
        self.path= path
        if mode=='train':
            self.data = read_xlrd('./dataset/training.xls')
        elif mode=='test':
            self.data = read_csv_data('./balanced_test.csv')
            # self.data = read_xlrd('./dataset/test.xlsx')
        else:
            self.data = read_data_single(path)

        self.transform = transform

    def __getitem__(self, index):
        data = Image.open(os.path.join(self.path, self.data[index,0]))
        data = data.convert('RGB')
        # data = np.array(data)
        label = 1 if '1.0'==self.data[index,1] else 0
        if self.transform!=None:
            data = self.transform(data)

        return data,label

    def __len__(self):
        return len(self.data)
    

if __name__ == '__main__':
    # data = read_data_with_label('./training.xls')
    # print(data)
    from torchvision.transforms import transforms
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    dataset = FetalPosture('./data/',mode='test',transform=train_transform)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=1)
    count=0
    for i,(data,label) in enumerate(train_dataloader):
        # if label=='1':
        #     count+=1
        print(label)
    # print(len(train_dataloader.dataset))
    # datas = read_data_single('./test_single')
    # print(datas)