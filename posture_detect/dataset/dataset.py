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


class FetalPosture(Dataset):
    def __init__(self,path,transform=None):
        self.path= path
        self.data = read_xlrd('./dataset/training.xls')
        self.transform = transform

    def __getitem__(self, index):
        data = Image.open(os.path.join(self.path, self.data[index,0]))
        label = 1 if 1.0==self.data[index,1] else 0
        if self.transform!=None:
            data = self.transform(data)

        return data,label

    def __len__(self):
        return len(self.data)