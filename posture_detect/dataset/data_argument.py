import numpy as np
import cv2
import imutils
import os
import xlrd
import csv

def read_xlrd(excelFile):
    img = []
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    for rowNum in range(table.nrows):
        rowVale = table.row_values(rowNum)
        img.append(rowVale)
    imgs = np.array(img)
    return imgs

def main(path1,path2,imgs):
    for i in range(800):
        img = imgs[i,0]
        label = imgs[i,1]
        if label == 1.0:
            rotate(path1, path2, img, 90, '_1')
        elif label==2.0:
            rotate(path1, path2, img, 180, '_2')
        else:
            rotate(path1, path2, img, 180, '_3')

def rotate(source_path,target_path,img,angle,no):
    _img = cv2.imread(os.path.join(source_path, img))
    _img_rotate = imutils.rotate(_img, angle)
    new_name = img.split('.')[0] + no + 'png'
    cv2.imwrite(os.path.join(target_path,new_name), _img_rotate)

def add_label(path):
    lists = os.listdir(path)
    file = open('tmp.csv','a+')
    csvwriter = csv.writer(file)
    for img in lists:
        if '_1' in img:
            label = 2.0
        elif '_2' in img:
            label = 2.0
        else:
            label = 3.0
        data = img,label
        csvwriter.writerow(data)
    file.close

if __name__=='__main__':
    sourece_path = 'E:\\files\\workspace\\fetal\\simple_u_net_v2\\dataset\\test'
    target_path = 'E:\\files\\workspace\\fetal\\simple_u_net_v2\\dataset\\test2'
    imgs = read_xlrd('/training.xls')
    main(sourece_path,target_path,imgs)
    add_label(target_path)