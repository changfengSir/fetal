# -*- coding: utf-8 -*-
# import csv
# #
# # path='./training.csv'
# #
# # with open(path,'rb') as f:
# #     csvfile = csv.reader(f)
# #     for row in csvfile:
# #         print(row)
#
# import pandas as pd
#
# csv_data = pd.read_csv('./training.csv',encoding='unicode_escape')  # 读取训练数据
# print(csv_data.shape)  # (189, 9)
# # N = 5
# csv_batch_data = csv_data.tail(N)  # 取后5条数据
# print(csv_batch_data.shape)  # (5, 9)
# train_batch_data = csv_batch_data[list(range(0, 2))]  # 取这20条数据的3到5列值(索引从0开始)
# print(train_batch_data)
# coding=utf-8


import xlrd

img = []

def read_xlrd(excelFile):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    for rowNum in range(table.nrows):
        rowVale = table.row_values(rowNum)
        # for colNum in range(table.ncols):
            # if rowNum > 0 and colNum == 0:
            #     print(rowVale[0])
            # else:
            #     print(rowVale[colNum])
        img.append(rowVale)
        # print("---------------")
    return img
    # if判断是将 id 进行格式化
    # print("未格式化Id的数据：")
    # print(table.cell(1, 0))
    # 结果：number:1001.0


if __name__ == '__main__':
    excelFile = './training.xls'
    img = read_xlrd(excelFile=excelFile)
    import numpy as np
    img_ = np.array(img)
    print(len(img_))
