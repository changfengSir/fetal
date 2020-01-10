import xlrd
import numpy as np


"""把3个类别的数据分好类"""
img_1 = []
img_2 = []
img_3 = []
xl_data = xlrd.open_workbook('test.xlsx')
table = xl_data.sheet_by_index(0)
for rowNum in range(table.nrows):
    rowVale = table.row_values(rowNum)
    if rowVale[1]==1.0:
        img_1.append(rowVale)
    elif rowVale[1] == 2.0:
        img_2.append(rowVale)
    else:
        img_3.append(rowVale)

# print(len(img_1))
# print(len(img_2))
# print(len(img_3))

"""从第1类中随机抽取与2,3类相等数量的图片,
    取出索引
"""
import random
h=[]
while(len(h)<(len(img_2)+len(img_3))):
    h.append(random.randint(0,len(img_1)-1))
# print(h)

""" 读取新1类的图片信息,
    并和2,3类合并
"""
new_img_1=[]
for i in h:
    new_img_1.append(img_1[i])
new_list = new_img_1+img_2+img_3
# print(new_list)
# new_list = np.array(new_list)

import csv
with open('balanced_test.csv','w',newline='') as f:

    # writer = csv.writer(f)
    # writer.writerows(zip(new_list))
    for i in range(len(new_list)):
        f.write(("{name},{place}\n".format(name='\n'.join(new_list[i][0]), place='\n'.join(new_list[i][1]))))
    #     writer.writerows(zip(new_list[i][0],new_list[i][1]))
        # writer.writerow([new_list[i][0]])
        # writer.writerow([new_list[i][1]])
