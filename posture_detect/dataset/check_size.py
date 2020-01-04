import os
import numpy as np
import PIL.Image as Image
import csv

path = 'E:\\files\\workspace\\fetal\\simple_u_net_v2\\dataset\\data'
imgs = os.listdir(path)
length = len(imgs)
list=[]
for i in range(length):
    img = imgs[i]
    open_img = Image.open(os.path.join(path,img))
    width,height = open_img.size
    # print(type(width))
    # print(height)
    if width != 800 or height != 540:
        list.append(img)
        with open('imgs.csv', 'a+',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([img])
