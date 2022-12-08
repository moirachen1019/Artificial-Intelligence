import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
#x = np.array([[1, 2, 3, 4],[5,6,7,8]])
#print(x.shape[0])

dataPath = 'data/detect-part5/detectData.txt'
f = open(dataPath)
location = []
filename = []
count = []
for line in f.readlines(): #逐行讀取整個檔案
    s = line.split(' ')
    if( str.isnumeric(s[0]) ): #是數字
        location.append(int(s[0]))
        location.append(int(s[1]))
        location.append(int(s[2]))
        location.append(int(s[3]))
    else: #是檔名
        filename.append(s[0])
        count.append(int(s[1]))
f.close

file = 0
for i in count: #有幾個圖片就跑幾次
    img = cv2.imread('data/detect-part5/'+filename[file],cv2.IMREAD_GRAYSCALE)
    plt.imshow(Image.open('data/detect-part5/'+filename[file]))
    file += 1 
    for j in range(i): #該圖片有幾個框就跑幾次
        # 裁切區域的 x 與 y 座標（左上角）
        x = location[0]
        y = location[1]
        # 裁切區域的長度與寬度
        w = location[2]
        h = location[3]
        # 裁切圖片
        crop_img = img[y:y+h, x:x+w]
        #cv2.imshow("cropped", crop_img)
        resize_img = cv2.resize(crop_img, (19, 19))
        del location[0]
        del location[0]
        del location[0]
        del location[0]
        #ans = clf.classify(resize_img)
        #if ans == 1:
            #print("yes")
        plt.gca().add_patch(Rectangle((x,y),w,h,linewidth=1,edgecolor='g',facecolor='none'))
        #else:
            #print("no")
            #plt.gca().add_patch(Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none'))
    plt.show()
