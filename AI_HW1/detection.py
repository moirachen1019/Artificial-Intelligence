import os
import cv2
import adaboost
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image



def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    f = open(dataPath+"detectData.txt")
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
      img = cv2.imread(dataPath+filename[file],cv2.IMREAD_GRAYSCALE)
      plt.imshow(Image.open(dataPath+filename[file]))
      file += 1 
      for j in range(i): #該圖片有幾個框就跑幾次
        x = location[0] #裁切區域的x與y座標（左上角）
        y = location[1]
        w = location[2] #裁切區域的長度與寬度
        h = location[3]
        crop_img = img[y:y+h, x:x+w]
        resize_img = cv2.resize(crop_img, (19, 19),interpolation=cv2.INTER_NEAREST)
        for k in range(4):
          del location[0]
        ans = clf.classify(resize_img)
        if ans == 1:
          plt.gca().add_patch(Rectangle((x,y),w,h,linewidth=1,edgecolor='g',facecolor='none'))
        else:
          plt.gca().add_patch(Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none'))
      plt.show()
    # End your code (Part 4)
