import numpy as np
# import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path, PurePath


def randomCrop(img, dim):
    dx,dy = dim
    height, width = img.shape[0], img.shape[1]
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx),]

def doubleRandomCrop(img1, img2, dim):
    dx,dy = dim
    height, width = img1.shape[0], img1.shape[1]
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img1[y:(y+dy), x:(x+dx),], img2[y:(y+dy), x:(x+dx),]
def dilate(image, kernelSize=(3,3)):
    kernel = np.ones(kernelSize, 'uint8')
    dilatedImg = np.zeros_like(image)    
    dilatedImg = cv2.dilate(image, kernel)
    return dilatedImg


if __name__ == '__main__':
    input_dir = "/home/dan/dem_site_project/"
    target_dir = "/home/dan/dem_site_project/"
    input_dir_save_dir = "/home/dan/dem_site_project/datacrop_3band_minmaxscaler/"
    target_save_dir = "/home/dan/dem_site_project/labelcrop_filled/"
    size=(256*3,256*3)


    x = cv2.imread(input_dir+"data2.png")
    y = cv2.imread(target_dir+"label2.png")
    # y = np.load("label.npy")
    print(y.max())

    # cv2.imshow("how"+inpath,x)
    #x = cv2.resize(x, (1024, 1024))
    #y = cv2.resize(y, (1024, 1024))

    # cv2.imshow("ho1"+inpath,x)
    # cv2.imshow(x,cmap="Greys_r")

    factor=2

    numocrops=8
    #size is of 7566, 8143
    for i in range(14):
        for j in range(15):
            # cropx,cropy = doubleRandomCrop(x,y,size)
            cropx = x[256* factor * i:256*factor *(i+1),256*factor*j:256*factor*(j+1),]
            cropy = y[256* factor * i:256*factor *(i+1),256*factor*j:256*factor*(j+1),]

            cv2.imwrite(input_dir_save_dir+str(i)+"_"+str(j)+"_"+"data"+".png",cropx)
            # dily = cropy
            cv2.imwrite(target_save_dir+str(i)+"_"+str(j)+"_"+"label"+"_mask.png",cropy)
            np.save(target_save_dir+str(i)+"_"+str(j)+"_"+"label"+"_mask.png",cropy)



    
    