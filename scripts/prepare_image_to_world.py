# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv.imread('map.bag.pgm',0)
# ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
# ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
# ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
# ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

import cv2

img=cv2.imread("map.bag.pgm")
# img=cv2.resize(img, (300,300))
img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width= img.shape

white = [255,255,255]
black = [0,0,0]

for x in range(0,width):
    for y in range(0,height):
        if(img[y,x]<=65):
            img[y,x]=0
        if(img[y,x]>=196):
            img[y,x]=254
        if(img[y,x]<196 and img[y,x]>65):
            img[y,x]=205
        # channels_xy = img[y,x]
        # if all(channels_xy == white):    
        #     img[y,x] = black

        # elif all(channels_xy == black):
        #     img[y,x] = white


filename = 'savedImage.pgm'
cv2.imwrite(filename, img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()