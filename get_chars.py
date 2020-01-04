import numpy as np
import cv2 as cv


def getChars(imageName):

    img = cv.imread(imageName)

    img = cv.resize(img, (200,200), interpolation=cv.INTER_AREA)

    kernel = np.ones((5,5),np.uint8)

    _,img = cv.threshold(img,150,255,cv.THRESH_BINARY)

    img_higher_thickness = cv.erode(img,kernel,iterations = 1)


    imgray = cv.cvtColor(img_higher_thickness, cv.COLOR_BGR2GRAY)


    contours, _ = cv.findContours(imgray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    chars_array = []
    for i in range(len(contours)):

        x,y,w,h = cv.boundingRect(contours[i])
        chars_array.append([x,y,w,h])
        if i is len(contours)-1:

            for cHr in chars_array:

                img_rectangle = img[cHr[1] : cHr[1]+cHr[3] , cHr[0] : cHr[0]+cHr[2]]
    
                cv.imwrite('./charsForSolver/img'+str(cHr[0])+'.png', img_rectangle)