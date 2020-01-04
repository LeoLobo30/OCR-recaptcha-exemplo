from keras.models import Sequential,load_model
import cv2 as cv
import numpy as np
import os, glob, re

labels = ['1','2','3','4','5','6','7',
    '8','9','a','b','c','d','e','f','g',
    'h','i','j','k','l','m','n','p',
    'q','r','s','t','u','v','w','x','y','z']


model = load_model('model.h5')

def deleteImages():
    files = glob.glob('charsForSolver/*')

    for f in files:
        os.remove(f)


def loadImagensAndGetChars():    


    paths = glob.glob('charsForSolver/img*.png')

    list_solve = []
    list_max_numbers = []

    for path in paths:
        valueSort = re.search(r'\d+',path)
        valueSort = int(valueSort.group())
        if valueSort > 99:
            list_max_numbers.append(path)
        else:
            list_solve.insert(valueSort,path)

    list_solve.extend(list_max_numbers)


    list_solve = [cv.imread(pos,1) for pos in list_solve]

    list_solve = [img for img in list_solve if img.shape > (40,14) and img.shape < (60,23)]

    deleteImages()

    return list_solve

def prepareImagePredict(imagem_list):

    list_return = []
    
    for img in imagem_list:        

        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        img = cv.resize(img,(23,60))

        img = cv.normalize(img,None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

        img = np.expand_dims(img, 0)

        img = np.expand_dims(img, 3)

        list_return.append(img)
    
    return list_return

def classifyPredict(image_list):

    list_temp_labels_solver = []

    for img in image_list:


        prevision = model.predict(img)

        max_prevision = np.argmax(prevision)

        list_temp_labels_solver.append(labels[max_prevision]) 
    
    return list_temp_labels_solver

def solver():
    return classifyPredict(prepareImagePredict(loadImagensAndGetChars()))