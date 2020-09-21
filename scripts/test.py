import os
import cv2

import numpy as np 
import segmentation_models as sm 

from time import time

def get_max_area_mask(mask):

    mask = mask.astype(dtype=np.uint8)
    _,thresh = cv2.threshold(mask,127,255,0)
    
    contours,_ = cv2.findContours(thresh, 1, 2)
        
    contour = sorted(contours, key=cv2.contourArea)[-1]

    return contour

def model_loader(model_name, path_to_weigth):
    model = sm.Unet(model_name)

    model.load_weights(path_to_weigth)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

    return model

def predict_mask(img, model):

    img_for_net = np.reshape(img, (1,*img.shape)).astype(dtype=np.float16)
    
    pred = model.predict(img_for_net)
    mask = pred[0]
    mask[mask < 0.5] = 0
    mask[mask != 0] = 255

    contour = get_max_area_mask(mask)

    mask = np.zeros(mask.shape, np.uint8)

    cv2.fillPoly(mask, pts =[contour], color=(255,255,255))
    return mask

def main():

    model_name = "resnet50"

    model = model_loader(model_name, "../models/fcn_best.h5")

    for name in os.listdir("../data/test/imgs"):

        img = cv2.imread(f"../data/test/imgs/{name}")

        start = time()
        plant_mask = predict_mask(img, model)
        print(time() - start)
        
        res = cv2.bitwise_and(img, img,  mask=plant_mask)
        cv2.imshow('liquid', img)
        cv2.imshow('img', res)
        cv2.waitKey(0)


main()