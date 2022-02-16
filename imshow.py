import cv2
import numpy as np
import sys
import argparse
from glob import glob
import os
from matplotlib import pyplot as plt

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--img_path', type= str, default = './data/train/images')
    p.add_argument('--mask_path', type= str, default= './data/train/masks')
    p.add_argument('--contrast_alpha',type= float, default= 0.8)

    config = p.parse_args()
    return config

def imread(img, mask, code= cv2.IMREAD_GRAYSCALE):
    src= cv2.imread(img, code) # source image (brain image)
    m = cv2.imread(mask, cv2.IMREAD_GRAYSCALE) # mask image
    h,w = src.shape[:2]  
    dst = np.zeros((h,w), np.uint8) # RoI
    if src is None or m is None:
        print('Image load failed')
        sys.exit()

    dst[m > 0] = src[m > 0]

    # obtain a histogram
    hist =cv2.calcHist([src], [0], None, [256], [0,256])
    #  cv2,calcHist([src], [0], None, [256], [0, 256])
    hist_roi = cv2.calcHist([dst], [0], None, [256], [0, 256])

    return src, m, dst, hist, hist_roi

def getGrayHistImage(hist):
    imgHist = np.full((100, 256), 255, dtype = np.uint8) # white background
    
    histMax = np.max(hist)
    for x in range(256):
        pt1 = (x, 100) # width, height
        pt2 = (x, 100 - int(hist[x, 0]*100 / histMax)) # notice that subtraction yields higher position in height.
        cv2.line(imgHist, pt1, pt2, 0)
    
    return imgHist

def onChange(pos):

    global img, canny, modified

    thr1 = cv2.getTrackbarPos('Thr1', 'Edge')
    thr2 = cv2.getTrackbarPos('Thr2', 'Edge')
    canny = cv2.Canny(img, thr1, thr2)
    cv2.imshow('Edge', canny)

def myContrast(img, alpha):
    # the larger the alpha is, the larger the contrast effect is.
    # if alpha < 1: contrast effect shrinks
    # else: contrast effect gets larger.
    vals, counts = np.unique(img, return_counts= True) 
    idx = np.argmax(counts)
    # vals[idx] = mode

    modified = (1+alpha)*img - alpha * vals[idx]  # externally dividing
    modified = np.clip(modified, 0, 255).astype(np.uint8)
    # print(f'mode: {vals[idx]}')
    return modified

def main(config):
    
    global img, mask, roi, canny, modified

    imgs = glob(os.path.join(config.img_path, '*.png'))
    masks = glob(os.path.join(config.mask_path,'*.png'))

    cv2.namedWindow('Edge')
    cv2.createTrackbar('Thr1', 'Edge', 100, 255, onChange)
    cv2.createTrackbar('Thr2', 'Edge', 200, 255, onChange)

    for i in range(len(imgs)):
        img, mask, roi, hist, hist_roi = imread(imgs[i], masks[i])

        modified = myContrast(img, config.contrast_alpha)
        # plot histogram
        cv2.imshow('Image', img)
        cv2.imshow('Mask', mask)
        cv2.imshow('RoI', roi)
        canny = cv2.Canny(img, 100, 200)
        cv2.imshow('Edge', canny)
        cv2.imshow('Modified', modified)

        # histImg = getGrayHistImage(hist)
        # histImg_roi = getGrayHistImage(hist_roi)
        # cv2.imshow('histImg', histImg)
        # cv2.imshow('RoIHistImg', histImg_roi)

        plt.figure(figsize=(15,15))
        plt.plot(hist, color= 'r', label= 'original image')
        plt.plot(hist_roi, color= 'g', label= 'RoI')
        plt.legend()
        plt.show()

        angle = 0
        while True:            
            query = cv2.waitKey()
            if query == 27:
                break
            
            if query == ord('r'):
                # apply transform
                # rotation
                h, w = img.shape[:2]
                center = (w//2, h//2)
                angle = (angle+10)%360 # rotation degree
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rot_img = cv2.warpAffine(img, M, (w,h))
                rot_mask = cv2.warpAffine(mask, M, (w,h))
                rot_roi = cv2.warpAffine(roi, M, (w,h))
                rot_modified = cv2.warpAffine(modified, M, (w,h))
                cv2.imshow('Image', rot_img)
                cv2.imshow('Mask', rot_mask)
                cv2.imshow('RoI', rot_roi)
                cv2.imshow('Modified', rot_modified)
            
            elif query == ord('l'):
                # apply transform
                # rotation
                h, w = img.shape[:2]
                center = (w//2, h//2)
                angle = (angle-10)%360 # rotation degree
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rot_img = cv2.warpAffine(img, M, (w,h))
                rot_mask = cv2.warpAffine(mask, M, (w,h))
                rot_roi = cv2.warpAffine(roi, M, (w,h))
                rot_modified = cv2.warpAffine(modified, M, (w,h))
                cv2.imshow('Image', rot_img)
                cv2.imshow('Mask', rot_mask)
                cv2.imshow('RoI', rot_roi)
                cv2.imshow('Modified', rot_modified)
        
        query = cv2.waitKey()
        if query == 27:
            break

if __name__ == '__main__':
    config = argparser()
    main(config)