import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def alpa_transform(gray, alpha, beta):
    out = alpha * gray + beta
    return np.clip(out, 0, 255).astype(np.uint8)

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\lab2_contrast_stretching.jpg'

img = cv.imread(file, 0)
alpla_img = alpa_transform(img, 1, 5)

cv.imshow('default', img)
cv.imshow('alpla_img', alpla_img)

cv.waitKey(0)
cv.destroyAllWindows()