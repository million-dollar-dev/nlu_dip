import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def gamma_transform(gray, gamma):
    input = gray / 255.0
    out = 255 * (input ** gamma)
    return np.clip(out, 0, 255).astype(np.uint8)

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\lab2_high_contrast.jpg'

img = cv.imread(file, 0)
out = gamma_transform(img, 0.5)

cv.imshow('default', img)
cv.imshow('gamma_transform', out)

cv.waitKey()
cv.destroyAllWindows()
