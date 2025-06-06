import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def arithmetic_mean_filter(gray, ksize):
    out = np.zeros(gray.shape, dtype=np.float64)
    padding = ksize // 2
    padded = np.pad(gray, padding, mode='edge')
    m, n = gray.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row:row+ksize, col:col+ksize]
            out[row, col] = np.mean(sub)

    return np.clip(out, 0, 255).astype(np.uint8)

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\lenna.jpg'

img = cv.imread(file, 0)
am_img = arithmetic_mean_filter(img, 3)

cv.imshow('original', img)
cv.imshow('arithmetic mean', am_img)

cv.waitKey(0)
cv.destroyAllWindows()