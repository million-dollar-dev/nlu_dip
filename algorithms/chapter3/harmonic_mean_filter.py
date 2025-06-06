from statistics import harmonic_mean

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def harmonic_mean_filter(gray, ksize):
    out = np.zeros(gray.shape, dtype=np.float64)
    padding = ksize // 2
    padded = np.pad(gray, padding, mode='edge')
    m, n = gray.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row:row+ksize, col:col+ksize]
            sub = np.where(sub == 0, 1e-6, sub)
            out[row, col] = (ksize * ksize) / (np.sum(1 / sub))
    return np.clip(out, 0, 255).astype(np.uint8)

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\salt_noise.jpg'

img = cv.imread(file, 0)
hm_img = harmonic_mean_filter(img, 5)

cv.imshow('original', img)
cv.imshow('harmonic mean', hm_img)

cv.waitKey(0)
cv.destroyAllWindows()