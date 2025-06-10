import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def amf(gray, row, col, k_max, m, n):
    ksize = 3
    while (ksize <= k_max):
        padding = ksize // 2
        sub = gray[
              max(0, row - padding): min(m, row + padding + 1),
              max(0, col - padding): min(n, col + padding + 1)
              ]
        z_xy = gray[row, col]
        z_min = np.min(sub)
        z_med = np.median(sub)
        z_max = np.max(sub)
        if z_min < z_med < z_max:
            if z_min < z_xy < z_max:
                return z_xy
            else:
                return z_med
        else:
            ksize += 2
    return z_med

def amf_filter(gray, k_max):
    out = np.zeros(gray.shape, dtype=gray.dtype)
    m, n = gray.shape
    for row in range(m):
        for col in range(n):
            out[row, col] = amf(gray, row, col, k_max, m, n)
    return out

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\salt_and_pepper_2.jpg'

img = cv.imread(file, 0)
amf_img = amf_filter(img, 7)

cv.imshow('original', img)
cv.imshow('amf', amf_img)

cv.waitKey(0)
cv.destroyAllWindows()


