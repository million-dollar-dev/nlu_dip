import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def contra_harmonic(gray, ksize, Q):
    out = np.zeros(gray.shape, dtype=np.float64)
    padding = (ksize - 1) // 2
    padded = np.pad(gray, padding, mode='edge')
    m, n = gray.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row:row + ksize, col:col + ksize].astype(np.float64)
            epsilon = 1e-8
            a = np.sum((sub + epsilon) ** (Q + 1))
            b = np.sum((sub + epsilon) ** Q)
            out[row, col] = a / b if b != 0 else 0
    return np.clip(out, 0, 255).astype(np.uint8)

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\salt_noise.jpg'

img = cv.imread(file, 0)
hm_img = contra_harmonic(img, 3, -2)

cv.imshow('original', img)
cv.imshow('contra harmonic mean', hm_img)

cv.waitKey(0)
cv.destroyAllWindows()