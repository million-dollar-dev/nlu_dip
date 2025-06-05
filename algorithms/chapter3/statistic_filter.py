import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib.pyplot import imshow


def median_filter(gray, ksize):
    padding = (ksize - 1) // 2
    padded = np.pad(gray, padding, mode='edge')
    out = np.zeros(gray.shape, dtype=np.float64)
    m, n = gray.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row:row + ksize, col:col + ksize]
            out[row, col] = np.median(sub)
    return np.clip(out, 0, 255).astype(np.uint8)

def max_filter(gray, ksize):
    padding = (ksize - 1) // 2
    padded = np.pad(gray, padding, mode='edge')
    out = np.zeros(gray.shape, dtype=np.float64)
    m, n = gray.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row:row + ksize, col:col + ksize]
            out[row, col] = np.max(sub)
    return np.clip(out, 0, 255).astype(np.uint8)

def min_filter(gray, ksize):
    padding = (ksize - 1) // 2
    padded = np.pad(gray, padding, mode='edge')
    out = np.zeros(gray.shape, dtype=np.float64)
    m, n = gray.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row:row + ksize, col:col + ksize]
            out[row, col] = np.min(sub)
    return np.clip(out, 0, 255).astype(np.uint8)

img_salt = cv.imread('E:\\Workspace\\LEARN\\NLU\\DIP\\images\\salt_noise.jpg', 0)
img_pepper = cv.imread('E:\\Workspace\\LEARN\\NLU\\DIP\\images\\pepper_noise.jpg', 0)
img_sp = cv.imread('E:\\Workspace\\LEARN\\NLU\\DIP\\images\\salt_and_pepper_2.jpg', 0)

salt_res = min_filter(img_salt, 3)
pepper_res = max_filter(img_pepper, 3)
sp_res = median_filter(img_sp, 3)

cv.imshow('origin 1', img_salt)
cv.imshow('origin 2', img_pepper)
cv.imshow('origin 3', img_sp)
cv.imshow('salt_res', salt_res)
cv.imshow('pepper_res', pepper_res)
cv.imshow('sp_res', sp_res)

cv.waitKey(0)
cv.destroyAllWindows()
