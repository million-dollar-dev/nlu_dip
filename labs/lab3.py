import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def convolution(gray, kernel):
    ksize = kernel.shape[0]
    padding = (ksize - 1) // 2
    padded = np.pad(gray, padding, mode='edge')

    out = np.zeros(gray.shape, dtype=np.float64)
    m, n = out.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row: row + ksize, col: col + ksize]
            out[row, col] = np.sum(sub * kernel)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def mean_filter_padding(gray, ksize):
    mean_filter = np.ones((ksize, ksize), dtype=np.float64) / (ksize ** 2)
    return convolution(gray, mean_filter)

def median_filter_padding(gray, ksize):
    padding = (ksize - 1) // 2
    padded = np.pad(gray, padding, mode='edge')

    out = np.zeros(gray.shape, dtype=np.float64)
    m, n = out.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row: row + ksize, col: col + ksize]
            out[row, col] = np.median(sub)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def min_filter_padding(gray, kernel):
    ksize = kernel.shape[0]
    padding = (ksize - 1) // 2
    padded = np.pad(gray, padding, mode='edge')

    out = np.zeros(gray.shape, dtype=np.float64)
    m, n = out.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row: row + ksize, col: col + ksize]
            out[row, col] = np.min(sub)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def max_filter_padding(gray, kernel):
    ksize = kernel.shape[0]
    padding = (ksize - 1) // 2
    padded = np.pad(gray, padding, mode='edge')

    out = np.zeros(gray.shape, dtype=np.float64)
    m, n = out.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row: row + ksize, col: col + ksize]
            out[row, col] = np.max(sub)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def mean_filter_no_padding(gray, ksize):
    padding = (ksize - 1) // 2
    out = np.zeros(gray.shape, dtype=np.float64)
    m, n = out.shape
    for row in range(m):
        for col in range(n):
            sub = gray[max(0, row - padding): row - padding + ksize, max(0, col - padding): col - padding + ksize]
            out[row, col] = np.mean(sub)
    return np.clip(out, 0, 255).astype(np.uint8)

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\salt_and_pepper.jpg'
img = cv.imread(file, 0)
mean_filter_img = mean_filter_padding(img, 3)
median_filter_img = median_filter_padding(img, 3)

cv.imshow('default', img)
cv.imshow('mean_filter', mean_filter_img)
cv.imshow('median_filter', median_filter_img)

cv.waitKey(0)
cv.destroyAllWindows()