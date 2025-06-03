import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def convolve(gray, kernel):
    ksize = kernel.shape[0]
    padding = (ksize - 1) // 2
    padded = np.pad(gray, padding, mode='edge')
    out = np.zeros(gray.shape, dtype=np.float64)
    m, n = gray.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row:row + ksize, col:col + ksize]
            out[row, col] = np.sum(sub * kernel)
    return np.clip(out, 0, 255).astype(np.uint8)

def laplacian_filter(gray):
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    laplacian = convolve(gray, laplacian_kernel)
    k = 1.5
    out = gray - k * laplacian
    return np.clip(out, 0, 255).astype(np.uint8)

def laplacian_filter_edge(gray):
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    laplacian = convolve(gray, laplacian_kernel)
    return np.clip(laplacian, 0, 255).astype(np.uint8)

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\edge_detection.jpg'
img = cv.imread(file, 0)
ll_img = laplacian_filter(img)
ll_edge = laplacian_filter_edge(img)

cv.imshow('default', img)
cv.imshow('rb', ll_img)
cv.imshow('edge', ll_edge)

cv.waitKey(0)
cv.destroyAllWindows()