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

def sobel_filter(gray):
    g_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    g_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    g_x_convolved = convolve(gray, g_x)
    g_y_convolved = convolve(gray, g_y)

    sobel = g_x_convolved + g_y_convolved
    out = gray + sobel
    return np.clip(out, 0, 255).astype(np.uint8)

def sobel_filter_edge(gray):
    g_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    g_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    g_x_convolved = convolve(gray, g_x)
    g_y_convolved = convolve(gray, g_y)

    sobel = g_x_convolved + g_y_convolved
    return np.clip(sobel, 0, 255).astype(np.uint8)

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\edge_detection.jpg'
img = cv.imread(file, 0)
sb_img = sobel_filter(img)
sb_edge = sobel_filter_edge(img)

cv.imshow('default', img)
cv.imshow('rb', sb_img)
cv.imshow('edge', sb_edge)

cv.waitKey(0)
cv.destroyAllWindows()