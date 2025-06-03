import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def convolve(gray, kernel):
    ksize = kernel.shape[0]
    padding = (ksize - 1) // 2
    padded = np.pad(gray, padding, mode='edge')
    out = np.zeros(gray.shape, dtype=np.float64)

    m,n = gray.shape
    for row in range(m):
        for col in range(n):
            sub = padded[row:row+ksize, col:col+ksize]
            out = np.sum(sub * kernel)
    return np.clip(out, 0, 255).astype(np.uint8)

def robert_cross_filter(gray):
    g_x = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ])
    g_y = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ])

    g_x_convolved = convolve(gray, g_x)
    g_y_convolved = convolve(gray, g_y)

    robert_cross = g_x_convolved + g_y_convolved

    out = gray + robert_cross
    return np.clip(out, 0, 255).astype(np.uint8)