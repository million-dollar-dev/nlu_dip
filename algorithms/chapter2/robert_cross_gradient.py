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
            out[row, col] = np.sum(sub * kernel)
    return out

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

    g = g_x_convolved + g_y_convolved

    out = gray + g
    return np.clip(out, 0, 255).astype(np.uint8)

def robert_cross_edge(gray):
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

    g = g_x_convolved + g_y_convolved

    return np.clip(g, 0, 255).astype(np.uint8)

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\edge_detection.jpg'
img = cv.imread(file, 0)
rb_img = robert_cross_filter(img)
rb_edge = robert_cross_edge(rb_img)

cv.imshow('default', img)
cv.imshow('rb', rb_img)
cv.imshow('edge', rb_edge)

cv.waitKey(0)
cv.destroyAllWindows()