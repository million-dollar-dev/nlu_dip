import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def convolution(gray, kernel):
    ksize = kernel.shape[0]
    padding = (ksize - 1) // 2
    gray_padded = np.pad(gray, padding, mode='edge')
    out = np.zeros(gray.shape, dtype=np.float64)

    m,n = gray.shape
    for row in range(m):
        for col in range(n):
            sub = gray_padded[row:row+ksize, col:col+ksize]
            out[row, col] = np.sum(sub * kernel)
    return np.clip(out, 0, 255).astype(np.uint8)

def mean_filter_padding(gray, ksize):
    mean_filter = np.ones((ksize, ksize), dtype=np.float64) / (ksize ** 2)
    return convolution(gray, mean_filter)

def mean_filter_no_padding(gray, ksize):
    padding = (ksize - 1) // 2
    out = np.zeros(gray.shape, dtype=np.float32)
    m, n = gray.shape
    for row in range(m):
        for col in range(n):
            sub = gray[max(row - padding, 0):row - padding + ksize, max(col - padding, 0):col - padding + ksize]
            out[row, col] = np.mean(sub)
    return np.clip(out, 0, 255).astype(np.uint8)

gray = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]],
    dtype=np.uint8)

kernel = np.ones((3, 3), dtype=np.float32) / 9
result_cv2 = cv.filter2D(gray.astype(np.float32), -1, kernel)
print("Kết quả từ OpenCV:\n", result_cv2)

print("Kết quả tay:\n", mean_filter_no_padding(gray, 3))
