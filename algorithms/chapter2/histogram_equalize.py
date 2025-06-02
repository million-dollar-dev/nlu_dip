import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def hist_calc(gray, L):
    hist = np.zeros((L,), dtype=np.float32)
    N, M = gray.shape
    for row in range(N):
        for col in range(M):
            g = gray[row, col]
            hist[g] += 1
    std_hist = hist / gray.size
    return std_hist

def hist_equalize(gray, L):
    hist = hist_calc(gray, L)

    cdf = np.zeros_like(hist).astype(np.float32)
    for i in range(cdf.size):
        cdf[i] = np.sum(hist[:i + 1])

    cdf_min = cdf[cdf > 0].min()
    sk = np.round((cdf - cdf_min) * (L - 1) / (1 - cdf_min)).astype(np.uint8)

    N, M = gray.shape
    result = np.zeros_like(gray)
    for row in range(N):
        for col in range(M):
            result[row, col] = sk[gray[row, col]]

    print(result)
    return result

img = np.array([
    [4, 7, 4, 7, 3],
    [2, 5, 5, 3, 6],
    [7, 4, 3, 1, 3],
    [5, 6, 2, 4, 3],
    [4, 4, 5, 3, 3]
])

hist_equalize(img, 8)
