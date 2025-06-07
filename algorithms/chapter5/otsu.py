import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def threshold(img, thres):
    gray = img.copy()
    gray[gray < thres] = 0
    gray[gray > thres] = 255
    return gray

def calc_histogram(gray, L):
    hist = np.zeros((L,), dtype=np.float64)
    m, n = gray.shape
    for row in range(m):
        for col in range(n):
            hist[gray[row, col]] += 1
    return hist

def find_otsu(gray, L):
    hist = calc_histogram(gray, L)
    indexes = np.arange(L)
    found_T = 0
    var0 = 0
    for t in range(L):
        total1 = np.sum(hist[:t + 1]) + 1e-6
        total2 = np.sum(hist[t + 1:]) + 1e-6

        w1 = total1 / gray.size
        w2 = 1 - w1

        m1 = np.sum(hist[:t + 1] * indexes[:t + 1]) / total1
        m2 = np.sum(hist[t + 1:] * indexes[t + 1:]) / total2

        var = w1 * w2 * (m1 - m2) ** 2
        if var > var0:
            found_T = t
            var0 = var
    return found_T

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\otsu.jpg'

img = cv.imread(file, 0)
thres = find_otsu(img, 256)
print(thres)
res = threshold(img, thres)

cv.imshow('original', img)
cv.imshow('otsu', res)

cv.waitKey(0)
cv.destroyAllWindows()