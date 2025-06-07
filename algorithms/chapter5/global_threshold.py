import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def threshold(img, thres):
    gray = img.copy()
    gray[gray < thres] = 0
    gray[gray > thres] = 255
    return gray

def find_global_threshold(gray, thres):
    T0 = np.mean(gray)
    T = T0
    while True:
        g_1 = gray[gray > T0]
        g_2 = gray[gray <= T0]
        m_1 = np.mean(g_1)
        m_2 = np.mean(g_2)
        T = (m_1 + m_2) / 2
        delta_T = np.abs(T - T0)
        T0 = T
        if delta_T > thres:
            break
    return T

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\otsu.jpg'

img = cv.imread(file, 0)
thres = find_global_threshold(img, 0.01)
print(thres)
res = threshold(img, thres)

cv.imshow('original', img)
cv.imshow('global', res)

cv.waitKey(0)
cv.destroyAllWindows()