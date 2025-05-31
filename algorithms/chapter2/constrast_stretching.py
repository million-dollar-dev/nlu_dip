import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def constrast_stretching(gray):
    f_min = np.min(gray)
    f_max = np.max(gray)
    print(f_min, f_max)
    temp = 255 / (f_max - f_min)
    out = (gray - f_min) * temp
    return out.astype(np.uint8)

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\image.jpg'
img = cv.imread(file, 0)

stretching_img = constrast_stretching(img)

cv.imshow('Default', img)
cv.imshow('Stretched', stretching_img)

plt.hist(img.ravel(), bins=256, range=(0, 255))
plt.show()
plt.hist(stretching_img.ravel(), bins=256, range=(0, 255))
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()