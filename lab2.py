import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# HIỆN THỰC CONTRAST STRETCHING, PHÉP BIẾN ĐỔI GAMMA, ALPHA

# Ảnh bị mờ, nhạt màu, các vùng sáng và tối không rõ ràng.
# Ảnh có độ sáng đồng đều, không có điểm thực sự sáng (gần 255) hoặc tối (gần 0).
def stretch(gray):
    min_v = gray.min()
    print(min_v)
    max_v = gray.max()
    print(max_v)
    temp = 255.0 / (max_v - min_v)
    out = (gray - min_v) * temp
    return out.astype(np.uint8)


# gamma < 1: Làm ảnh sáng hơn
# gamma > 1: làm ảnh tối hơn
def gammaTransform(gray, gamma):
    input = gray / 255.0
    out = 255 * (input ** gamma)
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

img_path_strech = "E:\\Workspace\\LEARN\\NLU\\DIP\\images\\lab2_contrast_stretching.jpg"
original_img = cv.imread(img_path_strech, 0)

stretched_img = stretch(original_img)
gamma_img = gammaTransform(original_img, 0.5)

cv.imshow('origin', original_img)
cv.imshow('stretch', stretched_img)
cv.imshow('gamma', gamma_img)
plt.figure('origin')
plt.hist(original_img.ravel(), 256)
plt.show()
plt.figure('stretched')
plt.hist(stretched_img.ravel(), 256)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows

