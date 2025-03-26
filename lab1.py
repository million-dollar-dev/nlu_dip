import numpy as np
import cv2 as cv2

img_path = "E:\\Workspace\\LEARN\\NLU\\DIP\\images\\image.jpg"
original_image = cv2.imread(img_path)

print(original_image.shape)
print(original_image.size)
print(original_image.ndim)
print(original_image.dtype)

cropped_img = original_image[355:465, 275:408]

img_only_blue = original_image.copy()
img_only_blue[:, :, 1] = 0
img_only_blue[:, :, 2] = 0

img_without_red = original_image.copy()
img_without_red[:, :, 2] = 0

def red_enhancer(img, flat):
    img[:, :, 1] = np.clip(0 + flat, 0, 255).astype(np.uint8)
    return img

img_enhancer = red_enhancer(original_image, 1.5)

cv2.imshow('original', original_image)
cv2.imshow('cropped', cropped_img)
cv2.imshow('only blue', img_only_blue)
cv2.imshow('without red', img_without_red)
cv2.imshow('enhancer', img_enhancer)
cv2.waitKey(0)
cv2.destroyAllWindows