import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def stretching(gray):
    f_min = np.min(gray)
    f_max = np.max(gray)
    out = 255 * (gray - f_min) / (f_max - f_min)
    return out.astype(np.uint8)

def gamma_transform(gray, gamma):
    out = gray / 255.0
    out = 255 * (out ** gamma)
    return np.clip(out, 0, 255).astype(np.uint8)

def alpha_transform(gray, alpha, belta):
    out = alpha * gray + belta
    return np.clip(out, 0, 255).astype(np.uint8)

