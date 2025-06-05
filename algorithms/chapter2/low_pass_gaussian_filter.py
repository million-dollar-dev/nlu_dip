import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def create_gaussian_filter(shape, d0):
    h = np.zeros(shape, dtype=np.float64)
    m, n = shape
    for u in range(m):
        for v in range(n):
            d = np.sqrt((u - m/2) ** 2 + (v - n/2) ** 2)
            h[u, v] = np.exp(-d * d / (2 * d0 * d0))
    return h

def low_pass_filter(gray, d):
    #B1
    f = np.fft.fft2(gray)
    f = np.fft.fftshift(f)
    show_spectrum(f, title="Spectrum")
    #B2
    h = create_gaussian_filter(gray.shape, d)
    #B3
    g = f * h
    show_spectrum(g, title="Spectrum")
    #B4
    g = np.fft.ifftshift(g)
    g = np.fft.ifft2(g)
    out = np.abs(g)
    return np.clip(out, 0, 255).astype(np.uint8)

def show_spectrum(freq, title="Spectrum"):
    magnitude = 20 * np.log(np.abs(freq) + 1)
    plt.figure(figsize=(6, 5))
    plt.imshow(magnitude, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\lenna.jpg'

img = cv.imread(file, 0)
low_pass_img = low_pass_filter(img, 10)

cv.imshow('default', img)
cv.imshow('low_pass_img', low_pass_img)

cv.waitKey(0)
cv.destroyAllWindows()

