import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def create_filter(shape, d0):
    h = np.zeros(shape, dtype=np.float64)
    m, n = shape
    for u in range(m):
        for v in range(n):
            d = np.sqrt((u - m/2) ** 2 + (v - n/2) ** 2)
            if (d > d0):
                h[u, v] = 0.0
            else:
                h[u, v] = 1.0
    return h

def low_pass_filter(gray, d):
    f = np.fft.fft2(gray)
    f = np.fft.fftshift(f)
    show_spectrum(f, 'f')

    h = create_filter(gray.shape, d)
    show_spectrum(h, 'h')

    g = f * h
    show_spectrum(g, 'g')

    g = np.fft.ifftshift(g)
    g = np.fft.ifft2(g)

    out = np.abs(g)
    return np.clip(out, 0, 255).astype(np.uint8)

def show_spectrum(freq, title="Spectrum"):
    plt.imshow(np.log1p(np.abs(freq)),
               cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\lenna.jpg'

img = cv.imread(file, 0)
low_pass_img = low_pass_filter(img, 50)

cv.imshow('default', img)
cv.imshow('low_pass_img', low_pass_img)

cv.waitKey(0)
cv.destroyAllWindows()