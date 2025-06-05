import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def create_filter(shape, d0, n):
    h = np.zeros(shape, dtype=np.float64)
    m, k = shape
    for u in range(m):
        for v in range(k):
            d = np.sqrt((u - m/2) ** 2 + (v - k/2) ** 2)
            d = max(d, 1e-5)
            h[u, v] = 1 / (1 + (d0/d) ** (2*n))
    return h

def high_pass_filter(img, d0, n):
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    show_spectrum(f, 'f')

    h = create_filter(f.shape, d0, n)
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
high_pass_img = high_pass_filter(img, 30, 2)

cv.imshow('default', img)
cv.imshow('high_pass_img', high_pass_img)

cv.waitKey(0)
cv.destroyAllWindows()