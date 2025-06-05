import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def create_gaussian_filter(shape, d):
    h = np.zeros(shape, dtype=float64)
    m, n = shape
    for u in range(m):
        for v in range(n):
            d = np.sqrt((u + m/2) ** 2 + (v + n/2) ** 2)
            h[u, v] = np.exp(-d ** d / 2 * d0 * d0)
    return h

def f_low_pass(gray, d):
    f = np.fft.fft2(gray)
    f = np.fft.fftshift(f)

    h = create_gaussian_filter(f.shape, d)

    g = f * h

    g = np.fft.ifftshift(g)
    g = np.fft.ifft2(g)

    out = np.abs(g)
    return np.clip(out, 0, 255).astype(np.uint8)

def f_high_pass(gray, d):
    f = np.fft.fft2(gray)
    f = np.fft.fftshift(f)

    h = create_gaussian_filter(f.shape, d)
    h = 1 - h

    g = f * h
    g = np.fft.ifftshift(g)
    g = np.fft.ifft2(g)

    out = np.abs(g)
    return np.clip(out, 0, 255).astype(np.uint8)

def unsharp_masking(gray, k, d):
    f = np.fft.fft2(gray)
    f = np.fft.fftshift(f)

    h = create_gaussian_filter(f.shape, d)
    flp = f * h
    flp = np.fft.ifftshift(flp)
    flp = np.fft.ifft2(flp)

    flp = np.abs(flp)
    g_mask = gray - flp

    out = gray + k * g_mask
    return np.clip(out, 0, 255).astype(np.uint8)

