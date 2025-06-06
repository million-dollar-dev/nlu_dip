import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def create_spectrum(f_uv):
    spectrum = np.log1p(np.abs(f_uv))
    min = np.min(spectrum)
    max = np.max(spectrum)
    spectrum = 255 * (spectrum - min) / (max - min)
    return np.clip(spectrum, 0, 255).astype(np.uint8)

def export_spectrum(file):
    gray = cv.imread(file, 0)
    f_uv = np.fft.fft2(gray)
    f_uv = np.fft.fftshift(f_uv)
    spectrum = create_spectrum(f_uv)
    cv.imwrite('periodic_spec.jpg', spectrum)

def create_notch_filter(shape, points, d0):
    h = np.ones(shape, dtype=np.float64)
    c_x = shape[0] // 2
    c_y = shape[1] // 2
    for d in range(len(points)):
        x = points[d][0]
        y = points[d][1]
        cv.circle(h, (x, y), d0, 0, -1)
        # x1 = x + 2 * (c_x - x)
        # y1 = y + 2 * (c_y - y)
        # cv.circle(h, (x1, y1), d0, 0, -1)
    return h

def notch_filter(gray, points, d0):
    f = np.fft.fft2(gray)
    f = np.fft.fftshift(f)

    h = create_notch_filter(img.shape, points, d0)
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

file = 'E:\\Workspace\\LEARN\\NLU\\DIP\\images\\periodic_noise.jpg'
#export_spectrum(file)
points = [[475, 387], [525, 437]]

img = cv2.imread(file, 0)
res = notch_filter(img, points, 5)

cv.imshow('original', img)
cv.imshow('notch_filter', res)

cv.waitKey(0)
cv.destroyAllWindows()


