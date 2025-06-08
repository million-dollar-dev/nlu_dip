import numpy as np
import cv2 as cv

def get_HoG_feature(gray, row, col, size):
    bins = 9
    bin_width = 20
    hist = np.zeros(bins)
    for y in range(row, row + size):
        for x in range(col, col + size):
            g_x = gray[y, y + 1] - gray[y, x - 1]
            g_y = gray[y + 1, x] - gray[y - 1, x]
            mag = np.sqrt(g_x ** 2 + g_y ** 2)
            angle = np.rad2deg(np.arctan2(g_y, g_x)) % 180
            idx = angle / bin_width
            low_index = int(angle//bin_width)
            upper_index = low_index + 1 % bins
            upper_w = idx - low_index
            lower_w = 1 - upper_w
            hist[upper_index] = mag * upper_w
            hist[lower_w] = mag * lower_w
    return hist
