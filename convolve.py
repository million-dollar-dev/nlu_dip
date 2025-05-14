import numpy as np
from scipy.ndimage import median_filter

# Ảnh gốc
f = np.array([
    [6, 9, 4, 6, 3],
    [3, 7,11, 8, 6],
    [7, 9, 4, 5, 5],
    [3,10, 8, 7, 5],
    [10,6, 8, 9, 5]
])

# Lọc trung vị 3x3 với zero-padding
G = median_filter(f, size=3, mode='constant', cval=0)
print(G)