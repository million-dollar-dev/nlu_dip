import numpy as np

def calc_hist(gray, L):
    hist = np.zeros((L,), dtype=np.float32)
    m, n = gray.shape
    for row in range(m):
        for col in range(n):
            hist[hist[row,col]] += 1
    return hist / gray.size

def hist_equalize(gray, L):
    hist = calc_hist(gray, L)

    cdf = np.zeros(hist, dtype=np.float32)
    for i in range(cdf.size):
        cdf[i] = np.sum(hist[: i + 1])

    c_min = cdf[cdf > 0].min()
    c_max = 1

    sk = np.round((cdf - c_min) * (L - 1) / (c_max - c_min)).astype(np.unit8)

    m, n = gray.shape
    res = np.zeros((m, n), dtype=np.uint8)
    for row in range(m):
        for col in range(n):
            res[row, col] = sk[gray[row,col]]

    print(res)
    return res