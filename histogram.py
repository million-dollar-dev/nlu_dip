import numpy as np

def hist_calc(gray, L):
    hist = np.zeros((L,), dtype=np.float32)
    N, M = gray.shape
    for row in range(N):
        for col in range(M):
            g = gray[row, col]
            hist[g] += 1
    stdHist = hist / gray.size
    return stdHist

def equalize_histogram(gray, L):
    # 1. Tính histogram chuẩn hóa
    hist = hist_calc(gray, L)
    print("📊 Histogram chuẩn hóa:")
    print(hist)

    # 2. Tính CDF
    cdf = np.cumsum(hist)
    print("\n📈 CDF:")
    print(cdf)

    # 3. Xác định Cmin (khác 0 đầu tiên)
    cdf_min = cdf[cdf > 0].min()
    print(f"\n🔍 CDF min (≠ 0): {cdf_min:.4f}")

    # 4. Tính các giá trị Sk bằng công thức: Sk = round((Ck - Cmin)*(L-1)/(1 - Cmin))
    sk = np.round((cdf - cdf_min) * (L - 1) / (1 - cdf_min)).astype(np.uint8)
    print("\n🎯 Mảng Sk (mapping):")
    print(sk)

    # 5. Tạo ảnh cân bằng mới
    N, M = gray.shape
    result = np.zeros_like(gray)
    for row in range(N):
        for col in range(M):
            result[row, col] = sk[gray[row, col]]

    print("\n🖼️ Ảnh sau cân bằng histogram:")
    print(result)
    return result


gray = np.array([
    [6.5,9.1,4.4,6.2,3.1],
    [3.4,7.1,11.2,8.4,6.5],
    [7.7,9,4.2,5.1,5.5],
    [3.9,10.9,8,7,5],
    [10,6,8,9,5.4]
], dtype=np.uint8)

L = 16  # ảnh xám 4-bit

equalized_img = equalize_histogram(gray, L)
