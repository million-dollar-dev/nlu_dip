import numpy as np
import matplotlib.pyplot as plt
import cv2

def compute_histogram(image):
    """Tính bảng histogram trả về dưới dạng dict"""
    hist = {}
    flat = image.ravel()
    for value in flat:
        hist[value] = hist.get(value, 0) + 1
    return dict(sorted(hist.items()))

def print_histogram_table(hist, title='Histogram'):
    print(f"\n📊 {title}")
    print("Giá trị\tTần suất")
    for value, count in hist.items():
        print(f"{value}\t{count}")

def plot_histogram(image, title='Histogram'):
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=[0, 256], color='gray')
    plt.title(title)
    plt.xlabel('Giá trị pixel')
    plt.ylabel('Tần suất')
    plt.grid(True)

def main():
    # 1. Tạo ma trận ảnh xám ngẫu nhiên 5x5
    matrix = np.random.randint(0, 7, size=(5, 5), dtype=np.uint8)
    print("🖼️ Ma trận ảnh gốc (5x5):\n", matrix)

    # 2. Tính & in bảng histogram gốc
    original_hist = compute_histogram(matrix)
    print_histogram_table(original_hist, "Histogram gốc")

    # 3. Cân bằng histogram
    equalized = cv2.equalizeHist(matrix)
    print("\n🖼️ Ma trận ảnh sau cân bằng (G):\n", equalized)

    # 4. Tính & in bảng histogram sau cân bằng
    equalized_hist = compute_histogram(equalized)
    print_histogram_table(equalized_hist, "Histogram sau cân bằng")

    # 5. Vẽ biểu đồ histogram
    plot_histogram(matrix, 'Histogram trước cân bằng')
    plot_histogram(equalized, 'Histogram sau cân bằng')

    # 6. Hiển thị ảnh
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(matrix, cmap='gray', vmin=0, vmax=255)
    plt.title('Ảnh gốc')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(equalized, cmap='gray', vmin=0, vmax=255)
    plt.title('Ảnh sau cân bằng')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
