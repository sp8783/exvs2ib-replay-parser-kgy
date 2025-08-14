import cv2
import numpy as np

# 前処理パラメータ
PREPROCESS_SCALE = 2
PREPROCESS_ALPHA = 1.2
PREPROCESS_BETA = 10
PREPROCESS_KERNEL_SIZE = 1
PREPROCESS_SHARP_KERNEL = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])
PREPROCESS_ERODE_KERNEL_SIZE = 2
PREPROCESS_ERODE_ITER = 1

def preprocess_for_ocr(img):
    """
    OCR用に画像を前処理する関数。
    グレースケール化、拡大、コントラスト強調、二値化、ノイズ除去、シャープ処理、収縮処理を行い、
    認識精度を向上させた画像を返す。
    """
    # グレースケール化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 拡大
    gray = cv2.resize(gray, None, fx=PREPROCESS_SCALE, fy=PREPROCESS_SCALE, interpolation=cv2.INTER_CUBIC)

    # コントラスト強調
    gray = cv2.convertScaleAbs(gray, alpha=PREPROCESS_ALPHA, beta=PREPROCESS_BETA)

    # 二値化（OTSU）
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ノイズ除去
    kernel = np.ones((PREPROCESS_KERNEL_SIZE, PREPROCESS_KERNEL_SIZE), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # シャープ処理（エッジ強調）
    sharp = cv2.filter2D(cleaned, -1, PREPROCESS_SHARP_KERNEL)

    # 収縮処理（文字を細くする）
    erode_kernel = np.ones((PREPROCESS_ERODE_KERNEL_SIZE, PREPROCESS_ERODE_KERNEL_SIZE), np.uint8)
    slim = cv2.erode(sharp, erode_kernel, iterations=PREPROCESS_ERODE_ITER)
    
    return slim
