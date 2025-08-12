import cv2
import numpy as np

def preprocess_for_ocr(img):
    # グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 拡大（文字を大きくして認識しやすく）
    scale = 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # コントラスト強調
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    
    # 二値化（OTSU）
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ノイズ除去
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return cleaned
