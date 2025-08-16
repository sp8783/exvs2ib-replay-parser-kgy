import cv2
import numpy as np

PREPROCESS_SCALE = 4
PREPROCESS_ALPHA = 1
PREPROCESS_BETA = 10
PREPROCESS_SHARP_KERNEL = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])
PREPROCESS_THRESH = 127

def preprocess_for_ocr(img):
    """
    OCR用に画像を前処理する関数。
    グレースケール化、反転、拡大、コントラスト強調、二値化を行い、
    認識精度を向上させた画像を返す。
    """
    # グレースケール化
    target = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 反転（白文字→黒文字）
    target = cv2.bitwise_not(target)

    # 拡大
    target = cv2.resize(target, None, fx=PREPROCESS_SCALE, fy=PREPROCESS_SCALE, interpolation=cv2.INTER_CUBIC)

    # コントラスト強調
    target = cv2.convertScaleAbs(target, alpha=PREPROCESS_ALPHA, beta=PREPROCESS_BETA)

    # 二値化
    _, target = cv2.threshold(target, PREPROCESS_THRESH, 255, cv2.THRESH_BINARY)

    return target
