import cv2
import numpy as np
from src.core.config import Config

# 設定ファイルから前処理パラメータを取得
_config = Config("config/config.yaml")
_preprocess = _config.get("preprocess", default={})
PREPROCESS_SCALE = _preprocess.get("scale", 4)
PREPROCESS_ALPHA = _preprocess.get("alpha", 1)
PREPROCESS_BETA = _preprocess.get("beta", 10)
PREPROCESS_SHARP_KERNEL = np.array(_preprocess.get("sharp_kernel", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
PREPROCESS_THRESH = _preprocess.get("thresh", 127)

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

    # シャープ化
    target = cv2.filter2D(target, -1, PREPROCESS_SHARP_KERNEL)

    # 二値化
    _, target = cv2.threshold(target, PREPROCESS_THRESH, 255, cv2.THRESH_BINARY)

    return target
