import cv2
import numpy as np
from src.core.config import Config


def preprocess_for_ocr(img: np.ndarray, config: Config) -> np.ndarray:
    """
    OCR用に画像を前処理する関数。
    グレースケール化、反転、拡大、コントラスト強調、二値化を行い、
    認識精度を向上させた画像を返す。
    """
    preprocess = config.get("preprocess", default={})
    scale = preprocess.get("scale", 4)
    alpha = preprocess.get("alpha", 1)
    beta = preprocess.get("beta", 10)
    kernel = np.array(preprocess.get("sharp_kernel", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    thresh = preprocess.get("thresh", 127)

    # グレースケール化
    target = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 反転（白文字→黒文字）
    target = cv2.bitwise_not(target)

    # 拡大
    target = cv2.resize(target, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # コントラスト強調
    target = cv2.convertScaleAbs(target, alpha=alpha, beta=beta)

    # シャープ化
    target = cv2.filter2D(target, -1, kernel)

    # 二値化
    _, target = cv2.threshold(target, thresh, 255, cv2.THRESH_BINARY)

    return target
