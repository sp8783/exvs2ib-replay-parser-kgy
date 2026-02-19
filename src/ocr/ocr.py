import numpy as np
from src.core.config import Config
from src.util.image import get_roi, get_player_unit_roi_from_ratio
from .preprocess import preprocess_for_ocr
from .matcher import Matcher
import pytesseract


def ocr_roi(processed_img: np.ndarray, config: Config) -> str:
    """
    前処理済み画像からOCR生テキストを抽出して返す関数。
    """
    ocr_conf = config.get("ocr", default={})
    lang = ocr_conf.get("lang", "jpn+eng")
    psm = ocr_conf.get("psm", 7)
    return pytesseract.image_to_string(processed_img, lang=lang, config=f"--psm {psm}").strip()


def preprocess_ocr_text(text: str) -> str:
    """
    OCR結果の前処理（半角スペース除去＋数字正規化）を行う関数。
    """
    if not text:
        return text
    text = text.replace(" ", "")
    replacements = {
        "①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5",
        "⑥": "6", "⑦": "7", "⑧": "8", "⑨": "9", "⑩": "10",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def get_preprocessed_text_from_roi(img: np.ndarray, roi: tuple[int, int, int, int], config: Config) -> str:
    """
    指定画像とROIから前処理済みテキストを返す関数。
    """
    processed = preprocess_for_ocr(get_roi(img, roi), config)
    ocr_text = ocr_roi(processed, config)
    preprocessed_text = preprocess_ocr_text(ocr_text)
    return preprocessed_text


def ocr_on_matching_regions(img: np.ndarray, config: Config, matcher: Matcher) -> dict[str, str | None]:
    """
    マッチング画面画像から各領域を切り出し、OCR・候補マッチングを行い、
    プレイヤー名・機体名を辞書で返す。
    領域ごと（プレイヤー名・機体名）に適切なスコアリング関数を利用。
    """
    if img is None:
        raise FileNotFoundError("Image not found")

    results = {}
    roi_config = config.get("roi", "player_unit", default={})
    regions = get_player_unit_roi_from_ratio(img, roi_config)

    for key, roi in regions.items():
        preprocessed_text = get_preprocessed_text_from_roi(img, roi, config)
        results[key] = matcher.match_text(key, preprocessed_text)

    return results
