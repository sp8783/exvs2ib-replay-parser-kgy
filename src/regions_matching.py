import pytesseract
import csv
from rapidfuzz import process
from .preprocess import preprocess_for_ocr
from .ocr_matching_scorer import matching_scorer_for_unit_name, matching_scorer_for_player_name

PLAYER_UNIT_REGIONS_RATIO = {
    "player1_name": (0.045, 0.693, 0.18, 0.718),
    "player1_unit": (0.035, 0.738, 0.22, 0.762),
    "player2_name": (0.27, 0.65, 0.39, 0.67),
    "player2_unit": (0.265, 0.69, 0.43, 0.71),
    "player3_name": (0.565, 0.65, 0.68, 0.67),
    "player3_unit": (0.575, 0.69, 0.74, 0.71),
    "player4_name": (0.77, 0.693, 0.90, 0.718),
    "player4_unit": (0.785, 0.738, 0.97, 0.762),
}

def load_candidates(path):
    """
    指定したCSVファイルから候補リスト（1列目）を読み込んでリストで返す。
    """
    with open(path, encoding="utf-8") as f:
        return [row[0] for row in csv.reader(f) if row]

player_candidates = load_candidates("data/player_names.csv")
unit_candidates = load_candidates("data/unit_names.csv")

def ocr_on_matching_regions(img):
    """
    マッチング画面画像から各領域を切り出し、OCR・候補マッチングを行い、
    プレイヤー名・機体名を辞書で返す。
    領域ごと（プレイヤー名・機体名）に適切なスコアリング関数を利用。
    """
    frame = img
    if frame is None:
        raise FileNotFoundError(f"Image not found at {img}")
    results = {}
    regions = get_player_unit_roi_from_ratio(frame)
    for key, roi in regions.items():
        preprocessed_text = get_preprocessed_text_from_roi(frame, roi)
        results[key] = match_text(key, preprocessed_text)
    return results

def get_player_unit_roi_from_ratio(img):
    """
    画像サイズに応じて割合から領域座標を計算して返す関数。
    プレイヤー名・機体名の領域座標を辞書で返す。
    """
    h, w = img.shape[:2]
    regions = {}
    for key, (x1r, y1r, x2r, y2r) in PLAYER_UNIT_REGIONS_RATIO.items():
        x1 = int(w * x1r)
        y1 = int(h * y1r)
        x2 = int(w * x2r)
        y2 = int(h * y2r)
        regions[key] = (x1, y1, x2, y2)
    return regions

def get_preprocessed_text_from_roi(img, roi):
    """
    指定画像とROIから前処理済みテキストを返す関数。
    """
    processed = preprocess_for_ocr(get_roi(img, roi))
    ocr_text = ocr_roi(processed)
    preprocessed_text = preprocess_ocr_text(ocr_text)
    return preprocessed_text

def match_text(key, text):
    """
    領域種別（name/unit）に応じて、テキストを候補リストとマッチングして返す関数。
    """
    if "name" in key:
        return match_candidate(text, player_candidates, matching_scorer_for_player_name)
    else:
        return match_candidate(text, unit_candidates, matching_scorer_for_unit_name)

def get_roi(img, roi):
    """
    指定したROI座標から画像領域を切り出して返す関数。
    """
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2]

def ocr_roi(processed_img):
    """
    前処理済み画像からOCR生テキストを抽出して返す関数。
    """
    return pytesseract.image_to_string(processed_img, lang="jpn+eng", config="--psm 7").strip()

def preprocess_ocr_text(text):
    """
    OCR結果の前処理（半角スペース除去＋数字正規化）を行う関数。
    """
    if not text:
        return text
    # 半角スペース除去
    text = text.replace(" ", "")
    # 数字正規化
    replacements = {
        "①": "1",
        "②": "2",
        "③": "3",
        "④": "4",
        "⑤": "5",
        "⑥": "6",
        "⑦": "7",
        "⑧": "8",
        "⑨": "9",
        "⑩": "10",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def match_candidate(text, candidates, scorer):
    """
    OCRで得たテキストを候補リストから最も近いものにマッチングして返す。
    scorerで指定したスコアリング関数を利用。
    閾値以下の場合はNoneを返す。
    """
    if not text:
        return None
    result = process.extractOne(
        text, candidates, scorer=scorer, score_cutoff=30
    )
    if result is None:
        return None
    match, _, _ = result
    return match
