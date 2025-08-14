import cv2
import pytesseract
from rapidfuzz import process
import csv
from .preprocess import preprocess_for_ocr

# 座標設定（各プレイヤー名、機体名）
PLAYER_UNIT_REGIONS_RATIO = {
    "player1_name": (0.045, 0.693, 0.18, 0.72),
    "player1_unit": (0.035, 0.74, 0.22, 0.76),
    "player2_name": (0.27, 0.65, 0.39, 0.67),
    "player2_unit": (0.265, 0.69, 0.43, 0.71),
    "player3_name": (0.565, 0.65, 0.68, 0.67),
    "player3_unit": (0.575, 0.69, 0.74, 0.71),
    "player4_name": (0.77, 0.693, 0.90, 0.72),
    "player4_unit": (0.78, 0.74, 0.97, 0.76),
}

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

def load_candidates(path):
    """
    指定したCSVファイルから候補リスト（1列目）を読み込んでリストで返す。
    """
    with open(path, encoding="utf-8") as f:
        return [row[0] for row in csv.reader(f) if row]

player_candidates = load_candidates("data/player_names.csv")
unit_candidates = load_candidates("data/unit_names.csv")

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

def match_candidate(text, candidates):
    """
    OCRで得たテキストを候補リストから最も近いものにマッチングして返す。
    閾値以下の場合はNoneを返す。
    """
    if not text:
        return None
    result = process.extractOne(text, candidates, score_cutoff=30)  # 閾値はテスト結果から低めに設定
    if result is None:
        return None
    match, _, _ = result
    return match

def ocr_on_matching_regions(img):
    """
    マッチング画面画像から各領域を切り出し、OCR・候補マッチングを行い、
    プレイヤー名・機体名を辞書で返す。
    """
    frame = img
    if frame is None:
        raise FileNotFoundError(f"Image not found at {img}")
    results = {}
    regions = get_player_unit_roi_from_ratio(frame)
    for key, (x1, y1, x2, y2) in regions.items():
        roi = frame[y1:y2, x1:x2]
        # cv2.imwrite(f"output/debug/debug_{key}.png", roi) # debug:検出領域を保存
        if roi is None or roi.size == 0:
            print(f"警告: ROIが空です。フレーム: {getattr(img, 'filename', '不明')}, 領域: {key}, 座標: ({x1}, {y1}, {x2}, {y2})")
            continue
        processed = preprocess_for_ocr(roi)
        text = pytesseract.image_to_string(processed, lang="jpn", config="--psm 7").strip()
        text = preprocess_ocr_text(text)
        # print(f"OCR raw text [{key}]: {text}") # debug:OCR生テキスト
        if "name" in key:
            results[key] = match_candidate(text, player_candidates)
        else:
            results[key] = match_candidate(text, unit_candidates)
    return results
