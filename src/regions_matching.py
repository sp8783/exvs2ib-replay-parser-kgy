import cv2
import pytesseract
from rapidfuzz import process
import csv
from .preprocess import preprocess_for_ocr

# 座標設定
regions = {
    "player1_name": (83, 746, 343, 776),
    "player1_unit": (65, 796, 424, 819),
    "player2_name": (518, 700, 754, 725),
    "player2_unit": (501, 744, 826, 767),
    "player3_name": (1080, 700, 1316, 725),
    "player3_unit": (1099, 744, 1421, 767),
    "player4_name": (1479, 746, 1737, 776),
    "player4_unit": (1500, 796, 1860, 819),
}

def load_candidates(path):
    """
    指定したCSVファイルから候補リスト（1列目）を読み込んでリストで返す。
    """
    with open(path, encoding="utf-8") as f:
        return [row[0] for row in csv.reader(f) if row]

player_candidates = load_candidates("data/player_names.csv")
unit_candidates = load_candidates("data/unit_names.csv")

def normalize_numbers(text):
    """
    丸数字など特殊な数字表記を通常の数字に変換する。
    """
    if not text:
        return text
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
    for key, (x1, y1, x2, y2) in regions.items():
        roi = frame[y1:y2, x1:x2]
        # cv2.imwrite(f"output/debug/debug_{key}.png", roi) # debug:検出領域を保存
        processed = preprocess_for_ocr(roi)
        text = pytesseract.image_to_string(processed, lang="jpn", config="--psm 7").strip()
        text = normalize_numbers(text)
        # print(f"OCR raw text [{key}]: {text}") # debug:OCR生テキスト
        if "name" in key:
            results[key] = match_candidate(text, player_candidates)
        else:
            results[key] = match_candidate(text, unit_candidates)
    return results
