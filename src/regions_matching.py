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

# 候補リスト読み込み
def load_candidates(path):
    with open(path, encoding="utf-8") as f:
        return [row[0] for row in csv.reader(f) if row]

player_candidates = load_candidates("data/player_names.csv")
unit_candidates = load_candidates("data/unit_names.csv")

# 丸数字などの数字を通常の数字に変換
def normalize_numbers(text):
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

# 文字列を候補リストから最も近いものにマッチング
def match_candidate(text, candidates):
    if not text:
        return None
    result = process.extractOne(text, candidates, score_cutoff=30)  # 閾値はテスト結果から低めに設定
    if result is None:
        return None
    match, score, _ = result
    return match

def ocr_on_matching_regions(img):
    frame = img
    if frame is None:
        raise FileNotFoundError(f"Image not found at {img}")
    results = {}
    for key, (x1, y1, x2, y2) in regions.items():
        roi = frame[y1:y2, x1:x2]
        # cv2.imwrite(f"output/debug/debug_{key}.png", roi) # デバッグ用に検出領域を保存
        processed = preprocess_for_ocr(roi)
        text = pytesseract.image_to_string(processed, lang="jpn", config="--psm 7").strip()
        text = normalize_numbers(text)

        print(f"OCR raw text [{key}]: {text}")

        if "name" in key:
            results[key] = match_candidate(text, player_candidates)
        else:
            results[key] = match_candidate(text, unit_candidates)
    return results
