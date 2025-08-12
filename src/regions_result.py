import cv2
import pytesseract
from rapidfuzz import process
from .preprocess import preprocess_for_ocr

# 座標設定
result_region = (66, 4, 621, 118)

# 候補リスト
result_candidates = ["WIN", "LOSE"]

# 文字列を候補リストから最も近いものにマッチング
def match_result(text, candidates=result_candidates):
    if not text:
        return None
    result = process.extractOne(text.upper(), candidates, score_cutoff=30)  # 閾値はテスト結果から低めに設定
    if result is None:
        return None
    match, score, _ = result
    return match

def ocr_result_winlose_regions(img):
    x1, y1, x2, y2 = result_region
    roi = img[y1:y2, x1:x2]

    processed = preprocess_for_ocr(roi)

    text = pytesseract.image_to_string(processed, lang="eng", config="--psm 7").strip()    
    print(f"OCR raw text [result]: {text}")

    result = match_result(text)
    result_inverse = "LOSE" if result == "WIN" else "WIN"

    results = {
        "player1_result": result,
        "player2_result": result,
        "player3_result": result_inverse,
        "player4_result": result_inverse,
    }

    return results
