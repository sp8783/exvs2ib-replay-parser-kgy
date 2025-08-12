import pytesseract
import cv2

def run_ocr(image_path, lang="jpn+eng"):
    """画像ファイルからOCRで文字列を抽出"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")
    
    # 必要に応じて前処理（グレースケール化など）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray, lang=lang)
    return text.strip()
