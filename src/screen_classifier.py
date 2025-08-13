import cv2
import numpy as np
import os

# テンプレート画像のパス
TEMPLATE_DIR = "templates"
VS_TEMPLATE = os.path.join(TEMPLATE_DIR, "vs.png")
WIN_TEMPLATE = os.path.join(TEMPLATE_DIR, "win.png")
LOSE_TEMPLATE = os.path.join(TEMPLATE_DIR, "lose.png")

def resize_to_template(region, template_path):
    """
    region: 元画像から切り出したROI（numpy配列）
    template_path: テンプレート画像ファイルパス
    戻り値: テンプレート画像と同じサイズにリサイズしたregion
    """
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"テンプレート画像が見つかりません: {template_path}")
    h, w = template.shape[:2]
    region_resized = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA)
    return region_resized

def _match_template(region, template_path, threshold=0.3):  # 0.4以上だとWIN画面が検出できない
    """
    region: 比較対象領域（numpy配列）
    template_path: テンプレート画像ファイルパス
    threshold: マッチングの閾値
    戻り値: 一致判定（True/False）
    """
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return False
    region_resized = resize_to_template(region, template_path)
    region_gray = cv2.cvtColor(region_resized, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(region_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val >= threshold

def classify_screen(img):
    """
    テンプレートマッチングで画面種別を判定
    - マッチング画面: 中央の「VS」ロゴ
    - リザルト画面: 左上の「WIN」「LOSE」ロゴ
    - その他: "other"
    """
    if img is None:
        return "other"

    # マッチング画面判定（中央の「VS」ロゴ）
    vs_x1, vs_y1, vs_x2, vs_y2 = (789, 347, 1120, 619)
    vs_roi = img[vs_y1:vs_y2, vs_x1:vs_x2]
    if _match_template(vs_roi, VS_TEMPLATE):
        return "matching"

    # WIN判定（左上の「WIN」ロゴ）
    win_x1, win_y1, win_x2, win_y2 = (60, 10, 440, 120)
    win_roi = img[win_y1:win_y2, win_x1:win_x2]
    if _match_template(win_roi, WIN_TEMPLATE):
        return "result_win"

    # LOSE判定（左上の「LOSE」ロゴ）
    lose_x1, lose_y1, lose_x2, lose_y2 = (60, 10, 615, 120)
    lose_roi = img[lose_y1:lose_y2, lose_x1:lose_x2]
    if _match_template(lose_roi, LOSE_TEMPLATE):
        return "result_lose"

    return "other"
