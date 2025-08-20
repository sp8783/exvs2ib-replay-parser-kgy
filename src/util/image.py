from src.core.config import Config
import cv2

_config = Config("config/config.yaml")
roi_config = _config.get("roi", "player_unit", default={})

def get_player_unit_roi_from_ratio(img):
    """
    画像サイズに応じて割合から領域座標を計算して返す関数。
    プレイヤー名・機体名の領域座標を辞書で返す。
    """
    h, w = img.shape[:2]
    regions = {}
    for key, (x1r, y1r, x2r, y2r) in roi_config.items():
        x1 = int(w * x1r)
        y1 = int(h * y1r)
        x2 = int(w * x2r)
        y2 = int(h * y2r)
        regions[key] = (x1, y1, x2, y2)
    return regions

def get_roi(img, roi):
    """
    指定したROI座標から画像領域を切り出して返す関数。
    """
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2]

def resize_to_template(region, template):
    """
    regionをテンプレート画像のサイズにリサイズする関数。
    """
    if template is None:
        raise FileNotFoundError(f"テンプレート画像が見つかりません。")
    h, w = template.shape[:2]
    region_resized = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA)
    return region_resized

def roi_ratio_to_absolute(img, roi_ratio):
    """
    ROIの割合指定を絶対座標に変換する汎用関数。
    """
    h, w = img.shape[:2]
    x1 = int(w * roi_ratio[0])
    y1 = int(h * roi_ratio[1])
    x2 = int(w * roi_ratio[2])
    y2 = int(h * roi_ratio[3])
    return (x1, y1, x2, y2)
