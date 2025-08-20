import cv2
import glob
from src.core.config import Config
from src.ocr import (
    ocr_on_matching_regions,
    get_preprocessed_text_from_roi,
    match_text,
    preprocess_for_ocr
)
from src.util.image import get_player_unit_roi_from_ratio, get_roi
from src.core.config import Config
from tools.util import draw_roi, show_image

def show_roi_info(img, key, roi, color):
    """
    1つのROI領域について、前処理済みテキスト・マッチング結果・前処理画像プレビューを表示する。
    """
    img_with_roi = draw_roi(img, roi, color, key)
    preprocessed_text = get_preprocessed_text_from_roi(img, roi)
    matched = match_text(key, preprocessed_text)
    print(f"  {key} OCR preprocessed: {preprocessed_text}")
    print(f"  {key} matched: {matched}")
    roi_img = get_roi(img, roi)
    processed_img = preprocess_for_ocr(roi_img)
    show_image(processed_img, f"{key} processed preview")
    return img_with_roi

def show_roi_debug(img, regions):
    """
    各ROIごとにshow_roi_infoを使って情報表示・プレビューを行う。
    """
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (128,128,128), (0,128,255)]
    preview = img.copy()
    for idx, (key, roi) in enumerate(regions.items()):
        preview = show_roi_info(preview, key, roi, colors[idx % len(colors)])
    return preview

def show_final_result(result):
    """
    ocr_on_matching_regionsの最終結果を表示する。
    """
    print("\nocr_on_matching_regionsの最終結果:")
    for key, val in result.items():
        print(f"  {key}: {val}")
    print("-" * 40)

def process_image(img_path):
    """
    画像1枚分のデバッグ処理を行う。
    """
    print(f"Processing: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Failed to load image at {img_path}")
        return

    result = ocr_on_matching_regions(img)
    regions = get_player_unit_roi_from_ratio(img)
    preview = show_roi_debug(img, regions)
    show_final_result(result)
    show_image(preview, f"Preview: {img_path}")

def main():
    """
    マッチング画面画像を一括でOCR解析し、各領域の認識結果（前処理済みテキスト、マッチング結果、前処理画像プレビュー）とROIプレビューを表示するデバッグツール。
    """
    config = Config("config/config.yaml")
    img_folder = config.get("tools", "matching_ocr_visualizer", "matching_frames_dir")
    img_paths = glob.glob(img_folder + "*.png")
    if not img_paths:
        print(f"No images found in {img_folder}")
        return

    for img_path in img_paths:
        process_image(img_path)

if __name__ == "__main__":
    main()
