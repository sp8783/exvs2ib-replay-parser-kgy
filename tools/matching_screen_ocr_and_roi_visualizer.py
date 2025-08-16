import cv2
import glob
from src.regions_matching import (
    ocr_on_matching_regions,
    get_player_unit_roi_from_ratio,
    get_preprocessed_text_from_roi,
    match_text,
    get_roi
)
from src.preprocess import preprocess_for_ocr

def draw_roi(img, roi, color, label):
    """
    指定したROI領域を矩形で描画し、ラベルを付与した画像を返す関数。
    """
    x1, y1, x2, y2 = roi
    img = img.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font_scale = 0.5
    thickness = 1
    label_y = max(y1 - 5, 0)
    cv2.putText(img, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img

def show_roi_info(img, key, roi, color):
    """
    1つのROI領域について、前処理済みテキスト・マッチング結果・前処理画像プレビューを表示する。
    """
    # ROI枠描画
    img_with_roi = draw_roi(img, roi, color, key)
    # 前処理済みテキスト取得
    preprocessed_text = get_preprocessed_text_from_roi(img, roi)
    # マッチング結果取得
    matched = match_text(key, preprocessed_text)
    print(f"  {key} OCR preprocessed: {preprocessed_text}")
    print(f"  {key} matched: {matched}")
    # 前処理画像プレビュー
    roi_img = get_roi(img, roi)
    processed_img = preprocess_for_ocr(roi_img)
    cv2.imshow(f"{key} processed preview", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
    cv2.imshow(f"Preview: {img_path}", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    マッチング画面画像を一括でOCR解析し、各領域の認識結果（前処理済みテキスト、マッチング結果、前処理画像プレビュー）とROIプレビューを表示するデバッグツール。
    """
    img_folder = "data/sample_frames/matching2/"
    img_paths = glob.glob(img_folder + "*.png")
    if not img_paths:
        print(f"No images found in {img_folder}")
        return

    for img_path in img_paths:
        process_image(img_path)

if __name__ == "__main__":
    main()
