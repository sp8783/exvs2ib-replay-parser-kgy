import cv2
import glob
from src.regions_matching import ocr_on_matching_regions, get_player_unit_roi_from_ratio
import pytesseract
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

def main():
    """
    マッチング画面画像を一括でOCR解析し、各領域の認識結果とROIプレビューを表示するツール。
    前処理パラメータはpreprocess.pyの定数を編集することで調整可能。
    """
    img_folder = "data/sample_frames/matching/"
    img_paths = glob.glob(img_folder + "*.png")
    if not img_paths:
        print(f"No images found in {img_folder}")
        return

    for img_path in img_paths:
        print(f"Processing: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Failed to load image at {img_path}")
            continue

        result = ocr_on_matching_regions(img)
        regions = get_player_unit_roi_from_ratio(img)
        preview = img.copy()
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (128,128,128), (0,128,255)]
        for idx, (key, roi) in enumerate(regions.items()):
            preview = draw_roi(preview, roi, colors[idx % len(colors)], key)
            x1, y1, x2, y2 = roi
            roi_img = img[y1:y2, x1:x2]
            processed = preprocess_for_ocr(roi_img)
            ocr_text = pytesseract.image_to_string(processed, lang="jpn", config="--psm 7").strip()
            print(f"  {key} OCR raw: {ocr_text}")
            cv2.imshow(f"{key} processed", processed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        for key, val in result.items():
            print(f"  {key}: {val}")
        print("-" * 40)
        cv2.imshow(f"Preview: {img_path}", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
