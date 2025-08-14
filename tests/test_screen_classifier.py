import cv2
import glob
import os
from src.screen_classifier import VS_ROI_RATIO, WIN_ROI_RATIO, LOSE_ROI_RATIO, classify_screen, get_logo_roi_from_ratio

def get_rois(img):
    """
    マッチング画面の画像サイズに応じて、VSロゴ・WINロゴ・LOSEロゴの領域座標を計算して返す関数。
    VSロゴ・WINロゴ・LOSEロゴの領域座標を辞書で返す。
    """
    return {
        "VS": get_logo_roi_from_ratio(img, VS_ROI_RATIO),
        "WIN": get_logo_roi_from_ratio(img, WIN_ROI_RATIO),
        "LOSE": get_logo_roi_from_ratio(img, LOSE_ROI_RATIO)
    }

def draw_roi(img, roi, color, label):
    """
    指定したROI領域を矩形で描画し、ラベルを付与した画像を返す関数。
    デバッグ用プレビュー表示に利用。
    """
    x1, y1, x2, y2 = roi
    img = img.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img

def main():
    """
    サンプル画像群に対して画面種別判定（classify_screen）を実行し、結果を表示するテスト関数。
    """
    base_folder = "data/sample_frames/"
    img_paths = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith(".png"):
                img_paths.append(os.path.join(root, file))
    if not img_paths:
        print(f"No images found in {base_folder}")
        return

    for img_path in img_paths:
        img = cv2.imread(img_path)
        print(f"{img_path}: {classify_screen(img)}")

        # debug:プレビュー表示
        # rois = get_rois(img)
        # preview = img.copy()
        # preview = draw_roi(preview, rois["VS"], (255,0,0), "VS")
        # preview = draw_roi(preview, rois["WIN"], (0,255,0), "WIN")
        # preview = draw_roi(preview, rois["LOSE"], (0,0,255), "LOSE")
        # cv2.imshow(f"Preview: {img_path}", preview)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
