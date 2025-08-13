import cv2
import glob
from src.regions_matching import ocr_on_matching_regions

def main():
    """
    マッチング画面画像を一括でOCR解析し、各領域の認識結果を表示するテスト関数。
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
        
        for key, val in result.items():
            print(f"  {key}: {val}")
        print("-" * 40)

if __name__ == "__main__":
    main()
