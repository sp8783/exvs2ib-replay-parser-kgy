import cv2
import glob
from src.regions_result import ocr_result_winlose_regions

def main():
    img_folder = "data/sample_frames/result/"
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

        result = ocr_result_winlose_regions(img)

        for player, res in result.items():
            print(f"  {player}: {res}")
        print("-" * 40)

if __name__ == "__main__":
    main()
