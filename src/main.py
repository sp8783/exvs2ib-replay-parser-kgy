import cv2
from src.regions_matching import ocr_on_matching_regions

frame = cv2.imread("data/sample_frame.png")
results = ocr_on_matching_regions(frame)
print(results)
