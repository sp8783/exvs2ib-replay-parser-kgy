import cv2
from src.core.config import Config

def shape_selection(event, x, y, flags, param):
    """
    マウス操作で画像上の矩形領域を選択するためのコールバック関数。
    左クリックで始点、ドラッグで矩形表示、離すと終点を記録し座標と割合を表示する。
    """
    global ref_point, cropping, img

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        img_temp = img.copy()
        cv2.rectangle(img_temp, ref_point[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("image", img_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        cv2.rectangle(img, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", img)
        print(f"Selected region: {ref_point[0]} to {ref_point[1]}")

        h, w = img.shape[:2]
        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1_ratio = x1 / w
        y1_ratio = y1 / h
        x2_ratio = x2 / w
        y2_ratio = y2 / h
        print(f"Ratios: x1={x1_ratio:.4f}, y1={y1_ratio:.4f}, x2={x2_ratio:.4f}, y2={y2_ratio:.4f}")

def main():
    """
    画像を表示し、マウスドラッグでROI（矩形領域）を選択できるツール。
    選択した座標と画像サイズに対する割合を表示する。
    """
    global img, ref_point, cropping
    ref_point = []
    cropping = False

    # config.yamlから画像パスを取得
    config = Config("config/config.yaml")
    img_path = config.get("tools", "roi_selector", "image_path")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image '{img_path}'")
        exit(1)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", shape_selection)

    print("ドラッグで範囲選択してください。")
    print("rキーでリセット、qキーで終了。")

    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            img = cv2.imread(img_path)  # 画像をリセット

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
