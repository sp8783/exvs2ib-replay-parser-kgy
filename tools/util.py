import cv2

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

def show_image(img, window_name="Preview"):
    """
    画像をウィンドウで表示し、任意のキー入力で閉じる。
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
