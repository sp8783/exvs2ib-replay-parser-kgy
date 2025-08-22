import cv2
from src.core.config import Config
from src.util.image import resize_to_template, roi_ratio_to_absolute

class ScreenClassifier:
    """
    画面種別（vs, win, lose, unknown）をテンプレートマッチングで判定するクラス。
    設定ファイルからテンプレート画像とROI情報を取得し、初期化時にテンプレート画像を読み込む。
    """

    def __init__(self, config_path="config/config.yaml"):
        """
        設定ファイルからテンプレート画像パス・ROI情報を取得し、テンプレート画像を事前に読み込む。
        """
        config = Config(config_path)
        self.threshold = config.get("template", "threshold", default=0.3)
        self.template_config = config.get("template", default={})
        self.roi_config = config.get("roi", default={})
        self.vs_template = self._load_template(self.template_config.get("vs"))
        self.win_template = self._load_template(self.template_config.get("win"))
        self.lose_template = self._load_template(self.template_config.get("lose"))

    def _load_template(self, template_path):
        """
        指定パスのテンプレート画像をグレースケールで読み込む。
        """
        if template_path:
            return cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        return None

    def _roi_to_abs(self, img, roi_ratio):
        """
        ROIの割合指定を絶対座標に変換する。
        """
        return roi_ratio_to_absolute(img, roi_ratio)

    def match_template(self, img, template, roi=None):
        """
        指定したROI領域でテンプレートマッチングを行い、類似度が閾値以上ならTrueを返す。
        """
        threshold = self.threshold
        if roi:
            x1, y1, x2, y2 = roi
            img = img[y1:y2, x1:x2]
        region_resized = resize_to_template(img, template)
        region_gray = cv2.cvtColor(region_resized, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(region_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= threshold

    def classify(self, img):
        """
        画面種別（matching, result_win, result_lose, unknown）を判定して返す。
        """
        vs_roi = self.roi_config.get("vs")
        win_roi = self.roi_config.get("win")
        lose_roi = self.roi_config.get("lose")

        if self.vs_template is not None and vs_roi and self.match_template(img, self.vs_template, self._roi_to_abs(img, vs_roi)):
            return "matching"
        if self.win_template is not None and win_roi and self.match_template(img, self.win_template, self._roi_to_abs(img, win_roi)):
            return "result_win"
        if self.lose_template is not None and lose_roi and self.match_template(img, self.lose_template, self._roi_to_abs(img, lose_roi)):
            return "result_lose"
        return "other"
