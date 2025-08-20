import pytesseract
import easyocr
from src.core.config import Config

class OCREngineAdapter:
    """
    OCRエンジンの切り替えを管理するシンプルなアダプター
    """

    def __init__(self, config_path="config/config.yaml"):
        self.config = Config(config_path)
        self.engine_name = self.config.get("ocr", "engine", default="tesseract")
        self._easyocr_reader = None
                
    def extract_text(self, image):
        """
        設定されたエンジンでテキストを抽出
        """
        if self.engine_name == "tesseract":
            return self._tesseract_extract(image)
        elif self.engine_name == "easyocr":
            return self._easyocr_extract(image)
        else:
            raise ValueError(f"未対応のOCRエンジンです: {self.engine_name}")
    
    def _tesseract_extract(self, image):
        """
        Tesseract OCR
        """
        tesseract_conf = self.config.get("ocr", "engines", "tesseract", default={})
        lang = tesseract_conf.get("lang", "jpn+eng")
        psm = tesseract_conf.get("psm", 7)
        config_str = tesseract_conf.get("config", f"--psm {psm}")

        return pytesseract.image_to_string(image, lang=lang, config=config_str).strip()

    def _easyocr_extract(self, image):
        """
        EasyOCR
        """
        if self._easyocr_reader is None:
            easyocr_conf = self.config.get("ocr", "engines", "easyocr", default={})
            lang = easyocr_conf.get("languages", ["ja", "en"])
            self._easyocr_reader = easyocr.Reader(lang)
        results = self._easyocr_reader.readtext(image)        
        texts = [result[1] for result in results]
        return " ".join(texts).strip()

# グローバルインスタンス（シングルトン的に使用）
_ocr_adapter = None

def get_ocr_adapter():
    """OCRアダプターのインスタンスを取得"""
    global _ocr_adapter
    if _ocr_adapter is None:
        _ocr_adapter = OCREngineAdapter()
    return _ocr_adapter
