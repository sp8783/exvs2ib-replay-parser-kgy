import yaml
from typing import Any


class Config:
    """
    config.yamlの読み込み・アクセスを提供する設定管理クラス。
    階層化された設定値を辞書として保持し、getメソッドで取得可能。
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        with open(config_path, encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        階層化されたキーで設定値を取得する。
        例: get("ocr", "lang") → "jpn+eng"
        """
        val = self._config
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

    def as_dict(self) -> dict:
        """
        設定全体を辞書で返す。
        """
        return self._config
