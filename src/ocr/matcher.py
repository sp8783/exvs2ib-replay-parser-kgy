from src.core.config import Config
from src.util.io import load_csv_candidates
from .scorer import matching_scorer_for_unit_name, matching_scorer_for_player_name
from rapidfuzz import process
from typing import Callable


class Matcher:
    """
    OCRテキストを候補リストにマッチングするクラス。
    初期化時に一度だけ候補リストをロードする。
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        player_path = config.get("data", "player_names", default="data/player_names.csv")
        unit_path = config.get("data", "unit_names", default="data/unit_names.csv")
        self.player_candidates: list[str] = load_csv_candidates(player_path)
        self.unit_candidates: list[str] = load_csv_candidates(unit_path)

    def match_candidate(self, text: str, candidates: list[str], scorer: Callable) -> str | None:
        """
        OCRで得たテキストを候補リストから最も近いものにマッチングして返す。
        scorerで指定したスコアリング関数を利用。
        閾値以下の場合はNoneを返す。
        """
        if not text:
            return None
        score_cutoff = self.config.get("ocr", "score_cutoff", default=30)
        result = process.extractOne(
            text, candidates, scorer=scorer, score_cutoff=score_cutoff
        )
        if result is None:
            return None
        match, _, _ = result
        return match

    def match_text(self, key: str, text: str) -> str | None:
        """
        領域種別（name/unit）に応じて、テキストを候補リストとマッチングして返す関数。
        """
        if "name" in key:
            return self.match_candidate(text, self.player_candidates, matching_scorer_for_player_name)
        else:
            return self.match_candidate(text, self.unit_candidates, matching_scorer_for_unit_name)
