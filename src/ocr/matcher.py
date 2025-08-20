from src.core.config import Config
from src.util.io import load_csv_candidates
from .scorer import matching_scorer_for_unit_name, matching_scorer_for_player_name
from rapidfuzz import process

_config = Config("config/config.yaml")
player_names_path = _config.get("data", "player_names", default="data/player_names.csv")
unit_names_path = _config.get("data", "unit_names", default="data/unit_names.csv")
player_candidates = load_csv_candidates(player_names_path)
unit_candidates = load_csv_candidates(unit_names_path)

def match_candidate(text, candidates, scorer):
    """
    OCRで得たテキストを候補リストから最も近いものにマッチングして返す。
    scorerで指定したスコアリング関数を利用。
    閾値以下の場合はNoneを返す。
    """
    if not text:
        return None
    score_cutoff = _config.get("ocr", "score_cutoff", default=30)
    result = process.extractOne(
        text, candidates, scorer=scorer, score_cutoff=score_cutoff
    )
    if result is None:
        return None
    match, _, _ = result
    return match

def match_text(key, text):
    """
    領域種別（name/unit）に応じて、テキストを候補リストとマッチングして返す関数。
    """
    if "name" in key:
        return match_candidate(text, player_candidates, matching_scorer_for_player_name)
    else:
        return match_candidate(text, unit_candidates, matching_scorer_for_unit_name)
