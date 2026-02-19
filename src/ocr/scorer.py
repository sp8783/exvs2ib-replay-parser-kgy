import re
from rapidfuzz import fuzz


def get_char_type(ch: str) -> str:
    """
    1文字の文字種を判定して返す（K:漢字, C:カタカナ, E:英字, N:数字, S:その他）。
    """
    if re.match(r'[\u4E00-\u9FFF]', ch):  # 漢字
        return "K"
    elif re.match(r'[\u30A0-\u30FF]', ch):  # カタカナ
        return "C"
    elif re.match(r'[A-Za-z]', ch):  # 英字
        return "E"
    elif re.match(r'[0-9０-９]', ch):  # 数字
        return "N"
    else:
        return "S"


def get_pattern(s: str) -> str:
    """
    文字列sの各文字の文字種パターンを連結して返す。
    例: "騎士ガンダム" → "KKCCCC", "νガンダム" → "SCCCC"
    """
    return "".join(get_char_type(ch) for ch in s if ch.strip())


def matching_scorer_for_unit_name(a: str, b: str, **kwargs) -> float:
    """
    機体名用スコアリング関数。
    文字列類似度（70%）＋文字種パターン一致度（30%）－長さペナルティでスコアを算出。
    機体名はOCR検出しづらい漢字やギリシャ文字を含む名称が少なく、文字種パターン列の違いがマッチング精度に比較的大きく影響するため、
    文字種パターン列を考慮したスコアリングを行う。
    """
    score_text = fuzz.ratio(a, b)
    pattern_a = get_pattern(a)
    pattern_b = get_pattern(b)
    score_pattern = fuzz.ratio(pattern_a, pattern_b)
    length_penalty = abs(len(a) - len(b)) * 10
    return score_text * 0.7 + score_pattern * 0.3 - length_penalty


def matching_scorer_for_player_name(a: str, b: str, **kwargs) -> float:
    """
    プレイヤー名用スコアリング関数。
    文字列類似度－長さペナルティでスコアを算出。
    """
    score = fuzz.ratio(a, b)
    length_penalty = abs(len(a) - len(b)) * 5
    return score - length_penalty
