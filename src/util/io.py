import os
import csv
import pickle
from typing import Any
import pandas as pd


def ensure_dir(dir_path: str) -> None:
    """
    ディレクトリが存在しない場合は作成する。
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_csv_candidates(csv_path: str) -> list[str]:
    """
    CSVファイルから候補リストを読み込む。
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} が見つかりません。空のリストを返します。")
        return []

    candidates = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                candidates.append(row[0])
    return candidates


def save_csv(data: list, header: list, csv_path: str) -> None:
    """
    データをCSVファイルに保存する（リスト形式）。
    """
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def save_dataframe_csv(dataframe: pd.DataFrame, csv_path: str) -> None:
    """
    DataFrameをCSVファイルに保存する。
    """
    ensure_dir(os.path.dirname(csv_path))
    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")


def save_pickle(data: Any, file_path: str) -> None:
    """
    データをpickleファイルに保存する。
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path: str) -> Any:
    """
    pickleファイルからデータを読み込む。
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_screen_log(log_rows: list[dict], results_dir: str) -> None:
    """
    画面判定結果をCSVファイルに保存する。
    """
    ensure_dir(results_dir)
    log_path = os.path.join(results_dir, "screen_log.csv")
    pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"画面判定結果を {log_path} に保存しました。")
