import os
import csv
import pickle
import pandas as pd

def ensure_dir(dir_path):
    """
    ディレクトリが存在しない場合は作成する。
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_csv_candidates(csv_path):
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

def save_csv(data, header, csv_path):
    """
    データをCSVファイルに保存する（リスト形式）。
    """
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def save_dataframe_csv(dataframe, csv_path):
    """
    DataFrameをCSVファイルに保存する。
    """
    ensure_dir(os.path.dirname(csv_path))
    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")

def save_pickle(data, file_path):
    """
    データをpickleファイルに保存する。
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    """
    pickleファイルからデータを読み込む。
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)
