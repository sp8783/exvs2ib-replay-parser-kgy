import argparse
import os
import yaml
from src.video_handler import extract_frames
from src.ocr_handler import run_ocr
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="EXVS2IB 戦績トラッカー")
    parser.add_argument("--input", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--output", default="output/results/results.csv", help="出力CSVファイルパス")
    args = parser.parse_args()

    # 設定読み込み
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # フレーム抽出
    frame_paths = extract_frames(args.input, config["frame_interval"], "output/frames")

    # OCR実行
    results = []
    for frame_path in frame_paths:
        text = run_ocr(frame_path, lang=config["ocr_lang"])
        results.append({"frame": os.path.basename(frame_path), "text": text})

    # CSV保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"結果を {args.output} に保存しました。")

if __name__ == "__main__":
    main()
