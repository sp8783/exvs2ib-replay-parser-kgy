import argparse
from src.core.pipeline import Pipeline

def main():
    """
    コマンドライン引数を受け取り、パイプライン処理を実行する。
    """
    parser = argparse.ArgumentParser(description="EXVS2IB 戦績トラッカー")
    parser.add_argument("--input", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--config", default="config/config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()

    pipeline = Pipeline(args.input, args.config)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
