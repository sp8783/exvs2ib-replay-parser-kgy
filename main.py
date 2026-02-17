import argparse
from src.core.pipeline import Pipeline

def main():
    """
    コマンドライン引数を受け取り、パイプライン処理を実行する。
    """
    parser = argparse.ArgumentParser(description="EXVS2IB 戦績トラッカー")
    parser.add_argument("--input", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--config", default="config/config.yaml", help="設定ファイルのパス")
    parser.add_argument("--mode", choices=["full", "timestamps"], default="full",
                        help="実行モード: full=全工程実行（デフォルト）, timestamps=試合開始タイムスタンプのみ出力")
    args = parser.parse_args()

    pipeline = Pipeline(args.input, args.config, mode=args.mode)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
