import argparse
from src.core.pipeline import Pipeline


def main() -> None:
    """
    コマンドライン引数を受け取り、パイプライン処理を実行する。
    """
    parser = argparse.ArgumentParser(description="EXVS2IB 戦績トラッカー")
    parser.add_argument("--input", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--config", default="config/config.yaml", help="設定ファイルのパス")
    parser.add_argument("--with-ocr", action="store_true",
                        help="【実験的】プレイヤー名・機体名・勝敗も抽出する（精度は保証されない）")
    args = parser.parse_args()

    pipeline = Pipeline(args.input, args.config, with_ocr=args.with_ocr)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
