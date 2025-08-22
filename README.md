# EXVS2IB Replay Parser (KGY専用)

**機動戦士ガンダム エクストリームバーサス2 インフィニットブースト（EXVS2IB）** の対戦動画から、戦績情報（勝敗・プレイヤー名・機体名など）を自動抽出するツールです。  
このリポジトリは **KGY専用・実験的実装** です。

---

## 概要
YouTubeなどに投稿された対戦動画を対象に、OCRで画面上の文字を解析し、試合ごとの情報をCSVに記録します。  
現在は **動画入力 → フレーム抽出 → OCR → CSV出力** まで対応しています。  

---

## 機能
- 🎞️ **動画処理**: 動画からフレームを抽出（デフォルトは2秒間隔）  
- 🖼️ **画面判定**: マッチング画面・リザルト画面の識別  
- 🔍 **OCR解析**: プレイヤー名・機体名を自動抽出（Tesseract使用）  
- 📊 **出力**: CSV保存（キャッシュ機能付き）  
- 🛠️ **デバッグ用ツール**: 画面判定結果・OCR結果・ROI選択の可視化  

---

## 技術スタック
- 言語: Python 3.13.5  
- OCRエンジン: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)  
- 主要ライブラリ: OpenCV, pytesseract, NumPy, pandas  

---

## 導入方法

### 1. リポジトリを取得
```bash
git clone https://github.com/yourname/exvs2ib-replay-parser-kgy.git
```

### 2. 必要なライブラリをインストール
```bash
pip install -r requirements.txt
```

### 3. Tesseractをインストール
- macOS:`brew install tesseract`
- Ubuntu: `sudo apt-get install tesseract-ocr`
- Windows: [Tesseract公式ページ](https://github.com/UB-Mannheim/tesseract/wiki) からインストーラを利用

---

## 使用例
動画ファイルを指定して解析を実行します：
```bash
python main.py --input data/video.mp4
```
処理結果は `output/results/` に CSV として保存されます。

---

## ファイル構成
```plaintext
exvs2ib-replay-parser-kgy/
├── main.py                          # メインエントリーポイント
├── requirements.txt
├── config/
│   └── config.yaml                  # OCR設定・ROI座標など
├── data/
│   ├── player_names.csv             # プレイヤー名候補DB
│   └── unit_names.csv               # 機体名候補DB
├── output/
│   ├── cache/                       # キャッシュ
│   ├── frames/                      # 抽出フレーム
│   └── results/                     # CSV出力
├── src/ ...                         # 実装モジュール
├── templates/ ...                   # 判定用テンプレート画像
└── tools/ ...                       # デバッグツール類
```

---

## 既知の課題
⚠️ 現在のOCR精度は実用レベルに達していません。改善が必要です。
- プレイヤー名・機体名の誤認識が約2.5%発生
- 数字を含む名称（ex. 試作2号機）、特殊文字（ex. ν）、漢字（ex. 神、騎士）で精度低下

---

## 今後の開発予定
- プレイヤー名・機体名の検出精度向上
- リアルタイム解析対応
- Webアプリへの統合

---

## バージョン情報
- **v0.1.0 (初期実装)**  
  - 動画フレーム抽出（デフォルトは2秒間隔）  
  - 画面判定（マッチング/リザルト画面）  
  - OCRによるプレイヤー名・機体名の抽出  
  - CSV出力とキャッシュ機能  
  - デバッグツール（画面判定・OCR結果・ROI選択）  
