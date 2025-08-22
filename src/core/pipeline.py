import os
import pandas as pd
from src.core.config import Config
from src.processing.match_extractor import MatchExtractor
from src.processing.screen_detector import ScreenDetector
from src.util.io import ensure_dir, save_dataframe_csv
from src.util.cache import CacheManager
from src.video.handler import extract_frames

class Pipeline:
    """
    動画解析パイプライン全体を統括するメインクラス。
    各処理は専用クラスに委譲し、全体のフローを管理する。
    """
    
    def __init__(self, video_path, config_path):
        self.config = Config(config_path)
        self.video_path = video_path
        self.video_basename = os.path.splitext(os.path.basename(video_path))[0]
        
        # ディレクトリ設定
        self.frames_dir = os.path.join(self.config.get("output", "frames", default="output/frames"), self.video_basename)
        self.cache_dir = os.path.join(self.config.get("output", "cache", default="output/cache"))
        self.results_dir = os.path.join(self.config.get("output", "results", default="output/results"))
        
        # 各種マネージャー・プロセッサーの初期化
        self.cache_manager = CacheManager(self.cache_dir, self.video_basename)
        self.screen_detector = ScreenDetector(config_path)
        self.match_extractor = MatchExtractor(self.config.get("video", "frame_interval"))
        
        self._prepare_output_dirs()
    
    def _prepare_output_dirs(self):
        """
        必要なディレクトリを作成する。
        """
        ensure_dir(self.frames_dir)
        ensure_dir(self.results_dir)
    
    def run_pipeline(self):
        """
        パイプライン全体を実行する。
        """
        # 1. フレーム抽出（キャッシュ対応）
        frame_paths = self._extract_frames_with_cache()
        
        # 2. 画面判定とマッチングカウント算出（キャッシュ対応）
        screens, match_count = self._detect_screens_with_cache(frame_paths)
        
        # 3. 試合結果抽出
        results = self.match_extractor.extract_match_results(screens, match_count)
        
        # 4. CSV出力
        self._save_results_to_csv(results)

    def _extract_frames_with_cache(self):
        """
        フレーム抽出（キャッシュ対応）
        """
        if self.cache_manager.has_frames_cache():
            return self.cache_manager.load_frames_cache()
        
        frame_paths = extract_frames(self.video_path, self.config.get("video", "frame_interval"), self.frames_dir)
        self.cache_manager.save_frames_cache(frame_paths)
        return frame_paths
    
    def _detect_screens_with_cache(self, frame_paths):
        """
        画面検出（キャッシュ対応）
        """
        if self.cache_manager.has_screens_cache():
            return self.cache_manager.load_screens_cache()
        
        screens, match_count = self.screen_detector.detect_screens(frame_paths, self.results_dir)
        self.cache_manager.save_screens_cache(screens, match_count)
        return screens, match_count

    def _save_results_to_csv(self, results):
        """
        試合結果をCSVファイルに保存する。
        """
        output_path = os.path.join(self.results_dir, f"result_{self.video_basename}.csv")
        dataframe = pd.DataFrame(results)
        save_dataframe_csv(dataframe, output_path)
        print(f"結果を {output_path} に保存しました。")
