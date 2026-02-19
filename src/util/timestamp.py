def calculate_timestamp(frame_name: str, frame_interval: float) -> str:
    """
    フレーム画像ファイル名から連番を取得し、frame_intervalからタイムスタンプをhh:mm:ss形式で返す。
    """
    # frame_00001.png形式から番号を抽出
    frame_num = int(frame_name.split("_")[1].split(".")[0])
    total_seconds = int(frame_num * frame_interval)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
