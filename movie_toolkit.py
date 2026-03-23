import cv2
import os
import glob
import inspect
from enum import Enum
import logging
import natsort
import argparse
from pathlib import Path
import math
import sys
import numpy as np

logger =  logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

def format_time(seconds):
    """秒数をHH:MM:SS.ms形式の文字列に変換する"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}h{m:02d}m{s:02d}s{ms:03d}ms"

def open_video(video_path):
    """
    日本語パスに対応したVideoCaptureを返す。
    cv2.VideoCaptureは非ASCIIパスを扱えない場合があるため、
    失敗時は一時ファイル経由で開く。
    """
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap

    # 非ASCIIパス対策: 一時ファイルにコピーして開く
    logger.info(f"標準のVideoCaptureで開けないため、一時ファイル経由で再試行します: {video_path}")
    import tempfile
    import shutil
    suffix = Path(video_path).suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    shutil.copy2(str(video_path), tmp.name)
    cap = cv2.VideoCapture(tmp.name)
    if cap.isOpened():
        # 一時ファイルパスを属性として保持し、後で削除できるようにする
        cap._tmp_path = tmp.name
        return cap
    else:
        os.unlink(tmp.name)
        return None

def release_video(cap):
    """VideoCaptureを解放し、一時ファイルがあれば削除する"""
    cap.release()
    tmp_path = getattr(cap, '_tmp_path', None)
    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

def imwrite_unicode(filepath, img, params=None):
    """
    日本語パスに対応した画像書き出し。
    cv2.imwriteの代わりにcv2.imencodeとPython標準I/Oを使用する。
    """
    filepath = str(filepath)
    ext = os.path.splitext(filepath)[1]
    encode_params = params if params else []
    result, buf = cv2.imencode(ext, img, encode_params)
    if not result:
        logger.error(f"画像のエンコードに失敗しました: {filepath}")
        return False
    buf.tofile(filepath)
    return True

def extract_frames_by_interval_in_folder(input_folder_path, output_folder_path, interval_ms=10000):
    """
    フォルダ内の全動画ファイルに対してフレーム抽出を実行する

    :param input_folder_path: 動画ファイルが格納されたフォルダのパス
    :param output_folder_path: フレームの保存先ディレクトリ
    :param interval_ms: 抽出間隔（ミリ秒）。デフォルトは10000ms（10秒）
    """
    if(not os.path.exists(input_folder_path)):
        logger.fatal(f"フォルダ：{input_folder_path}は存在しません。{inspect.currentframe().f_code.co_name}を終了します。")
        return
    
    input_folder_abspath = os.path.abspath(input_folder_path)

    logger.info(f"{input_folder_abspath}から動画ファイルを抽出します。")

    for(video_path) in glob.glob(input_folder_abspath+"/*"):
        try:
            extract_frames_by_interval(video_path=video_path, output_folder_path=output_folder_path, interval_ms=interval_ms)
        except Exception as e:
            logger.error(f"エラーが発生しました： {e}")
        finally:
            logger.info("処理を続行します。")

def extract_frames_by_interval(video_path, output_folder_path, interval_ms=10000):
    """
    指定された時間間隔で動画からフレームを抽出する

    :param video_path: 動画ファイルのパス
    :param output_folder_path: フレームの保存先ディレクトリ
    :param interval_ms: 抽出間隔（ミリ秒）。デフォルトは10000ms（10秒）
    """
    # ミリ秒を秒に変換
    interval_sec = interval_ms / 1000.0

    # pathlibを使用してパスをオブジェクトとして扱う
    video_path = Path(video_path)
    if not video_path.is_file():
        logger.info(f"エラー: 指定された動画ファイルが見つかりません: {video_path}")
        return

    # VideoCaptureオブジェクトを作成（日本語パス対応）
    cap = open_video(video_path)
    if cap is None:
        logger.info(f"エラー: 動画ファイルを開けませんでした: {video_path}")
        return

    # ビデオのメタデータを取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # FPSの妥当性を検証
    if fps == 0:
        logger.info("エラー: 動画のFPSを取得できませんでした。ファイルが破損しているか、対応していないコーデックの可能性があります。")
        release_video(cap)
        return

    logger.info("--- ビデオ情報 ---")
    logger.info(f"ファイルパス: {video_path}")
    logger.info(f"FPS: {fps:.2f}")
    logger.info(f"総フレーム数: {total_frames}")
    logger.info(f"動画の長さ: {duration_sec:.2f} 秒")
    logger.info(f"抽出間隔: {interval_ms} ミリ秒 ({interval_sec:.3f} 秒)")
    logger.info("--------------------")

    # 出力ディレクトリを作成（存在しない場合）
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"フレームの保存先: {output_folder_path}")

    # 抽出処理
    current_time_sec = 0.0
    saved_count = 0
    while current_time_sec <= duration_sec:
        # 目的の時間のフレーム番号を計算
        target_frame_num = math.floor(current_time_sec * fps)

        if target_frame_num >= total_frames:
            break
        
        # 指定したフレーム番号にシーク
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
        
        ret, frame = cap.read()
        if not ret:
            logger.info(f"警告: フレーム {target_frame_num} ({current_time_sec:.2f}秒地点) の読み込みに失敗しました。")
            current_time_sec += interval_sec
            continue

        # タイムスタンプに基づいたファイル名を生成
        timestamp_str = format_time(current_time_sec)
        output_filename = f"{video_path.name}_{timestamp_str}.jpg"
        output_path = output_folder_path / output_filename

        # フレームを画像ファイルとして保存（日本語パス対応）
        imwrite_unicode(output_path, frame)
        saved_count += 1
        logger.info(f"保存しました: {output_path} (フレーム番号: {target_frame_num})")
        
        # 次の抽出時間へ
        current_time_sec += interval_sec

    logger.info("\n--- 処理完了 ---")
    logger.info(f"合計 {saved_count} 個のフレームを抽出しました。")

    # リソースを解放（一時ファイルも削除）
    release_video(cap)

if __name__ == '__main__':
    extract_frames_by_interval_in_folder(sys.argv[1], sys.argv[2],interval_ms=500)
