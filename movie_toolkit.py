# copyright (c) 2025 Tomohisa Saisho all rights reserved.
# This software is released under the MIT License.

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

def extract_frames_by_interval_in_folder(input_folder_path, output_folder_path,interval=10):
    if(not os.path.exists(input_folder_path)):
        logger.fatal(f"フォルダ：{input_folder_path}は存在しません。{inspect.currentframe().f_code.co_name}を終了します。")
        return
    
    input_folder_abspath = os.path.abspath(input_folder_path)

    logger.info(f"{input_folder_abspath}から動画ファイルを抽出します。")

    for(video_path) in glob.glob(input_folder_abspath+"/*"):
        try:
            extract_frames_by_interval(video_path=video_path,output_folder_path=output_folder_path,interval=interval)
        except Exception as e:
            logger.error(f"エラーが発生しました： {e}")
        finally:
            logger.info("処理を続行します。")

def extract_frames_by_interval(video_path, output_folder_path,interval=10, ):
    """
    指定された時間間隔で動画からフレームを抽出する

    :param video_path: 動画ファイルのパス
    :param interval: 抽出間隔（秒）
    :param output_folder_path: フレームの保存先ディレクトリ
    """
    # pathlibを使用してパスをオブジェクトとして扱う
    video_path = Path(video_path)
    if not video_path.is_file():
        logger.info(f"エラー: 指定された動画ファイルが見つかりません: {video_path}")
        return

    # VideoCaptureオブジェクトを作成
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.info(f"エラー: 動画ファイルを開けませんでした: {video_path}")
        return

    # ビデオのメタデータを取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # FPSの妥当性を検証
    if fps == 0:
        logger.info("エラー: 動画のFPSを取得できませんでした。ファイルが破損しているか、対応していないコーデックの可能性があります。")
        cap.release()
        return

    logger.info("--- ビデオ情報 ---")
    logger.info(f"ファイルパス: {video_path}")
    logger.info(f"FPS: {fps:.2f}")
    logger.info(f"総フレーム数: {total_frames}")
    logger.info(f"動画の長さ: {duration_sec:.2f} 秒")
    logger.info(f"抽出間隔: {interval} 秒")
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
            current_time_sec += interval
            continue

        # タイムスタンプに基づいたファイル名を生成
        timestamp_str = format_time(current_time_sec)
        output_filename = f"{os.path.basename(video_path)}_{timestamp_str}.jpg"
        output_path = output_folder_path / output_filename

        # フレームを画像ファイルとして保存
        cv2.imwrite(str(output_path), frame)
        saved_count += 1
        logger.info(f"保存しました: {output_path} (フレーム番号: {target_frame_num})")
        
        # 次の抽出時間へ
        current_time_sec += interval

    logger.info("\n--- 処理完了 ---")
    logger.info(f"合計 {saved_count} 個のフレームを抽出しました。")

    # リソースを解放
    cap.release()

if __name__ == '__main__':

   movie_toolkit.extract_frames_by_interval_in_folder(sys.argv[1],sys.argv[2])
