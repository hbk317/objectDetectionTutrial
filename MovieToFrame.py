import os

import cv2


def extract_frames_ToMakeDataset(video_path, train_folder, val_folder, frame_interval=1, start_frame=0, end_frame=0):
    # 動画ファイルの読み込み
    cap = cv2.VideoCapture(video_path)
    if end_frame == 0 or end_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    train_data = 0
    val_data = 0
    key = False

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= frame_count <= end_frame:
            # frame_interval間隔でフレームを保存
            if frame_count % frame_interval == 0:
                output_path = f"{train_folder}/{frame_count:04d}.png"
                cv2.imwrite(output_path, frame)
                train_data += 1
            # 学習用のフレームの中間のフレームを検証用に保存
            # traningデータと同じ数はいらないので半分にする
            if frame_count % frame_interval == frame_interval / 2:
                if key is True:
                    output_path = f"{val_folder}/{frame_count:04d}.png"
                    cv2.imwrite(output_path, frame)
                    val_data += 1
                    key = False
                else:
                    key = True
        frame_count += 1
    cap.release()
    print(f"{train_data} frames extracted for train dataset.")
    print(f"{val_data} frames extracted for val dataset.")


# ビデオファイルのパスと出力フォルダのパスを指定
data_path = "data/movie"
# video_names = [f.replace(".mp4", '') for f in os.listdir(data_path)]  # data_path直下にある全てのmp4ファイルのファイル名を取得
video_names = ["Hagihara_pilot"]
frame_interval = 100

for video_name in video_names:
    video_path = f"{data_path}/{video_name}.mp4"
    train_folder = f"data/dataset/train/{video_name}"
    val_folder = f"data/dataset/val/{video_name}"

    extract_frames_ToMakeDataset(video_path, train_folder, val_folder, frame_interval, start_frame=0, end_frame=0)
