import os
import csv
from datetime import datetime

import numpy as np
import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing._VISIBILITY_THRESHOLD = 0  # 描画するvisibilityの閾値
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

data_path = "data/movie"
# video_names = [f.replace(".mp4", '') for f in os.listdir(data_path)]
video_names = ["maruyamasan_attentive_front"]
save_path = "result/mediapipe/hand"
if not os.path.exists(save_path):
    os.makedirs(save_path)
print(len(video_names))
for i in range(len(video_names)):
    video_name = video_names[i]
    print(f"start detection from {video_name} at {datetime.now()}")
    # 検出結果を保存するディレクトリを作成
    if not os.path.exists(f"{save_path}/{video_name}"):
        os.makedirs(f"{save_path}/{video_name}")

    # 動画ファイル読み込み
    video_path = f"{data_path}/{video_name}.mp4"
    video_data = cv2.VideoCapture(video_path)
    video_width = int(video_data.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_hight = int(video_data.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_fps = video_data.get(cv2.CAP_PROP_FPS)
    print('FPS:', video_fps)
    print('Dimensions:', video_width, "x", video_hight)
    video_frames = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))
    print("VideoFrame:", video_frames)
    readframe = 0

    with open(f"{save_path}/{video_name}/{video_name}_hand.csv", "w") as f:
        writer = csv.writer(f)
        header = ['frame', 'time', ]
        landmarks_header = ['WRIST_x', 'WRIST_y', 'THUMB_CMC_x', 'THUMB_CMC_y', 'THUMB_MCP_x', 'THUMB_MCP_y', 'THUMB_IP_x', 'THUMB_IP_y', 'THUMB_TIP_x', 'THUMB_TIP_y', 'INDEX_FINGER_MCP_x', 'INDEX_FINGER_MCP_y', 'INDEX_FINGER_PIP_x', 'INDEX_FINGER_PIP_y', 'INDEX_FINGER_DIP_x', 'INDEX_FINGER_DIP_y', 'INDEX_FINGER_TIP_x', 'INDEX_FINGER_TIP_y', 'MIDDLE_FINGER_MCP_x', 'MIDDLE_FINGER_MCP_y', 'MIDDLE_FINGER_PIP_x', 'MIDDLE_FINGER_PIP_y', 'MIDDLE_FINGER_DIP_x', 'MIDDLE_FINGER_DIP_y', 'MIDDLE_FINGER_TIP_x', 'MIDDLE_FINGER_TIP_y', 'RING_FINGER_MCP_x', 'RING_FINGER_MCP_y', 'RING_FINGER_PIP_x', 'RING_FINGER_PIP_y', 'RING_FINGER_DIP_x', 'RING_FINGER_DIP_y', 'RING_FINGER_TIP_x', 'RING_FINGER_TIP_y', 'PINKY_MCP_x', 'PINKY_MCP_y', 'PINKY_PIP_x', 'PINKY_PIP_y', 'PINKY_DIP_x', 'PINKY_DIP_y', 'PINKY_TIP_x', 'PINKY_TIP_y']
        for landmark in landmarks_header:
            header.append(landmark + "_r")
        for landmark in landmarks_header:
            header.append(landmark + "_l")
        writer.writerow(header)

        output_videoname = f'{save_path}/{video_name}/{video_name}_hand.mp4'
        output_video = cv2.VideoWriter(output_videoname, cv2.VideoWriter_fourcc(*'MP4V'), video_fps, (video_width, video_hight))

        while readframe < video_frames:
            images = []
            rows = []
            startframe = readframe
            # ファイルが正常に読み込めている間ループすることで、各フレームをリストに格納する.　動画が長い場合メモリ使いすぎて落ちるので、1,000フレームで区切る
            while video_data.isOpened():
                # 1フレームごとに読み込み
                success, image = video_data.read()
                if success:
                    # フレームの画像を追加
                    images.append(image)
                    readframe += 1
                else:
                    break
                if readframe % 1000 == 0:
                    break

            print('Frames Read:', startframe, "~", readframe)

            with mp_hands.Hands(
                model_complexity=1,
                max_num_hands=2,
                min_detection_confidence=0,
            ) as hands:
                # フレームが格納されたリストを使ってループ処理
                for frame, image in enumerate(images):
                    row = np.full_like(header, float("nan"))  # 新しく挿入するデータを作成し、nanで埋める
                    video_time = (startframe + frame) / video_fps
                    row[0] = startframe + frame
                    row[1] = round(video_time, 3)
                    # 画像から骨格位置を検出
                    results_h = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    # 検出結果を動画に描画
                    if results_h.multi_hand_landmarks:
                        for hand_landmarks in results_h.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS
                            )
                    else:
                        rows.append(row)
                        continue
                    right_score = -1
                    num_right = -1
                    left_score = -1
                    num_left = -1
                    for n, tmp in enumerate(results_h.multi_handedness):
                        if tmp.classification[0].label == "Right":
                            if tmp.classification[0].score > right_score:
                                right_score = tmp.classification[0].score
                                num_right = n
                        else:
                            if tmp.classification[0].score > left_score:
                                left_score = tmp.classification[0].score
                                num_left = n
                    if num_right != -1:
                        for i in range(21):
                            row[2 + 2 * i], row[3 + 2 * i] = video_width * results_h.multi_hand_landmarks[num_right].landmark[i].x, video_hight * results_h.multi_hand_landmarks[num_right].landmark[i].y
                    if num_left != -1:
                        for i in range(21):
                            row[44 + 2 * i], row[45 + 2 * i] = video_width * results_h.multi_hand_landmarks[num_left].landmark[i].x, video_hight * results_h.multi_hand_landmarks[num_left].landmark[i].y
                    rows.append(row)
            writer.writerows(rows)
            for image in images:
                output_video.write(image)  # 画像処理したimagesを動画に変換
    output_video.release()
    video_data.release()
    print("finish detection at ", datetime.now())
