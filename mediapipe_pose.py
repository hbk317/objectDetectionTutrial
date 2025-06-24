import os
import csv
from datetime import datetime

import numpy as np
import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing._VISIBILITY_THRESHOLD = 0
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

data_path = "data/movie"
# video_names = [f.replace(".mp4", '') for f in os.listdir(data_path)]
video_names = ["maruyamasan_attentive_front"]
save_dir = "result/mediapipe/pose"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print(len(video_names))
for i in range(len(video_names)):
    video_name = video_names[i]
    print(f"start detection from {video_name} at {datetime.now()}")
    # 検出結果を保存するディレクトリを作成
    save_path = f"{save_dir}/{video_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(f"{save_path} already exists")
        continue

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

    with open(f"{save_path}/{video_name}_pose.csv", "w") as f:
        writer = csv.writer(f)
        header = ['frame', 'time', 'NOSE_visibility', 'NOSE_x', 'NOSE_y', 'LEFT_EYE_INNER_visibility', 'LEFT_EYE_INNER_x', 'LEFT_EYE_INNER_y', 'LEFT_EYE_visibility', 'LEFT_EYE_x', 'LEFT_EYE_y', 'LEFT_EYE_OUTER_visibility', 'LEFT_EYE_OUTER_x', 'LEFT_EYE_OUTER_y', 'RIGHT_EYE_INNER_visibility', 'RIGHT_EYE_INNER_x', 'RIGHT_EYE_INNER_y', 'RIGHT_EYE_visibility', 'RIGHT_EYE_x', 'RIGHT_EYE_y', 'RIGHT_EYE_OUTER_visibility', 'RIGHT_EYE_OUTER_x', 'RIGHT_EYE_OUTER_y', 'LEFT_EAR_visibility', 'LEFT_EAR_x', 'LEFT_EAR_y', 'RIGHT_EAR_visibility', 'RIGHT_EAR_x', 'RIGHT_EAR_y', 'MOUTH_LEFT_visibility', 'MOUTH_LEFT_x', 'MOUTH_LEFT_y', 'MOUTH_RIGHT_visibility', 'MOUTH_RIGHT_x', 'MOUTH_RIGHT_y', 'LEFT_SHOULDER_visibility', 'LEFT_SHOULDER_x', 'LEFT_SHOULDER_y', 'RIGHT_SHOULDER_visibility', 'RIGHT_SHOULDER_x', 'RIGHT_SHOULDER_y', 'LEFT_ELBOW_visibility', 'LEFT_ELBOW_x', 'LEFT_ELBOW_y', 'RIGHT_ELBOW_visibility', 'RIGHT_ELBOW_x', 'RIGHT_ELBOW_y', 'LEFT_WRIST_visibility', 'LEFT_WRIST_x', 'LEFT_WRIST_y', 'RIGHT_WRIST_visibility', 'RIGHT_WRIST_x', 'RIGHT_WRIST_y', 'LEFT_PINKY_visibility', 'LEFT_PINKY_x', 'LEFT_PINKY_y', 'RIGHT_PINKY_visibility', 'RIGHT_PINKY_x', 'RIGHT_PINKY_y', 'LEFT_INDEX_visibility', 'LEFT_INDEX_x', 'LEFT_INDEX_y', 'RIGHT_INDEX_visibility', 'RIGHT_INDEX_x', 'RIGHT_INDEX_y', 'LEFT_THUMB_visibility', 'LEFT_THUMB_x', 'LEFT_THUMB_y', 'RIGHT_THUMB_visibility', 'RIGHT_THUMB_x', 'RIGHT_THUMB_y', 'LEFT_HIP_visibility', 'LEFT_HIP_x', 'LEFT_HIP_y', 'RIGHT_HIP_visibility', 'RIGHT_HIP_x', 'RIGHT_HIP_y', 'LEFT_KNEE_visibility', 'LEFT_KNEE_x', 'LEFT_KNEE_y', 'RIGHT_KNEE_visibility', 'RIGHT_KNEE_x', 'RIGHT_KNEE_y', 'LEFT_ANKLE_visibility', 'LEFT_ANKLE_x', 'LEFT_ANKLE_y', 'RIGHT_ANKLE_visibility', 'RIGHT_ANKLE_x', 'RIGHT_ANKLE_y', 'LEFT_HEEL_visibility', 'LEFT_HEEL_x', 'LEFT_HEEL_y', 'RIGHT_HEEL_visibility', 'RIGHT_HEEL_x', 'RIGHT_HEEL_y', 'LEFT_FOOT_INDEX_visibility', 'LEFT_FOOT_INDEX_x', 'LEFT_FOOT_INDEX_y', 'RIGHT_FOOT_INDEX_visibility', 'RIGHT_FOOT_INDEX_x', 'RIGHT_FOOT_INDEX_y']
        writer.writerow(header)

        output_videoname = f'{save_path}/{video_name}_pose.mp4'
        output_video = cv2.VideoWriter(output_videoname, cv2.VideoWriter_fourcc(*'MP4V'), video_fps, (video_width, video_hight))

        while readframe < video_frames:
            images = []
            rows = []
            startframe = readframe
            # 動画が長い場合メモリ使いすぎて落ちるので、1,000フレーム毎に処理
            while video_data.isOpened():
                success, image = video_data.read()
                if success:
                    images.append(image)
                    readframe += 1
                else:
                    break
                if readframe % 1000 == 0:
                    break

            print('Frames Read:', startframe, "~", readframe)

            with mp_pose.Pose(
                model_complexity=2,
                min_detection_confidence=0.3
            ) as pose:
                for frame, image in enumerate(images):
                    # 画像から骨格位置を検出
                    results_p = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    # 検出結果を動画に描画
                    mp_drawing.draw_landmarks(
                        image,
                        results_p.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    # 検出結果をcsvに書き込む
                    row = np.full_like(header, float("nan"))  # 新しく挿入するデータを作成し、nanで埋める
                    video_time = (startframe + frame) / video_fps
                    row[0] = startframe + frame
                    row[1] = round(video_time, 3)
                    if results_p.pose_landmarks:
                        for index in range(len(mp_pose.PoseLandmark)):
                            row[2 + 3*index], row[3 + 3*index], row[4 + 3*index] = results_p.pose_landmarks.landmark[index].visibility, video_width*results_p.pose_landmarks.landmark[index].x, video_hight*results_p.pose_landmarks.landmark[index].y
                    rows.append(row)
            writer.writerows(rows)
            for image in images:
                output_video.write(image)  # 画像処理したimagesを動画に変換

    output_video.release()
    video_data.release()
    print("finish detection at ", datetime.now())
