import os
import csv
from datetime import datetime

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from ultralytics.engine.results import Results


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing._VISIBILITY_THRESHOLD = 0  # visibilityによらず描画するように変更
mp_drawing_styles = mp.solutions.drawing_styles

POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 0), (0, 3)])
label2Hand = {0: "LEFT", 1: "RIGHT", 2: "BOTH_LEFT", 3: "BOTH_RIGHT"}
puppet2label = {"purple": 0, "yellow": 1, "red": 2}
Both_frame = 0
frame_threshold = 3
distance_threshold = 25000  # 手とFocalPuppetの距離がこれ以下の時は持っていると判定する．単位はピクセル距離の二乗

data_path = "data/movie"
# video_names = [f.replace(".mp4", '') for f in os.listdir(data_path)]
video_names = ["maruyamasan_attentive_front"]

for video_name in video_names:
    save_path = f"result/focalHand/{distance_threshold}/{video_name}"  # 検出結果を保存するディレクトリを作成
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(f"{save_path}/{video_name}_focalHand.csv"):
        print(f"{video_name} is already processed")
        continue
    print(f"start detection from {video_name} at {datetime.now()}")

    # MediaPipeで検出した結果の読み込み
    with open(f"result/mediapipe/pose/{video_name}/{video_name}_pose.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows_pose = [row for row in reader]
    # FocalObjectの検出結果の読み込み
    with open(f"result/focalObject/7500/{video_name}/{video_name}_focalObject.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows_puppet = [row for row in reader]

    # まずcsvを利用してFocalHandを判定し，結果をindexに格納
    # indexの値とFocalHandの判定結果の関係は以下
    # -1: 該当なし
    # 0: 左手
    # 1: 右手
    # 2: 両手（ただし左手の方が先に触れていた）
    # 3: 両手（ただし右手〃）
    indexes = []
    for i in range(len(rows_pose)):
        row_pose = rows_pose[i]
        row_puppet = rows_puppet[i]
        index = -1
        if row_puppet[3] == "nan":  # FocalObjectが無ければ次の行へ
            Both_frame = 0
            indexes.append(index)
            continue
        x_puppet, y_puppet = float(row_puppet[11]), float(row_puppet[12])  # 人形の座標を取得

        # FocalObjectと近い方の手がどちらなのかを判定
        # それぞれの手と人形との距離を算出
        if row_pose[48] == "nan":
            Both_frame = 0
            indexes.append(index)  # nanの場合は-1とする
            continue
        else:
            distance_left = (float(row_pose[48]) - x_puppet) ** 2 + (float(row_pose[49]) - y_puppet) ** 2

        if row_pose[52] == "nan":
            Both_frame = 0
            indexes.append(index)
            continue
        else:
            distance_right = (float(row_pose[51]) - x_puppet) ** 2 + (float(row_pose[52]) - y_puppet) ** 2

        # 距離が小さい方の座標をプロット
        if distance_left < distance_right:
            index = 0
        if distance_left > distance_right:
            index = 1
        # 一定時間の間、両手と人形との距離が閾値以下ならBothと判定
        if abs(distance_left - distance_right) < distance_threshold:
            Both_frame += 1
            if Both_frame > frame_threshold:  # frame_threshold以上連続で閾値以下だったらBothと判定
                # 両手で掴み始めるまでに掴んでいた方の手を取得
                # 過去10フレームでFocalHandの頻度が多い方を取得
                if Both_frame == frame_threshold + 1:
                    left = 0
                    right = 0
                    if len(indexes) > 10:
                        for j in range(10):
                            if indexes[-j] == 0:
                                left += 1
                            if indexes[-j] == 1:
                                right += 1
                        index = 2 if left > right else 3
                    else:
                        for j in range(len(indexes)):
                            if indexes[-j] == 0:
                                left += 1
                            if indexes[-j] == 1:
                                right += 1
                        index = 2 if left > right else 3
                else:
                    index = indexes[-1]
                # 直前(frame_threshold-1)フレームもBothとして扱う
                for j in range(1, frame_threshold + 1):
                    indexes[-j] = index
        else:
            Both_frame = 0
        indexes.append(index)
    print("length of indexes: ", len(indexes))

    # indexを利用してFocalHandの座標を取得し描画
    # 重畳する動画の読み込み
    video_path = f"data/movie/{video_name}.mp4"
    video_data = cv2.VideoCapture(video_path)
    video_width = int(video_data.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_data.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = video_data.get(cv2.CAP_PROP_FPS)
    print('FPS:', video_fps)
    print('Dimensions:', video_width, "x", video_height)
    video_frames = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))
    print("VideoFrame:", video_frames)

    output_videoname = f'{save_path}/{video_name}_focalHand.mp4'
    output_video = cv2.VideoWriter(output_videoname, cv2.VideoWriter_fourcc(*'MP4V'), video_fps, (video_width, video_height))
    Imagesize = (1080, 1920, 3)
    header = ["video_name", "frame", "time", "FocalHand", "WRIST_visibility", "WRIST_x", "WRIST_y"]
    rows = []
    readframe = 0
    while readframe < video_frames:
        images = []
        row = []
        startframe = readframe
        # 動画が長い場合メモリ使い切って落ちるので、1000フレームごとに処理
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

        for i in range(len(images)):
            row_pose = rows_pose[startframe + i]
            index = indexes[startframe + i]
            # focalObjectを描画
            row_puppet = rows_puppet[startframe + i]
            if row_puppet[3] != "nan":
                x_left, y_bottom, x_right, y_top, conf, cls = float(row_puppet[5]), float(row_puppet[6]), float(row_puppet[7]), float(row_puppet[8]), float(row_puppet[4]), puppet2label[row_puppet[3]]
                result = Results(
                    images[i],
                    path=None,  # path to source file 今回は使わないのでNone
                    names=["purple", "yellow", "red"],  # ラベル名を渡す
                    boxes=np.array([x_left, y_bottom, x_right, y_top, conf, cls], dtype=np.float32)
                )
                images[i] = result.plot(img=images[i])
            image = images[i]
            # 距離が小さい方の座標をプロット
            if index == -1:
                pass
            else:
                # リストのままだと多分typeErrorなので、MediaPipeのデータ型に合わせる  https://qiita.com/Esp-v2/items/9c671ad29a263ce3675e
                left_landmarks_dicts = [dict(x=float(row_pose[48 + 6 * j]) / video_width, y=float(row_pose[49 + 6 * j]) / video_height, visibility=float(row_pose[47 + 6 * j])) for j in range(4)]  # x, yは正規化した値を代入
                leftHand = landmark_pb2.NormalizedLandmarkList(landmark=left_landmarks_dicts)

                right_landmarks_dicts = [dict(x=float(row_pose[51 + 6 * j]) / video_width, y=float(row_pose[52 + 6 * j]) / video_height, visibility=float(row_pose[50 + 6 * j])) for j in range(4)]  # x, yは正規化した値を代入
                rightHand = landmark_pb2.NormalizedLandmarkList(landmark=right_landmarks_dicts)

                # 画像に描画する線分や丸の色と太さを指定
                # FocalHandはオレンジで描画
                pose_landmark_style_focal = {}
                pose_landmark_style_other = {}
                spec_focal = DrawingSpec(
                    color=(0, 138, 255), thickness=2, circle_radius=2
                )
                # Bothだが座標を利用しない（後から添えられた）方の手は青色で描画
                spec_other = DrawingSpec(
                    color=(255, 0, 0), thickness=2, circle_radius=2  # BGR表記のため(255, 0, 0)は青色
                )
                for j in range(4):
                    pose_landmark_style_focal[j] = spec_focal
                    pose_landmark_style_other[j] = spec_other
                if index == 0:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=leftHand,
                        connections=POSE_CONNECTIONS,
                        landmark_drawing_spec=pose_landmark_style_focal
                    )
                if index == 1:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=rightHand,
                        connections=POSE_CONNECTIONS,
                        landmark_drawing_spec=pose_landmark_style_focal
                    )
                if index == 2:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=leftHand,
                        connections=POSE_CONNECTIONS,
                        landmark_drawing_spec=pose_landmark_style_focal
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=rightHand,
                        connections=POSE_CONNECTIONS,
                        landmark_drawing_spec=pose_landmark_style_other
                    )
                if index == 3:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=leftHand,
                        connections=POSE_CONNECTIONS,
                        landmark_drawing_spec=pose_landmark_style_other
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=rightHand,
                        connections=POSE_CONNECTIONS,
                        landmark_drawing_spec=pose_landmark_style_focal
                    )
            row = np.full_like(header, float("nan"))
            video_time = (startframe + i) / video_fps
            row[0] = video_name
            row[1] = startframe + i
            row[2] = round(video_time, 3)
            if not index == -1:
                FocaLHand = label2Hand[index]
                row[3] = FocaLHand
                row[4], row[5], row[6] = row_pose[47 + 3 * (index % 2)], row_pose[48 + 3 * (index % 2)], row_pose[49 + 3 * (index % 2)]
            rows.append(row)
        for image in images:
            output_video.write(image)
    output_video.release()
    video_data.release()
    with open(f"{save_path}/{video_name}_focalHand.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print("finish detection at ", datetime.now())
