"""
    2024/05/19作成
    YOLOの推論結果を用いて，養生テープから最も離れた位置に存在する人形をFocalPuppetとして抽出し，csvと重畳した動画を作成する
    初期位置からの距離だけでなく，人差し指がバウンディングボックスに入っているかどうかでもFocalPuppetを判定する
    2024/03/31編集
    人形の並び順を決定する処理(81行目~)について，映像開始直後の数フレームが暗転している映像があったため，初めて全ての人形が検出されたフレームの情報を用いて処理できるように変更
"""

import os
import csv
import datetime
import cv2
import math

import numpy as np
from ultralytics.engine.results import Results, Boxes
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec


mp_drawing = mp.solutions.drawing_utils
mp_drawing._VISIBILITY_THRESHOLD = 0
mp_holistic = mp.solutions.holistic
# mediapipeで手を描画するための設定
# FocalObjectの検出に利用しなかった情報は青で描画
pose_landmark_style_other = {}
spec_other = DrawingSpec(
    color=(255, 0, 0), thickness=2, circle_radius=2
)
for i in range(21):
    pose_landmark_style_other[i] = spec_other
# 利用した情報はオレンジで描画
pose_landmark_style_focal = {}
spec_focal = DrawingSpec(
    color=(0, 138, 255), thickness=2, circle_radius=2
)
for i in range(21):
    pose_landmark_style_focal[i] = spec_focal

header = ["video_name", "frame", "time", "FocalPuppet", "confidence", "x.left", "y.bottom", "x.right", "y.top", "width", "height", "x.center", "y.center", "distance", "hand"]
data_path = "data/movie"
# video_names = [f.replace(".mp4", '') for f in os.listdir(data_path)]
video_names = ["maruyamasan_attentive_front"]

distance_threshold = 7500  # 人形が持たれていると判定する閾値
label2rl = {0: "right", 1: "left"}
rl2label = {"right": 0, "left": 1}
for video_name in video_names:
    # 解析結果の保存先を指定
    save_path = f"result/focalObject/{distance_threshold}/{video_name}"
    if os.path.exists(f"{save_path}/{video_name}_focalObject.csv"):
        print(f"{video_name} is already processed")
        continue
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    print(video_name, datetime.datetime.now())
    video_path = f"{data_path}/{video_name}.mp4"
    video_data = cv2.VideoCapture(video_path)
    video_fps = video_data.get(cv2.CAP_PROP_FPS)
    video_width = int(video_data.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_data.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_videoname = f'{save_path}/{video_name}_focalObject.mp4'
    output_video = cv2.VideoWriter(output_videoname, cv2.VideoWriter_fourcc(*'MP4V'), video_fps, (video_width, video_height))

    # YOLOの検出結果を読み込む
    with open(f"result/yolo/{video_name}/{video_name}_detect_puppet.csv", "r") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    rows = rows[1:]  # ヘッダーを削除
    # mediapipeの検出結果を読み込む
    with open(f"result/mediapipe/hand/{video_name}/{video_name}_hand.csv", "r") as f:
        reader = csv.reader(f)
        rows_hand = [row for row in reader]
    rows_hand = rows_hand[1:]  # ヘッダーを削除

    # 初期位置から最も離れた人形が掴まれているとして、その人形のバウンディングボックスの情報を記録
    Init_Coords = [[1196, 793], [766, 787], [998, 780]]  # 養生の中点の座標
    Init_Coords_Dict = {}
    label2Puppet = {0: "purple", 1: "yellow", 2: "red"}

    # 開始時点でのバウンディングボックスを用いて、各人形および養生との距離を算出し，人形の並び順を決定
    frame = 0
    while (1):
        row = rows[frame]
        frame += 1
        for Init_Coord in Init_Coords:
            if (isinstance(row[10], float) and math.isnan(row[10])) or (isinstance(row[10], str) and row[10] == "nan"):
                print(f"{frame}フレーム目で人形が検出できていません")
                break
            distance = (Init_Coord[0] - float(row[10]))**2 + (Init_Coord[1] - float(row[11]))**2
            label = 0
            for i in (1, 2):
                if (isinstance(row[10 + 9 * i], float) and math.isnan(row[10 + 9 * i])) or (isinstance(row[10 + 9 * i], str) and row[10 + 9 * i] == "nan"):
                    print(f"{frame}フレーム目で人形が検出できていません")
                    break
                x_center, y_center = float(row[10 + 9 * i]), float(row[11 + 9 * i])
                if distance > (Init_Coord[0] - x_center)**2 + (Init_Coord[1] - y_center)**2:
                    distance = (Init_Coord[0] - x_center)**2 + (Init_Coord[1] - y_center)**2
                    label = i
            if distance > distance_threshold:  # 開始時点で初期位置から離れた位置に人形があった場合，ずっと持たれていると判定されてしまうので，エラーを出力して終了
                print("初期位置に十分近い位置に人形が存在しません")
                exit()
            # 辞書に人形の名前と初期位置の組を保存
            if not label2Puppet[label] in Init_Coords_Dict:
                Init_Coords_Dict[label2Puppet[label]] = Init_Coord
            else:
                print("初期位置の辞書に重複があります")
                exit()
        if len(Init_Coords_Dict) == 3:
            break

    # Init_Coordsを人形のラベルと同じ順番に並べ替える
    Init_Coords = [Init_Coords_Dict["purple"], Init_Coords_Dict["yellow"], Init_Coords_Dict["red"]]

    rows_new = []
    frame = 0
    # FocalObjectの情報だけを抽出
    for j, row in enumerate(rows):
        success, image = video_data.read()
        if success:
            frame += 1
        else:
            print("Failed to read video. 動画のフレーム数がcsvよりも少ない")
            break
        video_time = frame / video_fps
        FocalBox = None
        distance = 0
        cls_hand = -1
        hand_rl = None
        handTouch = False
        row_hand = rows_hand[j]
        index_r_x, index_r_y, index_l_x, index_l_y = float(row_hand[18]), float(row_hand[19]), float(row_hand[60]), float(row_hand[61])
        for i in range(3):
            if (isinstance(row[3 + 9 * i], float) and math.isnan(row[3 + 9 * i])) or (isinstance(row[3 + 9 * i], str) and row[3 + 9 * i] == "nan"):
                continue
            x_left, x_right, y_top, y_bottom = float(row[4 + 9 * i]), float(row[5 + 9 * i]), float(row[6 + 9 * i]), float(row[7 + 9 * i])
            conf = float(row[3 + 9 * i])
            cls = i
            x_center, y_center = float(row[10 + 9 * i]), float(row[11 + 9 * i])
            # いずれかの人差し指がバウンディングボックスに入っているかどうかを確認
            if not math.isnan(index_r_x):
                if x_left < index_r_x < x_right and y_top < index_r_y < y_bottom:
                    if cls_hand == -1:
                        hand_rl = "right"
                        cls_hand = i
                    else:  # 両手で別々の人形に触れている場合，FocalPuppetは無いとする
                        hand_rl = None
                        cls_hand = 4
            if not math.isnan(index_l_x):
                if x_left < index_l_x < x_right and y_top < index_l_y < y_bottom:
                    if cls_hand == -1 or cls_hand == i:
                        if cls_hand == -1:
                            hand_rl = "left"
                            cls_hand = i
                        else:
                            hand_rl = "both"
                    else:
                        hand_rl = None
                        cls_hand = 4
            # 初期位置からの距離が閾値を超える人形がいる場合，その中で最も距離が長い人形をFocalBoxに保存
            if (Init_Coords[i][0] - x_center)**2 + (Init_Coords[i][1] - y_center)**2 > distance_threshold:
                if FocalBox is None:
                    distance = (Init_Coords[i][0] - x_center)**2 + (Init_Coords[i][1] - y_center)**2
                    FocalBox = Boxes(
                        boxes=np.array([x_left, y_bottom, x_right, y_top, conf, cls], dtype=np.float32),
                        orig_shape=(video_height, video_width)
                    )
                else:
                    # 養生からの距離が長い人形のバウンディングボックスの情報をFocalBoxに保存
                    if (Init_Coords[i][0] - x_center)**2 + (Init_Coords[i][1] - y_center)**2 > distance:
                        distance = (Init_Coords[i][0] - x_center)**2 + (Init_Coords[i][1] - y_center)**2
                        FocalBox = Boxes(
                            boxes=np.array([x_left, y_bottom, x_right, y_top, conf, cls], dtype=np.float32),
                            orig_shape=(video_height, video_width)
                        )
        # 初期位置からの距離が閾値を超える人形がない場合，両手の人差し指との距離が閾値以内の人形をFocalPuppetとする
        if FocalBox is None:
            if hand_rl is not None and cls_hand != 4:
                x_left, x_right, y_top, y_bottom = float(row[4 + 9 * cls_hand]), float(row[5 + 9 * cls_hand]), float(row[6 + 9 * cls_hand]), float(row[7 + 9 * cls_hand])
                conf = float(row[3 + 9 * cls_hand])
                cls = cls_hand
                x_center, y_center = float(row[10 + 9 * cls_hand]), float(row[11 + 9 * cls_hand])
                FocalBox = Boxes(
                    boxes=np.array([x_left, y_bottom, x_right, y_top, conf, cls], dtype=np.float32),
                    orig_shape=(video_height, video_width)
                )
                handTouch = True  # 初期位置から離れている人形はなく，加えて手が人形に触れていることを示す
        if not math.isnan(float(row_hand[2])):
            rightHandLandmarks_dicts = [dict(x=float(row_hand[2 * i]) / video_width, y=float(row_hand[1 + 2 * i]) / video_height, visibility=0.5) for i in range(1, 22)]  # x, yは正規化しておく
            rightHand = landmark_pb2.NormalizedLandmarkList(landmark=rightHandLandmarks_dicts)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=rightHand,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=pose_landmark_style_other
            )
        if not math.isnan(float(row_hand[44])):
            leftHandLandmarks_dicts = [dict(x=float(row_hand[2 * i + 42]) / video_width, y=float(row_hand[1 + 2 * i + 42]) / video_height, visibility=0.5) for i in range(1, 22)]  # x, yは正規化しておく
            leftHand = landmark_pb2.NormalizedLandmarkList(landmark=leftHandLandmarks_dicts)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=leftHand,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=pose_landmark_style_other
            )
        row_new = np.full_like(header, float("nan"))
        row_new[0] = video_name
        row_new[1] = frame
        row_new[2] = round(video_time, 3)
        if FocalBox is not None:
            conf = FocalBox.conf.item()
            cls = FocalBox.cls.item()
            x_left, y_bottom, x_right, y_top = FocalBox.xyxy[0].tolist()
            x_center, y_center, width, height = FocalBox.xywh[0].tolist()
            row_new[3] = label2Puppet[cls]
            row_new[4] = round(conf, 2)
            row_new[5], row_new[6], row_new[7], row_new[8] = round(x_left, 2), round(y_bottom, 2), round(x_right, 2), round(y_top, 2)
            row_new[9], row_new[10], row_new[11], row_new[12] = round(width, 2), round(height, 2), round(x_center, 2), round(y_center, 2)
            row_new[13] = distance
            if handTouch:
                row_new[14] = hand_rl
                # 人形のバウンディングボックスに入っている手をオレンジで上書き
                if hand_rl == "right":
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=rightHand,
                        connections=mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=pose_landmark_style_focal
                    )
                elif hand_rl == "left":
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=leftHand,
                        connections=mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=pose_landmark_style_focal
                    )
                elif hand_rl == "both":
                    # 両手をオレンジで上書き
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=rightHand,
                        connections=mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=pose_landmark_style_focal
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=leftHand,
                        connections=mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=pose_landmark_style_focal
                    )
            # バウンディングの描画のため，YOLOの推論結果のインスタンスを作成
            result = Results(
                image,
                path=None,
                names=["purple", "yellow", "red"],  # ラベル名を渡す
                boxes=np.array([x_left, y_bottom, x_right, y_top, conf, cls], dtype=np.float32)
            )
            image = result.plot(img=image)
        rows_new.append(row_new)
        output_video.write(image)

    output_video.release()
    video_data.release()
    with open(f"{save_path}/{video_name}_focalObject.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_new)
