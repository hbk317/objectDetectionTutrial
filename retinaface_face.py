import os
import csv
from datetime import datetime

import cv2
import numpy as np
from retinaface import RetinaFace


# retinafaceの推論結果を描画する関数
def draw_result(image, results):
    facial_area, right_eye, left_eye, nose, mouth_right, mouth_left = results
    cv2.rectangle(image, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (255, 255, 255), thickness=5)
    cv2.circle(image, (int(right_eye[0]), int(right_eye[1])), radius=5, color=(255, 255, 255), thickness=-1)  # thickness=-1の場合塗りつぶす
    cv2.circle(image, (int(left_eye[0]), int(left_eye[1])), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (int(nose[0]), int(nose[1])), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (int(mouth_right[0]), int(mouth_right[1])), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (int(mouth_left[0]), int(mouth_left[1])), radius=5, color=(255, 255, 255), thickness=-1)
    return image


data_path = "data/movie"
# video_names = [f.replace(".mp4", '') for f in os.listdir(data_path)]
video_names = ["maruyamasan_attentive_front"]
print("num videos: ", len(video_names))
print(video_names)
for video_name in video_names:
    print(f"start detection from {video_name} at {datetime.now()}")
    # 検出結果を保存するディレクトリを作成
    save_path = f"result/retinaface/{video_name}"
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

    # 映像毎にFPSが違ったりしたので，念の為出力して確認
    video_fps = video_data.get(cv2.CAP_PROP_FPS)
    video_frames = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))
    print('FPS:', video_fps)
    print('Dimensions:', video_width, "x", video_hight)
    print("VideoFrame:", video_frames)
    readframe = 0
    output_videoname = f'{save_path}/{video_name}_retinaface.mp4'
    output_video = cv2.VideoWriter(output_videoname, cv2.VideoWriter_fourcc(*'MP4V'), video_fps, (video_width, video_hight))

    with open(f"{save_path}/{video_name}_retinaface.csv", "w") as f:
        writer = csv.writer(f)
        header = ['video_name', 'frame', 'time', 'facial_area_x1', 'facial_area_y1', 'facial_area_x2', 'facial_area_y2', 'right_eye_x', 'right_eye_y', 'left_eye_x', 'left_eye_y', 'nose_x', 'nose_y', 'mouth_right_x', 'mouth_right_y', 'mouth_left_x', 'mouth_left_y']
        writer.writerow(header)
        while readframe < video_frames:
            images = []
            rows = []
            startframe = readframe
            # 動画が長い場合メモリ使いすぎて落ちるので、1,000フレーム毎に処理
            while video_data.isOpened():
                success, image = video_data.read()  # imageはBGRで読み込まれる
                if success:
                    images.append(image)
                    readframe += 1
                else:
                    break
                if readframe % 1000 == 0:
                    break

            print('Frames Read:', startframe, "~", readframe)

            for frame, image in enumerate(images):
                results = RetinaFace.detect_faces(image)
                row = np.full_like(header, float("nan"))
                video_time = (startframe + frame) / video_fps
                row[0] = video_name
                row[1] = startframe + frame
                row[2] = round(video_time, 3)
                key = results.keys()
                # 検出される数は１つと仮定
                # 複数検出された場合は別途処理を考えるため，該当箇所を出力（基本的に実験時間外のフレーム）
                for key in results.keys():
                    if key != 'face_1':
                        print("\nkey:", key, "frame:", startframe + frame)
                if 'face_1' in key:
                    identity = results['face_1']
                    facial_area = identity["facial_area"]
                    if not facial_area:
                        continue
                    right_eye = identity["landmarks"]["right_eye"]
                    left_eye = identity["landmarks"]["left_eye"]
                    nose = identity["landmarks"]["nose"]
                    mouth_right = identity["landmarks"]["mouth_right"]
                    mouth_left = identity["landmarks"]["mouth_left"]
                    results = (facial_area, right_eye, left_eye, nose, mouth_right, mouth_left)
                    row[3], row[4], row[5], row[6] = facial_area
                    row[7], row[8] = right_eye
                    row[9], row[10] = left_eye
                    row[11], row[12] = nose
                    row[13], row[14] = mouth_right
                    row[15], row[16] = mouth_left
                else:
                    results = None
                annotated_image = draw_result(image, results) if results else image
                rows.append(row)
                output_video.write(annotated_image)
            writer.writerows(rows)
    output_video.release()
    video_data.release()
    print("finish detection at ", datetime.now())
