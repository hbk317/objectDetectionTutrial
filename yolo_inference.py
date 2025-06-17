import os
import csv

import cv2
import numpy as np
import torch
from ultralytics import YOLO

header = ["video_name", "frame", "time", "purple_confidence", "purple_x.left", "purple_x.right", "purple_y.top", "purple_y.bottom", "purple_width", "purple_height", "purple_x.center", "purple_y.center", "yellow_confidence", "yellow_x.left", "yellow_x.right", "yellow_y.top", "yellow_y.bottom", "yellow_width", "yellow_height", "yellow_x.center", "yellow_y.center", "red_confidence", "red_x.left", "red_x.right", "red_y.top", "red_y.bottom", "red_width", "red_height", "red_x.center", "red_y.center"]

video_name = "Hagihara_pilot"
video_path = f"data/movie/{video_name}.mp4"
video_data = cv2.VideoCapture(video_path)
video_fps = video_data.get(cv2.CAP_PROP_FPS)
video_data.release()
model_path = "runs/detect/n/train_includeSubjectsData/weights/best.pt"
save_path = f"result/yolo/{video_name}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

with torch.no_grad():
    model = YOLO(model_path)
    results = model.predict(source=video_path,
                            data="dataset/training.yaml",
                            imgsz=(1920, 1088),
                            stream=True,  # Trueにするとメモリが小さくても動作しやすいらしい．良いPCならFalseでもOK
                            conf=0.30,  # confの閾値
                            max_det=6,  # バウンディングボックスの最大数
                            save=True,  # Trueならバウンディングボックスを描画した画像が保存される
                            save_frames=True,  # 映像を処理する時にTrueなら、各フレームの結果が保存される
                            project=save_path,  # 結果を保存するディレクトリ名を指定
                            )
frame = 0
rows = []
for result in results:
    video_time = frame / video_fps  # 動画時間を算出
    boxes = result.boxes
    # csvに frame, time と各人形についての x.left, x.right, y.top, y.bottom, width, height, x.center, y.center, confidence　を記入．検出できていない場合はfloat('nan')で埋める
    row = np.full_like(header, float("nan"))
    row[0] = video_name
    row[1] = frame
    row[2] = round(video_time, 3)
    for box in boxes:
        cls = int(box.cls.item())
        if row[10 + 9 * cls] != "nan":  # 一つのクラスに複数のバウンディングボックスがついて検出された場合、confが高い方を採用
            if box.conf.item() <= float(row[10 + 9 * cls]):
                continue
        x_left, y_bottom, x_right, y_top = box.xyxy[0].tolist()
        x_center, y_center, width, height = box.xywh[0].tolist()
        conf = box.conf.item()
        row[3 + 9*cls] = round(conf, 2)
        row[4 + 9*cls], row[5 + 9*cls], row[6 + 9*cls], row[7 + 9*cls] = round(x_left, 2), round(x_right, 2), round(y_bottom, 2), round(y_top, 2)
        row[8 + 9*cls], row[9 + 9*cls], row[10 + 9*cls], row[11 + 9*cls] = round(width, 2), round(height, 2), round(x_center, 2), round(y_center, 2)
    rows.append(row)
    frame += 1
with open(f"{save_path}/{video_name}_detect_puppet.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
