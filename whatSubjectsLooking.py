# TobiiデータとYOLO，RetinaFaceの推論結果を利用して，実験参加者が各時刻に注目している物体を解析するコード
# 共有可能なTobiiデータ付きの実験映像が無いため，このコードは実行できません．そのため，このコードはチュートリアルでは扱わず，結果の例だけお見せします

import os
import csv
import math
import cv2
import datetime

import numpy as np


# ボックスと視線データの重複判定を行う関数
def is_overlap(x_tobii, y_tobii, box, radius_pixel):
    x_left, y_top, x_right, y_bottom = box[0], box[1], box[2], box[3]
    # ボックス内部に視線があるかどうか
    if x_left <= x_tobii <= x_right and y_top <= y_tobii <= y_bottom:
        return True
    # ボックスの辺と視野が交差しているかどうか
    elif x_left <= x_tobii <= x_right and y_top - radius_pixel <= y_tobii <= y_bottom + radius_pixel:
        return True
    elif x_left - radius_pixel <= x_tobii <= x_right + radius_pixel and y_top <= y_tobii <= y_bottom:
        return True
    # ボックスの頂点と視野が交差しているかどうか
    elif math.sqrt((x_left - x_tobii) ** 2 + (y_top - y_tobii) ** 2) <= radius_pixel:
        return True
    elif math.sqrt((x_right - x_tobii) ** 2 + (y_top - y_tobii) ** 2) <= radius_pixel:
        return True
    elif math.sqrt((x_left - x_tobii) ** 2 + (y_bottom - y_tobii) ** 2) <= radius_pixel:
        return True
    elif math.sqrt((x_right - x_tobii) ** 2 + (y_bottom - y_tobii) ** 2) <= radius_pixel:
        return True
    return False


# Tobiiのデータを使うための準備
prop_resize = 1 * 1.0 / 2  # 画像のリサイズ比率
# Tobii Glasses2のカメラパラメータ
pixel_width = 1920
pixel_height = 1080
fov = 90  # 公称対角線画角(度)
radius_va = 4  # 注視点を表す円の半径(度)
radius_pixel = int(round((math.sqrt((pixel_height ** 2) + (pixel_width ** 2)) * 0.5 * math.tan(math.radians(radius_va)) / math.tan(math.radians(fov / 2))) * prop_resize))

header = ["video_name", "frame", "pkt_pts_time", "timestanp", "eyeTracker_valid", "Affetto", "purple", "yellow", "red"]
data_path = "data/movie"
video_names = [f.replace(".mp4", '') for f in os.listdir(data_path)]
for video_name in video_names:
    print(video_name, datetime.datetime.now())
    save_path = f"result/{video_name}_overlap.mp4"  # 結果の保存場所
    video_data = cv2.VideoCapture(save_path)
    video_fps = video_data.get(cv2.CAP_PROP_FPS)
    video_width = int(video_data.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_data.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_path = f"result/tobii_overlap/{video_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(f"{video_name} is already processed.")
        continue
    output_videoname = f'{save_path}/{video_name}_overlap.mp4'
    output_video = cv2.VideoWriter(output_videoname, cv2.VideoWriter_fourcc(*'MP4V'), video_fps, (video_width, video_height))

    # YOLOの検出結果を読み込む
    with open(f"result/yolo/{video_name}/{video_name}_detect_puppet.csv", "r") as f:
        reader = csv.reader(f)
        results_yolo = [row for row in reader]
        # 1列目のヘッダーが"video_name"の場合，不要なのでその列を削除
        if results_yolo[0][0] == "video_name":
            tmp = np.array(results_yolo).copy()
            tmp = np.delete(tmp, 0, axis=1)
            results_yolo = tmp.tolist()
        results_yolo = results_yolo[1:]  # ヘッダーを削除
    # retinafaceの検出結果を読み込む
    with open(f"result/retinaface/{video_name}/{video_name}_retinaface.csv", "r") as f:
        reader = csv.reader(f)
        results_retinaface = [row for row in reader]
        results_retinaface = results_retinaface[1:]
    # Tobiiのデータを読み込む
    with open(f"data/tobii_eyeTracker/{video_name}_Tobii.csv", "r") as f:  # 多分ここでエラー出ます．（共有可能なTobiiデータ付きの実験映像が無いため，実行できません）
        reader = csv.reader(f)
        results_tobii = [row for row in reader]
        results_tobii = results_tobii[1:]
    # ffprobeのデータを読み込む（tobiiのデータがサンプリングされた時刻における，映像のフレームを処理するため）
    # ヘッダーは付けていないため，上の処理と少し違う
    with open(f"data/ffprobe/{video_name}.csv", "r") as f:
        reader = csv.reader(f)
        ffprobe_pkt_pts_time = [row for row in reader]
    rows = []
    j = 1  # tobiiのサンプリング開始時間と動画の開始時間がずれている場合，下のアルゴリズムが正しく動作しない事があるので，csv見て調整する．基本的に1で問題ない
    for i in range(len(ffprobe_pkt_pts_time)):
        # TobiiのFPSは動画の2倍あるので，動画のフレーム数と合うように切り出す
        # 加えて，Tobiiのデータは時々欠損するため，欠損している場合は前後のデータを参照して補完する
        pkt_pts_time = int(float(ffprobe_pkt_pts_time[i][5]) * 1_000_000)  # 参照している動画のフレームの時間．tobiiのデータはus単位なので合わせる
        now = abs(float(results_tobii[j][0]) - pkt_pts_time)  # 現在参照している動画のフレームの秒数と，tobiiのデータの秒数の差
        pre = abs(pkt_pts_time - float(results_tobii[j - 1][0]))  # nowで参照したデータの一つ前にサンプリングされたtobiiのデータと動画時間の差
        next = abs(float(results_tobii[j + 1][0]) - pkt_pts_time) if j + 1 < len(results_tobii) else 30_000
        if now < 10_000:  # tobiiのサンプリング周期は若干誤差があるので幅を持たせる
            row = results_tobii[j].copy()
            row.append(pkt_pts_time)
            rows.append(row)
            j += 2
        elif pre < 30_000:
            row = results_tobii[j - 1].copy()
            row.append(pkt_pts_time)
            rows.append(row)
            j += 1
        elif next < 30_000:
            row = results_tobii[j + 1].copy()
            row.append(pkt_pts_time)
            rows.append(row)
            j += 3
        else:
            row = np.full_like(results_tobii[j], '').tolist()
            row[0] = 'nan'
            row.append(pkt_pts_time)
            rows.append(row)  # 空白で埋める．このフレームにおける視線データは無いことになる
    results_tobii = rows

    # それぞれのデータのフレーム数が一致しているか確認
    if len(results_tobii) != len(results_retinaface) or len(results_tobii) != len(results_yolo):
        print("The number of frames in the csv files are different.\n", len(results_tobii), len(results_retinaface), len(results_yolo))
        break
    frame = 0
    results = []
    for i in range(len(results_tobii)):
        success, img = video_data.read()
        if success:
            frame += 1
        else:
            print("Failed to read video. 動画のフレーム数がcsvよりも少ない")
            break

        # 視線データが無いフレームはスキップ
        if results_tobii[i][15] == '' or results_tobii[i][16] == '':
            output_video.write(img)
            results.append([video_name, frame, results_tobii[i][-1] / 1_000_000, results_tobii[i][0], "invalid", False, False, False, False])
            continue

        # 視線と注目物体の座標を読み込む
        x_tobii = int(results_tobii[i][15])
        y_tobii = int(results_tobii[i][16])
        cv2.circle(img, (x_tobii, y_tobii), radius_pixel, (0, 0, 255), thickness=2)
        conf_purple = float(results_yolo[i][2])
        box_purple = [float(results_yolo[i][3]), float(results_yolo[i][5]), float(results_yolo[i][4]), float(results_yolo[i][6])]  # x_left, y_top, x_right, y_bottom
        conf_yellow = float(results_yolo[i][11])
        box_yellow = [float(results_yolo[i][12]), float(results_yolo[i][14]), float(results_yolo[i][13]), float(results_yolo[i][15])]
        conf_red = float(results_yolo[i][20])
        box_red = [float(results_yolo[i][21]), float(results_yolo[i][23]), float(results_yolo[i][22]), float(results_yolo[i][24])]

        # 視線と重複している注目物体を検出
        overlap_purple = False
        overlap_yellow = False
        overlap_red = False
        if not math.isnan(box_purple[0]):
            overlap_purple = is_overlap(x_tobii, y_tobii, box_purple, radius_pixel)
        if not math.isnan(box_yellow[0]):
            overlap_yellow = is_overlap(x_tobii, y_tobii, box_yellow, radius_pixel)
        if not math.isnan(box_red[0]):
            overlap_red = is_overlap(x_tobii, y_tobii, box_red, radius_pixel)
        overlap_box = []
        conf = 0
        if overlap_purple:
            overlap_box = box_purple
            conf = conf_purple
        if overlap_yellow:
            if conf_yellow > conf:
                overlap_box = box_yellow
                conf = conf_yellow
                overlap_purple = False
            else:
                overlap_yellow = False
        if overlap_red:
            if conf_red > conf:
                overlap_box = box_red
                overlap_purple = False
                overlap_yellow = False
            else:
                overlap_red = False
        # 重複している場合は描画
        if overlap_box:
            cv2.rectangle(img, (int(overlap_box[0]), int(overlap_box[1])), (int(overlap_box[2]), int(overlap_box[3])), (255, 255, 255), thickness=5)

        # 視線とAffettoが重複しているか判定
        if results_retinaface[i][3] != 'nan':
            box_Affetto = [int(results_retinaface[i][3]), int(results_retinaface[i][4]), int(results_retinaface[i][5]), int(results_retinaface[i][6])]
            overlap_Affetto = is_overlap(x_tobii, y_tobii, box_Affetto, radius_pixel)
        else:
            overlap_Affetto = False
        # 重複している場合は描画
        if overlap_Affetto:
            cv2.rectangle(img, (box_Affetto[0], box_Affetto[1]), (box_Affetto[2], box_Affetto[3]), (255, 255, 255), thickness=5)

        results.append([video_name, frame, results_tobii[i][-1] / 1_000_000, results_tobii[i][0], "valid", overlap_Affetto, overlap_purple, overlap_yellow, overlap_red])
        output_video.write(img)
    with open(f"{save_path}/{video_name}_overlap.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    output_video.release()
    video_data.release()
